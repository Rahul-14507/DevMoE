from __future__ import annotations
from typing import Dict, List, Optional, Tuple, AsyncIterator, TYPE_CHECKING
from .base import (
    BaseProvider,
    ProviderResponse,
    ProviderRateLimitError,
    ProviderQuotaExhaustedError,
    ProviderOfflineError,
    ProviderError,
)
from .ollama import OllamaProvider
from .gemini import GeminiGemmaProvider
from .anthropic import AnthropicProvider
from .openai_compat import OpenAICompatibleProvider
from ..config import AppConfig
from ..profiler.advisor import MemoryAdvisor, ModelSpec

if TYPE_CHECKING:
    from ..router.quota_tracker import QuotaTracker


class ProviderManager:
    """Orchestrates all local and curated cloud providers with dynamic failover."""

    def __init__(
        self,
        config: AppConfig,
        quota_tracker: QuotaTracker,
        memory_advisor: MemoryAdvisor,
    ):
        self.config = config
        self.quota_tracker = quota_tracker
        self.memory_advisor = memory_advisor
        self.providers: Dict[str, BaseProvider] = {}
        self._init_providers()

    def _init_providers(self):
        # 1. Local Ollama Provider (Always available with mock fallback if offline)
        self.local_ollama = OllamaProvider(
            base_url=self.config.local.base_url,
            timeout_seconds=self.config.local.timeout_seconds,
            enable_mock_fallback=True,
        )
        self.providers["ollama_local"] = self.local_ollama

        # 2. Curated Free: Google AI Studio (Gemini & Gemma API)
        gemini_cfg = self.config.curated_cloud_providers.get("gemini")
        gemini_url = gemini_cfg.base_url if (gemini_cfg and gemini_cfg.base_url) else "https://generativelanguage.googleapis.com/v1beta/openai"
        self.gemini = GeminiGemmaProvider(base_url=gemini_url)
        self.providers["gemini_gemma_cloud"] = self.gemini

        # 3. Curated Free: Groq Cloud
        groq_cfg = self.config.curated_cloud_providers.get("groq")
        groq_url = groq_cfg.base_url if (groq_cfg and groq_cfg.base_url) else "https://api.groq.com/openai/v1"
        self.groq = OpenAICompatibleProvider(
            provider_id="groq_cloud",
            base_url=groq_url,
            env_key_name=groq_cfg.env_key if groq_cfg else "GROQ_API_KEY",
            is_free_tier=True,
            default_model="qwen-2.5-coder-32b",
        )
        self.providers["groq_cloud"] = self.groq

        # 4. Curated Free: Cerebras Cloud
        cerebras_cfg = self.config.curated_cloud_providers.get("cerebras")
        cerebras_url = cerebras_cfg.base_url if (cerebras_cfg and cerebras_cfg.base_url) else "https://api.cerebras.ai/v1"
        self.cerebras = OpenAICompatibleProvider(
            provider_id="cerebras_cloud",
            base_url=cerebras_url,
            env_key_name=cerebras_cfg.env_key if cerebras_cfg else "CEREBRAS_API_KEY",
            is_free_tier=True,
            default_model="llama3.1-70b",
        )
        self.providers["cerebras_cloud"] = self.cerebras

        # 5. Frontier Cloud: Anthropic Claude
        self.anthropic = AnthropicProvider()
        self.providers["anthropic_claude"] = self.anthropic

        # 6. Frontier Cloud: OpenAI
        openai_cfg = self.config.curated_cloud_providers.get("openai")
        openai_url = openai_cfg.base_url if (openai_cfg and openai_cfg.base_url) else "https://api.openai.com/v1"
        self.openai = OpenAICompatibleProvider(
            provider_id="openai_cloud",
            base_url=openai_url,
            env_key_name=openai_cfg.env_key if openai_cfg else "OPENAI_API_KEY",
            is_free_tier=False,
            default_model="gpt-4o",
        )
        self.providers["openai_cloud"] = self.openai

    def get_configured_providers(self) -> Dict[str, BaseProvider]:
        configured = {}
        for pid, p in self.providers.items():
            if p.is_local:
                configured[pid] = p
            elif hasattr(p, "is_configured") and getattr(p, "is_configured"):
                configured[pid] = p
        return configured

    def get_highest_local_model_target(self) -> str:
        """Dynamically fetch the best local model tag fitting current RAM/VRAM."""
        best_spec = self.memory_advisor.get_highest_param_local_fit()
        return best_spec.ollama_tag if best_spec else "qwen2.5-coder:7b"

    async def execute_with_failover(
        self,
        primary_provider_id: str,
        primary_model: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Executes request on primary provider. If rate limited or quota exhausted:
        1. Logs the rate-limit / quota event in QuotaTracker.
        2. Automatically falls back to the highest parameter local model!
        """
        provider = self.providers.get(primary_provider_id)

        # If primary provider is unconfigured or unavailable, redirect immediately to local fallback
        if not provider or (not provider.is_local and not self.quota_tracker.is_provider_available(primary_provider_id)):
            fallback_model = self.get_highest_local_model_target()
            res = await self.local_ollama.complete(
                messages=messages,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            res.was_fallback = True
            res.fallback_reason = f"Primary provider '{primary_provider_id}' is unconfigured or cooling down from rate limit."
            return res

        try:
            res = await provider.complete(
                messages=messages,
                model=primary_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            self.quota_tracker.record_success(
                primary_provider_id,
                prompt_tokens=res.usage.prompt_tokens,
                completion_tokens=res.usage.completion_tokens,
            )
            return res

        except (ProviderRateLimitError, ProviderQuotaExhaustedError) as e:
            # Caught 429 or quota limit!
            self.quota_tracker.record_rate_limit(
                provider_id=primary_provider_id,
                status_code=getattr(e, "status_code", 429),
                error_msg=str(e),
            )

            # Instant Fallback: Query Memory Advisor for the highest param local model
            best_local_spec = self.memory_advisor.get_highest_param_local_fit()
            fallback_model = best_local_spec.ollama_tag if best_local_spec else "qwen2.5-coder:7b"

            fallback_res = await self.local_ollama.complete(
                messages=messages,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            fallback_res.was_fallback = True
            fallback_res.fallback_reason = (
                f"Rate limit / quota hit on '{primary_provider_id}' ({str(e)}). "
                f"Seamlessly fell back to highest-parameter local model: {best_local_spec.name} ({fallback_model})."
            )
            return fallback_res

        except Exception as e:
            # If any other error occurs on a remote provider, fallback to local
            if not provider.is_local:
                best_local_spec = self.memory_advisor.get_highest_param_local_fit()
                fallback_model = best_local_spec.ollama_tag if best_local_spec else "qwen2.5-coder:7b"
                fallback_res = await self.local_ollama.complete(
                    messages=messages,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                fallback_res.was_fallback = True
                fallback_res.fallback_reason = f"Cloud provider error ({e}). Local fallback activated."
                return fallback_res
            raise
