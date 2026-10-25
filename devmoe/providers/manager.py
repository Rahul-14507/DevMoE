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
        self.model_cache: Dict[str, List[str]] = {}
        self.installed_local: List[str] = []
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

    # ------------------------------------------------------------------
    # Live model discovery
    # ------------------------------------------------------------------
    async def _fetch_cloud_models(self, pid: str, provider: BaseProvider) -> List[str]:
        import httpx

        key = getattr(provider, "api_key", "")
        base = getattr(provider, "base_url", "")
        if not key or not base:
            return []
        if pid == "anthropic_claude":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        else:
            headers = {"Authorization": f"Bearer {key}"}
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                res = await client.get(f"{base}/models", headers=headers)
            if res.status_code != 200:
                return []
            return [m.get("id", "") for m in res.json().get("data", []) if m.get("id")]
        except Exception:
            return []

    async def refresh_models(self) -> Dict[str, List[str]]:
        """Ask every configured provider what it serves right now (cached on the manager)."""
        import asyncio

        self.installed_local = await self.local_ollama.list_installed_models()
        cloud = {
            pid: p
            for pid, p in self.providers.items()
            if not p.is_local and getattr(p, "is_configured", False)
        }
        results = await asyncio.gather(*[self._fetch_cloud_models(pid, p) for pid, p in cloud.items()])
        for pid, ids in zip(cloud.keys(), results):
            self.model_cache[pid] = ids
        return self.model_cache

    def cloud_candidates(self, pid: str, tier: str) -> List[str]:
        """Ranked chat models for a provider/tier. A config pin is honoured only if the provider still serves it."""
        from .discovery import rank_models

        discovered = [m.replace("models/", "") for m in self.model_cache.get(pid, [])]
        ranked = rank_models(discovered, "strong" if tier == "complex" else "fast")
        cfg_key = {
            "gemini_gemma_cloud": "gemini", "groq_cloud": "groq", "cerebras_cloud": "cerebras",
            "anthropic_claude": "anthropic", "openai_cloud": "openai",
        }.get(pid)
        cfg = self.config.curated_cloud_providers.get(cfg_key) if cfg_key else None
        pin = cfg.models.get(tier) if cfg else None
        if pin and (not discovered or pin in discovered):
            ranked = [pin] + [m for m in ranked if m != pin]
        return ranked

    def pick_model(self, pid: str, tier: str) -> Optional[str]:
        cands = self.cloud_candidates(pid, tier)
        return cands[0] if cands else None

    # ------------------------------------------------------------------
    # Local fallback: highest-parameter model that fits NOW and is installed
    # ------------------------------------------------------------------
    def _is_installed(self, tag: str) -> bool:
        return tag in self.installed_local

    def get_highest_local_model_target(self) -> str:
        self.memory_advisor.refresh()
        spec = self.memory_advisor.get_highest_param_local_fit(self.installed_local or None)
        if spec and (not self.installed_local or self._is_installed(spec.ollama_tag)):
            return spec.ollama_tag
        if self.installed_local:
            return self.installed_local[0]
        return spec.ollama_tag if spec else ""

    def get_fast_local_model_target(self) -> str:
        """Smallest installed model that is in the catalog, for cheap/easy tasks."""
        from ..profiler.advisor import CURATED_MODEL_CATALOG

        sizes = {s.ollama_tag: s.param_size_b for s in CURATED_MODEL_CATALOG}
        known = [t for t in self.installed_local if t in sizes]
        if known:
            return min(known, key=lambda t: sizes[t])
        return self.get_highest_local_model_target()

    async def _local_fallback(self, messages, temperature, max_tokens, reason: str, **kwargs) -> ProviderResponse:
        if not self.installed_local:
            self.installed_local = await self.local_ollama.list_installed_models()
        model = self.get_highest_local_model_target()
        res = await self.local_ollama.complete(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        res.was_fallback = True
        res.fallback_reason = f"{reason} Ran on local {model} (largest installed model that fits current memory)."
        return res

    async def execute_with_failover(
        self,
        primary_provider_id: str,
        primary_model: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tier: str = "easy",
        **kwargs
    ) -> ProviderResponse:
        """
        Cloud: try the chosen model, then other live-discovered models on the same provider.
        Only when every candidate fails is the provider circuit-broken and the request sent to
        the largest installed local model that fits current RAM/VRAM.
        """
        import asyncio

        provider = self.providers.get(primary_provider_id)

        if not provider or (not provider.is_local and not self.quota_tracker.is_provider_available(primary_provider_id)):
            return await self._local_fallback(
                messages, temperature, max_tokens,
                f"Provider '{primary_provider_id}' is unconfigured or cooling down.", **kwargs
            )

        if provider.is_local:
            return await provider.complete(
                messages=messages, model=primary_model, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        if primary_provider_id not in self.model_cache:
            await self.refresh_models()

        chain: List[str] = []
        for m in [primary_model] + self.cloud_candidates(primary_provider_id, tier):
            if m and m != "auto" and m not in chain:
                chain.append(m)
        chain = chain[:4] or [None]

        attempts: List[str] = []
        saw_limit = False
        last_status = 429
        for model in chain:
            try:
                res = await asyncio.wait_for(
                    provider.complete(
                        messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
                    ),
                    timeout=20.0,
                )
                self.quota_tracker.record_success(
                    primary_provider_id,
                    prompt_tokens=res.usage.prompt_tokens,
                    completion_tokens=res.usage.completion_tokens,
                )
                if attempts:
                    res.was_fallback = True
                    res.fallback_reason = "Switched model on same provider after: " + "; ".join(attempts)
                return res
            except (ProviderRateLimitError, ProviderQuotaExhaustedError) as e:
                saw_limit = True
                last_status = getattr(e, "status_code", 429) or 429
                attempts.append(f"{model} -> HTTP {last_status}")
            except asyncio.TimeoutError:
                attempts.append(f"{model} -> timeout")
            except ProviderError as e:
                attempts.append(f"{model} -> HTTP {e.status_code or 'error'}")
            except Exception as e:
                attempts.append(f"{model} -> {type(e).__name__}")

        self.quota_tracker.record_rate_limit(
            provider_id=primary_provider_id,
            status_code=last_status if saw_limit else 503,
            error_msg="All candidate models failed: " + "; ".join(attempts),
        )
        return await self._local_fallback(
            messages, temperature, max_tokens,
            f"All {len(chain)} candidate model(s) on '{primary_provider_id}' failed ({'; '.join(attempts)}).",
            **kwargs
        )
