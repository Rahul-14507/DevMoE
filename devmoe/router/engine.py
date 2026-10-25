from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from ..config import AppConfig
from ..profiler.advisor import MemoryAdvisor, ModelSpec
from ..providers.base import ProviderResponse
from .classifier import CodeComplexityClassifier, TaskTier, ComplexityEvaluation
from .quota_tracker import QuotaTracker

if TYPE_CHECKING:
    from ..providers.manager import ProviderManager


@dataclass
class RoutingDecision:
    provider_id: str
    model: str
    tier: TaskTier
    complexity: ComplexityEvaluation
    reason: str
    is_air_gapped: bool = False


class DevRouter:
    """Core intelligent routing engine for software engineering LLM queries."""

    def __init__(
        self,
        config: AppConfig,
        provider_manager: ProviderManager,
        quota_tracker: QuotaTracker,
        memory_advisor: MemoryAdvisor,
    ):
        self.config = config
        self.provider_manager = provider_manager
        self.quota_tracker = quota_tracker
        self.memory_advisor = memory_advisor
        self.classifier = CodeComplexityClassifier(
            easy_threshold=config.routing.complexity_thresholds.easy_max,
            medium_threshold=config.routing.complexity_thresholds.medium_max,
        )

    def route_task(
        self,
        messages: List[Dict[str, str]],
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None,
    ) -> RoutingDecision:
        """Analyze prompt complexity and pick the optimal provider & model."""
        # Extract combined prompt text
        user_prompts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        system_prompts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        prompt_text = "\n".join(user_prompts)
        sys_text = "\n".join(system_prompts)

        # Run task complexity classification
        eval_result = self.classifier.evaluate(prompt_text, system_prompt=sys_text)

        # Check if user forced a specific provider
        if force_provider and force_provider in self.provider_manager.providers:
            return RoutingDecision(
                provider_id=force_provider,
                model=force_model or "default",
                tier=eval_result.tier,
                complexity=eval_result,
                reason=f"User forced provider: {force_provider}",
            )

        # Check if user forced strategy = "local_only"
        if self.config.routing.strategy == "local_only":
            best_local = self.memory_advisor.get_highest_param_local_fit()
            target_model = best_local.ollama_tag if best_local else "qwen2.5-coder:7b"
            return RoutingDecision(
                provider_id="ollama_local",
                model=target_model,
                tier=eval_result.tier,
                complexity=eval_result,
                reason="Routing strategy forced to local_only.",
                is_air_gapped=True,
            )

        # Check cloud availability & auto air-gap mode
        cloud_ids = ["gemini_gemma_cloud", "groq_cloud", "cerebras_cloud", "anthropic_claude", "openai_cloud"]
        all_cloud_down = self.quota_tracker.are_cloud_providers_exhausted(cloud_ids)

        if all_cloud_down and self.config.routing.failover.auto_air_gap_mode:
            best_local = self.memory_advisor.get_highest_param_local_fit()
            target_model = best_local.ollama_tag if best_local else "qwen2.5-coder:7b"
            return RoutingDecision(
                provider_id="ollama_local",
                model=target_model,
                tier=eval_result.tier,
                complexity=eval_result,
                reason="Cloud tiers depleted / rate limited. Auto-air gap mode active: 100% Local LLM routing.",
                is_air_gapped=True,
            )

        # --- ROUTING MATRIX ---

        # 1. Easy Task (Complexity <= 35) -> Local Ollama Model
        if eval_result.tier == TaskTier.EASY:
            # For easy tasks, use lightweight fast local model
            local_model = self.config.local.preferred_models[-1] if self.config.local.preferred_models else "qwen2.5-coder:1.5b"
            return RoutingDecision(
                provider_id="ollama_local",
                model=local_model,
                tier=TaskTier.EASY,
                complexity=eval_result,
                reason=f"Easy task (score {eval_result.score}/100). Routed to fast, zero-cost local model ({local_model}).",
            )

        # 2. Medium Task (Complexity 36 - 70) -> Curated Free Cloud (Gemma / Flash / Groq)
        if eval_result.tier == TaskTier.MEDIUM:
            # Check Gemini / Gemma Free Tier
            if self.provider_manager.gemini.is_configured and self.quota_tracker.is_provider_available("gemini_gemma_cloud"):
                return RoutingDecision(
                    provider_id="gemini_gemma_cloud",
                    model="gemini-1.5-flash",
                    tier=TaskTier.MEDIUM,
                    complexity=eval_result,
                    reason=f"Medium task (score {eval_result.score}/100). Routed to Google AI Studio Gemini 1.5 Flash (Curated Free Tier).",
                )
            # Check Groq Free Tier
            if hasattr(self.provider_manager, "groq") and self.provider_manager.groq.is_configured and self.quota_tracker.is_provider_available("groq_cloud"):
                return RoutingDecision(
                    provider_id="groq_cloud",
                    model="qwen-2.5-coder-32b",
                    tier=TaskTier.MEDIUM,
                    complexity=eval_result,
                    reason=f"Medium task (score {eval_result.score}/100). Routed to Groq Cloud Qwen 2.5 Coder 32B (Curated Free Tier).",
                )
            # Cloud free not configured or cooling down -> Local fallback
            best_local = self.memory_advisor.get_highest_param_local_fit()
            target_model = best_local.ollama_tag if best_local else "qwen2.5-coder:7b"
            return RoutingDecision(
                provider_id="ollama_local",
                model=target_model,
                tier=TaskTier.MEDIUM,
                complexity=eval_result,
                reason=f"Medium task. Cloud free tiers unavailable. Routed to local {best_local.name}.",
            )

        # 3. Complex Task (Complexity > 70) -> Cloud Frontier (Claude 3.5 / Gemini 1.5 Pro / GPT-4o) or Gemma 2 27B
        if eval_result.tier == TaskTier.COMPLEX:
            # Check Anthropic Claude
            if self.provider_manager.anthropic.is_configured and self.quota_tracker.is_provider_available("anthropic_claude"):
                return RoutingDecision(
                    provider_id="anthropic_claude",
                    model="claude-3-5-sonnet-20241022",
                    tier=TaskTier.COMPLEX,
                    complexity=eval_result,
                    reason=f"Complex task (score {eval_result.score}/100). Routed to Anthropic Claude 3.5 Sonnet.",
                )
            # Check Google AI Studio Gemini 1.5 Pro / Gemma 2 27B
            if self.provider_manager.gemini.is_configured and self.quota_tracker.is_provider_available("gemini_gemma_cloud"):
                return RoutingDecision(
                    provider_id="gemini_gemma_cloud",
                    model="gemini-1.5-pro",
                    tier=TaskTier.COMPLEX,
                    complexity=eval_result,
                    reason=f"Complex task (score {eval_result.score}/100). Routed to Gemini 1.5 Pro (Deep Reasoning).",
                )
            # Check OpenAI
            if self.provider_manager.openai.is_configured and self.quota_tracker.is_provider_available("openai_cloud"):
                return RoutingDecision(
                    provider_id="openai_cloud",
                    model="gpt-4o",
                    tier=TaskTier.COMPLEX,
                    complexity=eval_result,
                    reason=f"Complex task (score {eval_result.score}/100). Routed to OpenAI GPT-4o.",
                )

            # If all frontier cloud APIs are missing or rate limited:
            # Fall back to the highest parameter local model that fits in current host RAM/VRAM!
            best_local = self.memory_advisor.get_highest_param_local_fit()
            target_model = best_local.ollama_tag if best_local else "gemma2:9b"
            return RoutingDecision(
                provider_id="ollama_local",
                model=target_model,
                tier=TaskTier.COMPLEX,
                complexity=eval_result,
                reason=(
                    f"Complex task (score {eval_result.score}/100), but cloud keys unavailable or rate limited. "
                    f"Fell back to highest-parameter local model fitting current RAM ({self.memory_advisor.profile.ram_available_gb:.1f} GB free): {best_local.name} ({target_model})."
                ),
            )

    async def execute(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        force_provider: Optional[str] = None,
        force_model: Optional[str] = None,
        **kwargs
    ) -> Tuple[ProviderResponse, RoutingDecision]:
        """Route the prompt and execute with automatic failover."""
        decision = self.route_task(
            messages=messages,
            force_provider=force_provider,
            force_model=force_model,
        )

        response = await self.provider_manager.execute_with_failover(
            primary_provider_id=decision.provider_id,
            primary_model=decision.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return response, decision
