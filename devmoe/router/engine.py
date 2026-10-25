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
            target_model = self.provider_manager.get_highest_local_model_target()
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
            target_model = self.provider_manager.get_highest_local_model_target()
            return RoutingDecision(
                provider_id="ollama_local",
                model=target_model,
                tier=eval_result.tier,
                complexity=eval_result,
                reason="Cloud tiers depleted / rate limited. Auto-air gap mode active: 100% Local LLM routing.",
                is_air_gapped=True,
            )

        # --- ROUTING MATRIX (no model names are hardcoded: providers are asked what they serve) ---
        pm = self.provider_manager

        def usable(pid: str) -> bool:
            p = pm.providers.get(pid)
            return bool(p and getattr(p, "is_configured", False) and self.quota_tracker.is_provider_available(pid))

        def cloud_decision(pid: str, tier: TaskTier, label: str) -> RoutingDecision:
            model = pm.pick_model(pid, tier.value) or "auto"
            return RoutingDecision(
                provider_id=pid,
                model=model,
                tier=tier,
                complexity=eval_result,
                reason=f"{tier.value.capitalize()} task (score {eval_result.score}/100). Routed to {label} ({model}).",
            )

        def local_decision(tier: TaskTier, fast: bool) -> RoutingDecision:
            model = pm.get_fast_local_model_target() if fast else pm.get_highest_local_model_target()
            if fast:
                why = f"Easy task (score {eval_result.score}/100). Routed to smallest installed local model ({model}) at zero cost."
            else:
                why = (
                    f"{tier.value.capitalize()} task (score {eval_result.score}/100) but no cloud provider is usable. "
                    f"Routed to largest installed local model that fits {self.memory_advisor.profile.ram_available_gb:.1f} GB free RAM ({model})."
                )
            return RoutingDecision(provider_id="ollama_local", model=model, tier=tier, complexity=eval_result, reason=why)

        if eval_result.tier == TaskTier.EASY:
            return local_decision(TaskTier.EASY, fast=True)

        if eval_result.tier == TaskTier.MEDIUM:
            for pid, label in (
                ("gemini_gemma_cloud", "Google AI Studio free tier"),
                ("groq_cloud", "Groq free tier"),
                ("cerebras_cloud", "Cerebras free tier"),
            ):
                if usable(pid):
                    return cloud_decision(pid, TaskTier.MEDIUM, label)
            return local_decision(TaskTier.MEDIUM, fast=False)

        for pid, label in (
            ("anthropic_claude", "Anthropic"),
            ("gemini_gemma_cloud", "Google AI Studio"),
            ("openai_cloud", "OpenAI"),
            ("groq_cloud", "Groq free tier"),
            ("cerebras_cloud", "Cerebras free tier"),
        ):
            if usable(pid):
                return cloud_decision(pid, TaskTier.COMPLEX, label)
        return local_decision(TaskTier.COMPLEX, fast=False)

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
            tier=decision.tier.value,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return response, decision
