from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from .hardware import HardwareProfile, get_hardware_profile


class SuitabilityStatus(str, Enum):
    OPTIMAL = "optimal"          # Plenty of headroom, blazing fast
    RECOMMENDED = "recommended"  # Fits comfortably in available RAM/VRAM
    TIGHT = "tight"              # Will run, but leaves < 1GB headroom
    EXCEEDS = "exceeds"          # Too large for current hardware


@dataclass
class ModelSpec:
    name: str
    family: str                  # e.g., "gemma", "qwen", "deepseek", "llama"
    param_size_b: float          # In billions, e.g., 9.0
    required_ram_gb: float       # In GB for standard Q4_K_M quant
    context_window_k: int        # Max context in thousands of tokens
    specialization: str          # "code_specialist", "reasoning", "general"
    ollama_tag: str              # exact tag for `ollama run <tag>`


@dataclass
class ModelSuitability:
    spec: ModelSpec
    status: SuitabilityStatus
    est_headroom_gb: float
    reason: str
    recommended_for_fallback: bool


# Curated catalog of developer & coding-focused models
CURATED_MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec(
        name="Qwen 2.5 Coder 1.5B",
        family="qwen",
        param_size_b=1.5,
        required_ram_gb=1.5,
        context_window_k=32,
        specialization="code_specialist",
        ollama_tag="qwen2.5-coder:1.5b",
    ),
    ModelSpec(
        name="DeepSeek Coder 1.3B",
        family="deepseek",
        param_size_b=1.3,
        required_ram_gb=1.4,
        context_window_k=16,
        specialization="code_specialist",
        ollama_tag="deepseek-coder:1.3b",
    ),
    ModelSpec(
        name="Gemma 2 2B",
        family="gemma",
        param_size_b=2.6,
        required_ram_gb=2.2,
        context_window_k=8,
        specialization="general",
        ollama_tag="gemma2:2b",
    ),
    ModelSpec(
        name="Llama 3.2 3B",
        family="llama",
        param_size_b=3.2,
        required_ram_gb=2.8,
        context_window_k=128,
        specialization="general",
        ollama_tag="llama3.2:3b",
    ),
    ModelSpec(
        name="DeepSeek Coder 6.7B",
        family="deepseek",
        param_size_b=6.7,
        required_ram_gb=4.8,
        context_window_k=16,
        specialization="code_specialist",
        ollama_tag="deepseek-coder:6.7b",
    ),
    ModelSpec(
        name="Qwen 2.5 Coder 7B",
        family="qwen",
        param_size_b=7.0,
        required_ram_gb=5.2,
        context_window_k=32,
        specialization="code_specialist",
        ollama_tag="qwen2.5-coder:7b",
    ),
    ModelSpec(
        name="Llama 3.1 8B",
        family="llama",
        param_size_b=8.0,
        required_ram_gb=5.8,
        context_window_k=128,
        specialization="general",
        ollama_tag="llama3.1:8b",
    ),
    ModelSpec(
        name="Gemma 2 9B",
        family="gemma",
        param_size_b=9.2,
        required_ram_gb=6.8,
        context_window_k=8,
        specialization="reasoning",
        ollama_tag="gemma2:9b",
    ),
    ModelSpec(
        name="Qwen 2.5 Coder 14B",
        family="qwen",
        param_size_b=14.0,
        required_ram_gb=9.8,
        context_window_k=32,
        specialization="code_specialist",
        ollama_tag="qwen2.5-coder:14b",
    ),
    ModelSpec(
        name="DeepSeek Coder V2 16B (MoE)",
        family="deepseek",
        param_size_b=16.0,
        required_ram_gb=10.5,
        context_window_k=64,
        specialization="code_specialist",
        ollama_tag="deepseek-coder-v2:16b",
    ),
    ModelSpec(
        name="Gemma 2 27B",
        family="gemma",
        param_size_b=27.2,
        required_ram_gb=17.5,
        context_window_k=8,
        specialization="reasoning",
        ollama_tag="gemma2:27b",
    ),
    ModelSpec(
        name="Qwen 2.5 Coder 32B",
        family="qwen",
        param_size_b=32.0,
        required_ram_gb=22.0,
        context_window_k=32,
        specialization="code_specialist",
        ollama_tag="qwen2.5-coder:32b",
    ),
    ModelSpec(
        name="Llama 3.3 70B",
        family="llama",
        param_size_b=70.0,
        required_ram_gb=42.0,
        context_window_k=128,
        specialization="general",
        ollama_tag="llama3.3:70b",
    ),
]


class MemoryAdvisor:
    """Evaluates local host resources and advises on model suitability & fallback candidates."""

    def __init__(
        self,
        hardware_profile: Optional[HardwareProfile] = None,
        ram_headroom_gb: float = 1.5,
        vram_headroom_gb: float = 0.8,
    ):
        self.profile = hardware_profile or get_hardware_profile()
        self.ram_headroom_gb = ram_headroom_gb
        self.vram_headroom_gb = vram_headroom_gb

    def evaluate_model(self, spec: ModelSpec) -> ModelSuitability:
        """Determine suitability of a model given current hardware state."""
        usable_memory = self.profile.ram_available_gb
        if self.profile.has_gpu and self.profile.primary_vram_free_gb >= spec.required_ram_gb + self.vram_headroom_gb:
            # Model fits completely in VRAM!
            headroom = self.profile.primary_vram_free_gb - spec.required_ram_gb
            return ModelSuitability(
                spec=spec,
                status=SuitabilityStatus.OPTIMAL,
                est_headroom_gb=round(headroom, 2),
                reason=f"Fits 100% in GPU VRAM ({self.profile.primary_vram_free_gb:.1f} GB free). Super-fast GPU execution.",
                recommended_for_fallback=True,
            )

        # CPU RAM evaluation
        safe_ram = max(0.0, usable_memory - self.ram_headroom_gb)
        headroom = safe_ram - spec.required_ram_gb

        if headroom >= 1.5:
            status = SuitabilityStatus.RECOMMENDED
            reason = f"Comfortably fits in Host RAM ({usable_memory:.1f} GB available). Zero swapping."
            rec = True
        elif headroom >= 0.0:
            status = SuitabilityStatus.TIGHT
            reason = f"Runs in Host RAM with ~{headroom:.1f} GB headroom. Close other heavy apps."
            rec = True
        else:
            status = SuitabilityStatus.EXCEEDS
            deficit = abs(headroom)
            reason = f"Requires {spec.required_ram_gb:.1f} GB. Exceeds safe RAM budget by {deficit:.1f} GB."
            rec = False

        return ModelSuitability(
            spec=spec,
            status=status,
            est_headroom_gb=round(headroom, 2),
            reason=reason,
            recommended_for_fallback=rec,
        )

    def get_all_evaluations(self) -> List[ModelSuitability]:
        """Evaluate all models in the curated catalog."""
        return [self.evaluate_model(spec) for spec in CURATED_MODEL_CATALOG]

    def get_highest_param_local_fit(
        self, installed_tags: Optional[List[str]] = None
    ) -> Optional[ModelSpec]:
        """
        Identify the highest parameter local model that safely fits in current memory.
        If installed_tags is provided, restricts search to locally installed models.
        """
        evals = self.get_all_evaluations()
        
        # Sort by parameter size descending
        candidates = sorted(
            [e for e in evals if e.status in (SuitabilityStatus.OPTIMAL, SuitabilityStatus.RECOMMENDED, SuitabilityStatus.TIGHT)],
            key=lambda x: x.spec.param_size_b,
            reverse=True,
        )

        if installed_tags:
            # Match against pulled tags in Ollama
            installed_clean = [t.split(":")[0] for t in installed_tags]
            for c in candidates:
                tag_base = c.spec.ollama_tag.split(":")[0]
                if c.spec.ollama_tag in installed_tags or tag_base in installed_clean:
                    return c.spec

        # If no installed filter or no match, return the best candidate from the catalog
        if candidates:
            return candidates[0].spec

        # Extreme fallback to the smallest model
        return CURATED_MODEL_CATALOG[0]
