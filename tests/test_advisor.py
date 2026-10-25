from devmoe.profiler.hardware import HardwareProfile, GPUInfo
from devmoe.profiler.advisor import MemoryAdvisor, SuitabilityStatus, CURATED_MODEL_CATALOG


def test_advisor_evaluation():
    # Synthetic 16 GB RAM profile
    profile = HardwareProfile(
        platform="TestOS",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=16.0,
        ram_available_gb=10.0,
        ram_used_percent=37.5,
        gpus=[],
        has_gpu=False,
        primary_vram_free_gb=0.0,
    )
    advisor = MemoryAdvisor(profile, ram_headroom_gb=1.5)
    evals = advisor.get_all_evaluations()
    assert len(evals) == len(CURATED_MODEL_CATALOG)

    # 1.5B should easily fit in 10GB available RAM
    qwen1_5 = next(e for e in evals if e.spec.ollama_tag == "qwen2.5-coder:1.5b")
    assert qwen1_5.status == SuitabilityStatus.RECOMMENDED

    # 70B requires 42GB, so it must exceed 10GB RAM
    llama70 = next(e for e in evals if e.spec.ollama_tag == "llama3.3:70b")
    assert llama70.status == SuitabilityStatus.EXCEEDS

    # Highest param fit should be selected among fitting models
    best = advisor.get_highest_param_local_fit()
    assert best is not None
    assert best.param_size_b > 1.5
    assert best.param_size_b <= 14.0


def test_advisor_gpu_vram():
    # Synthetic GPU profile with 8GB VRAM
    profile = HardwareProfile(
        platform="TestOS",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=16.0,
        ram_available_gb=4.0,
        ram_used_percent=75.0,
        gpus=[GPUInfo(name="RTX 4070", vram_total_gb=8.0, vram_free_gb=7.0, cuda_available=True)],
        has_gpu=True,
        primary_vram_free_gb=7.0,
    )
    advisor = MemoryAdvisor(profile)
    # A 5GB model like DeepSeek Coder 6.7B should be OPTIMAL because it fits 100% in 7GB VRAM
    evals = advisor.get_all_evaluations()
    ds_coder = next(e for e in evals if e.spec.ollama_tag == "deepseek-coder:6.7b")
    assert ds_coder.status == SuitabilityStatus.OPTIMAL
