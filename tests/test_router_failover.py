import pytest
from devmoe.config import AppConfig
from devmoe.profiler.hardware import HardwareProfile
from devmoe.profiler.advisor import MemoryAdvisor
from devmoe.router.quota_tracker import QuotaTracker
from devmoe.router.engine import DevRouter
from devmoe.providers.manager import ProviderManager


@pytest.mark.asyncio
async def test_quota_tracker_and_fallback():
    config = AppConfig()
    hw = HardwareProfile(
        platform="TestOS",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        ram_used_percent=50.0,
    )
    advisor = MemoryAdvisor(hw)
    tracker = QuotaTracker(default_cooldown_seconds=60)
    mgr = ProviderManager(config, tracker, advisor)
    router = DevRouter(config, mgr, tracker, advisor)

    # Initial availability
    assert tracker.is_provider_available("gemini_gemma_cloud") is True

    # Record 429 rate limit
    tracker.record_rate_limit("gemini_gemma_cloud", status_code=429, error_msg="Rate limit")
    assert tracker.is_provider_available("gemini_gemma_cloud") is False

    # Route a medium query: because gemini is rate-limited, router falls back to local model
    decision = router.route_task([{"role": "user", "content": "Create a REST API with database schema and unit tests"}])
    assert decision.provider_id == "ollama_local"
    # Target should be selected via advisor
    best_fit = advisor.get_highest_param_local_fit()
    assert decision.model == best_fit.ollama_tag
