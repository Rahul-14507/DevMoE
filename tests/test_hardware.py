from devmoe.profiler.hardware import get_hardware_profile, HardwareProfile


def test_hardware_profile_basic():
    profile = get_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert profile.ram_total_gb > 0
    assert profile.ram_available_gb > 0
    assert profile.cpu_cores_physical >= 1
    assert profile.cpu_cores_logical >= 1
    assert 0 <= profile.ram_used_percent <= 100
    assert profile.total_compute_memory_gb > 0
