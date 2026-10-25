import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional
import psutil


@dataclass
class GPUInfo:
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vendor: str = "NVIDIA"
    cuda_available: bool = False


@dataclass
class HardwareProfile:
    platform: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    gpus: List[GPUInfo] = field(default_factory=list)
    has_gpu: bool = False
    primary_vram_free_gb: float = 0.0

    @property
    def total_compute_memory_gb(self) -> float:
        """Total memory available for LLM inference (VRAM + usable RAM)."""
        if self.has_gpu and self.primary_vram_free_gb > 1.0:
            # If dedicated GPU is available, VRAM is primary
            return self.primary_vram_free_gb
        return self.ram_available_gb


def _probe_nvidia_smi() -> List[GPUInfo]:
    """Fallback probe using nvidia-smi command line if available."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode != 0:
            return []
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_mb = float(parts[1])
                free_mb = float(parts[2])
                gpus.append(
                    GPUInfo(
                        name=name,
                        vram_total_gb=round(total_mb / 1024, 2),
                        vram_free_gb=round(free_mb / 1024, 2),
                        vendor="NVIDIA",
                        cuda_available=True,
                    )
                )
        return gpus
    except Exception:
        return []


def _probe_torch_cuda() -> List[GPUInfo]:
    """Probe GPUs via PyTorch if installed and CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        count = torch.cuda.device_count()
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            # Memory allocated vs total
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            free_gb = max(0.0, total_gb - allocated)
            gpus.append(
                GPUInfo(
                    name=name,
                    vram_total_gb=round(total_gb, 2),
                    vram_free_gb=round(free_gb, 2),
                    vendor="NVIDIA",
                    cuda_available=True,
                )
            )
        return gpus
    except Exception:
        return []


def get_hardware_profile() -> HardwareProfile:
    """Retrieve full system hardware specs including RAM, VRAM, and CPU."""
    # RAM inspection via psutil
    vm = psutil.virtual_memory()
    ram_total_gb = round(vm.total / (1024**3), 2)
    ram_available_gb = round(vm.available / (1024**3), 2)
    ram_used_percent = round(vm.percent, 1)

    # CPU inspection
    cpu_physical = psutil.cpu_count(logical=False) or 1
    cpu_logical = psutil.cpu_count(logical=True) or 1

    # GPU inspection: try torch first, then nvidia-smi
    gpus = _probe_torch_cuda()
    if not gpus:
        gpus = _probe_nvidia_smi()

    has_gpu = len(gpus) > 0
    primary_vram_free = gpus[0].vram_free_gb if has_gpu else 0.0

    return HardwareProfile(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        ram_used_percent=ram_used_percent,
        gpus=gpus,
        has_gpu=has_gpu,
        primary_vram_free_gb=primary_vram_free,
    )
