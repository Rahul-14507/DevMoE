import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Auto-load .env from working directory or project root
load_dotenv()


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: List[str] = ["*"]
    api_key: Optional[str] = None


class ComplexityThresholds(BaseModel):
    easy_max: int = 35
    medium_max: int = 70


class FailoverConfig(BaseModel):
    enabled: bool = True
    on_rate_limit: bool = True
    on_quota_exhausted: bool = True
    fallback_target: str = "highest_param_local_fit"
    auto_air_gap_mode: bool = True
    retry_delay_seconds: int = 60


class RoutingConfig(BaseModel):
    strategy: str = "smart_hybrid"
    complexity_thresholds: ComplexityThresholds = Field(default_factory=ComplexityThresholds)
    failover: FailoverConfig = Field(default_factory=FailoverConfig)


class HardwareConfig(BaseModel):
    ram_headroom_gb: float = 1.5
    vram_headroom_gb: float = 0.8
    preferred_quantization: str = "q4_k_m"


class LocalConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120
    preferred_models: List[str] = [
        "gemma2:9b",
        "qwen2.5-coder:7b",
        "deepseek-coder:6.7b",
        "llama3.2:3b",
        "qwen2.5-coder:1.5b",
    ]


class ProviderEntry(BaseModel):
    enabled: bool = True
    base_url: Optional[str] = None
    env_key: Optional[str] = None
    free_tier: bool = False
    models: Dict[str, str] = Field(default_factory=dict)


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)
    curated_cloud_providers: Dict[str, ProviderEntry] = Field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file and override with environment variables."""
    paths_to_try = []
    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.extend([
        Path("config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
        Path.home() / ".devmoe" / "config.yaml"
    ])

    raw_data = {}
    for p in paths_to_try:
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        raw_data = loaded
                        break
            except Exception as e:
                print(f"[Config] Warning: Failed to read {p}: {e}")

    config = AppConfig(**raw_data) if raw_data else AppConfig()

    # Environment variable overrides
    if os.getenv("OLLAMA_BASE_URL"):
        config.local.base_url = os.getenv("OLLAMA_BASE_URL")
    if os.getenv("DEVMOE_PORT"):
        config.server.port = int(os.getenv("DEVMOE_PORT"))
    if os.getenv("DEVMOE_HOST"):
        config.server.host = os.getenv("DEVMOE_HOST")
    if os.getenv("DEVMOE_API_KEY"):
        config.server.api_key = os.getenv("DEVMOE_API_KEY")

    return config
