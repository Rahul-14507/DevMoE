from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..config import AppConfig, load_config
from ..profiler import get_hardware_profile, MemoryAdvisor
from ..router import DevRouter, QuotaTracker
from ..providers import ProviderManager
from .routes import router as api_router


def create_app(config: AppConfig = None) -> FastAPI:
    """Factory function for FastAPI application."""
    app_config = config or load_config()

    app = FastAPI(
        title="DevMoE - Intelligent LLM Router",
        description="Smart hybrid LLM routing engine for software development blending Local MoE and Curated Free Cloud Tiers.",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Hardware & Memory profiling
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(
        hardware_profile=hw,
        ram_headroom_gb=app_config.hardware.ram_headroom_gb,
        vram_headroom_gb=app_config.hardware.vram_headroom_gb,
    )

    # Quota tracking & circuit breakers
    quota_tracker = QuotaTracker(
        default_cooldown_seconds=app_config.routing.failover.retry_delay_seconds
    )

    # Provider Manager & Dynamic Local Failover
    provider_manager = ProviderManager(
        config=app_config,
        quota_tracker=quota_tracker,
        memory_advisor=advisor,
    )

    # Intelligent Task Complexity Router
    dev_router = DevRouter(
        config=app_config,
        provider_manager=provider_manager,
        quota_tracker=quota_tracker,
        memory_advisor=advisor,
    )

    # Attach instances to app state
    app.state.config = app_config
    app.state.hardware = hw
    app.state.memory_advisor = advisor
    app.state.quota_tracker = quota_tracker
    app.state.provider_manager = provider_manager
    app.state.dev_router = dev_router

    # Register API Router
    app.include_router(api_router)

    # Static assets and Dashboard HTML
    web_dir = Path(__file__).parent.parent / "web"
    static_dir = web_dir / "static"
    templates_dir = web_dir / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    @app.get("/dashboard")
    async def serve_dashboard():
        html_file = templates_dir / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        return {"status": "ok", "message": "DevMoE Gateway active. Visit /v1/models or /api/status."}

    return app
