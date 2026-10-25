import time
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..profiler import get_hardware_profile, MemoryAdvisor
from ..router import DevRouter, QuotaTracker, RoutingDecision
from ..providers import ProviderManager

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "devmoe-auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    force_provider: Optional[str] = None


class TestRouteRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = ""


def get_dependencies(request: Request):
    """Retrieve shared instances attached to the app state."""
    app_state = request.app.state
    return {
        "config": app_state.config,
        "dev_router": app_state.dev_router,
        "provider_manager": app_state.provider_manager,
        "quota_tracker": app_state.quota_tracker,
        "memory_advisor": app_state.memory_advisor,
    }


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible models catalog endpoint."""
    deps = get_dependencies(request)
    advisor: MemoryAdvisor = deps["memory_advisor"]
    best_local = advisor.get_highest_param_local_fit()
    best_tag = best_local.ollama_tag if best_local else "qwen2.5-coder:7b"

    models = [
        {
            "id": "devmoe-auto",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "devmoe",
            "description": "Intelligent Smart Router: auto-routes between local Ollama and curated free cloud based on task complexity.",
        },
        {
            "id": "devmoe-local",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "devmoe",
            "description": f"Always routes to the highest-parameter local model fitting host memory ({best_tag}).",
        },
        {
            "id": "devmoe-fast",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "devmoe",
            "description": "Ultra-fast local model for routine tasks, typing, and docstrings.",
        },
        {
            "id": "devmoe-frontier",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "devmoe",
            "description": "Frontier Cloud routing (Claude 3.5 Sonnet / Gemini 1.5 Pro / GPT-4o) with local fallback.",
        },
        {
            "id": "gemini-1.5-pro",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
        },
        {
            "id": "gemini-1.5-flash",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
        },
        {
            "id": "gemma-2-27b-it",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
        },
        {
            "id": best_tag,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama",
        },
    ]
    return {"object": "list", "data": models}


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint with intelligent routing and failover."""
    deps = get_dependencies(request)
    dev_router: DevRouter = deps["dev_router"]
    provider_mgr: ProviderManager = deps["provider_manager"]

    messages_raw = [{"role": m.role, "content": m.content} for m in payload.messages]

    # Map virtual models to router behavior
    force_provider = payload.force_provider
    force_model = None

    if payload.model == "devmoe-local":
        force_provider = "ollama_local"
    elif payload.model == "devmoe-fast":
        force_provider = "ollama_local"
        force_model = "qwen2.5-coder:1.5b"
    elif payload.model == "devmoe-frontier":
        # Force cloud frontier target
        pass
    elif payload.model and payload.model != "devmoe-auto":
        # Check if requested model belongs to a known provider
        if "gemini" in payload.model or "gemma" in payload.model:
            force_provider = "gemini_gemma_cloud"
            force_model = payload.model
        elif "claude" in payload.model:
            force_provider = "anthropic_claude"
            force_model = payload.model
        elif "gpt" in payload.model:
            force_provider = "openai_cloud"
            force_model = payload.model
        elif ":" in payload.model:  # Ollama tag like qwen2.5-coder:7b
            force_provider = "ollama_local"
            force_model = payload.model

    # Determine routing
    decision = dev_router.route_task(
        messages=messages_raw,
        force_provider=force_provider,
        force_model=force_model,
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # Handle streaming request
    if payload.stream:
        async def event_generator():
            try:
                provider = provider_mgr.providers.get(decision.provider_id)
                if not provider:
                    provider = provider_mgr.local_ollama

                async for chunk in provider.stream_complete(
                    messages=messages_raw,
                    model=decision.model,
                    temperature=payload.temperature or 0.7,
                    max_tokens=payload.max_tokens,
                ):
                    chunk_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": decision.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Send final chunk
                done_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": decision.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": decision.model,
                    "choices": [{"index": 0, "delta": {"content": f"\n\n[DevMoE Router Error: {e}]"}, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "x-devmoe-routed-provider": decision.provider_id,
            "x-devmoe-routed-model": decision.model,
            "x-devmoe-complexity-score": str(decision.complexity.score),
            "x-devmoe-complexity-tier": decision.tier.value,
        }
        return StreamingResponse(event_generator(), headers=headers)

    # Synchronous execution with automatic failover
    response = await provider_mgr.execute_with_failover(
        primary_provider_id=decision.provider_id,
        primary_model=decision.model,
        messages=messages_raw,
        temperature=payload.temperature or 0.7,
        max_tokens=payload.max_tokens,
    )

    openai_response = {
        "id": req_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "_devmoe_meta": {
            "routed_provider": decision.provider_id,
            "executed_provider": response.provider_id,
            "complexity_score": decision.complexity.score,
            "complexity_tier": decision.tier.value,
            "signals": decision.complexity.signals,
            "was_fallback": response.was_fallback,
            "fallback_reason": response.fallback_reason,
            "latency_ms": response.latency_ms,
        },
    }

    resp = JSONResponse(content=openai_response)
    resp.headers["x-devmoe-routed-provider"] = response.provider_id
    resp.headers["x-devmoe-complexity-score"] = str(decision.complexity.score)
    resp.headers["x-devmoe-was-fallback"] = "true" if response.was_fallback else "false"
    return resp


@router.get("/api/status")
async def get_system_status(request: Request):
    """System health, hardware state, and provider readiness."""
    deps = get_dependencies(request)
    advisor: MemoryAdvisor = deps["memory_advisor"]
    tracker: QuotaTracker = deps["quota_tracker"]
    provider_mgr: ProviderManager = deps["provider_manager"]

    # Refresh hardware readings
    hw = get_hardware_profile()
    advisor.profile = hw

    # Check local Ollama tags
    installed_models = await provider_mgr.local_ollama.list_installed_models()
    best_local = advisor.get_highest_param_local_fit(installed_models)

    configured_providers = provider_mgr.get_configured_providers()

    return {
        "hardware": {
            "platform": hw.platform,
            "cpu_physical": hw.cpu_cores_physical,
            "cpu_logical": hw.cpu_cores_logical,
            "ram_total_gb": hw.ram_total_gb,
            "ram_available_gb": hw.ram_available_gb,
            "ram_used_percent": hw.ram_used_percent,
            "has_gpu": hw.has_gpu,
            "gpus": [
                {
                    "name": g.name,
                    "vram_total_gb": g.vram_total_gb,
                    "vram_free_gb": g.vram_free_gb,
                }
                for g in hw.gpus
            ],
            "primary_vram_free_gb": hw.primary_vram_free_gb,
        },
        "local_engine": {
            "provider": "ollama",
            "base_url": provider_mgr.local_ollama.base_url,
            "is_alive": await provider_mgr.local_ollama.is_healthy(),
            "installed_models": installed_models,
            "highest_param_fit": best_local.ollama_tag if best_local else "qwen2.5-coder:7b",
            "highest_param_name": best_local.name if best_local else "Qwen 2.5 Coder 7B",
        },
        "providers": {
            pid: {
                "configured": pid in configured_providers,
                "is_local": p.is_local,
                "is_free_tier": p.is_free_tier,
                "is_available": tracker.is_provider_available(pid),
            }
            for pid, p in provider_mgr.providers.items()
        },
        "quota_and_circuit_breakers": tracker.get_status_summary(),
    }


@router.get("/api/recommendations")
async def get_model_recommendations(request: Request):
    """Evaluates host memory and returns dynamic recommendations for all models in catalog."""
    deps = get_dependencies(request)
    advisor: MemoryAdvisor = deps["memory_advisor"]
    evaluations = advisor.get_all_evaluations()

    return {
        "host_ram_available_gb": advisor.profile.ram_available_gb,
        "host_vram_free_gb": advisor.profile.primary_vram_free_gb,
        "has_gpu": advisor.profile.has_gpu,
        "recommendations": [
            {
                "name": e.spec.name,
                "tag": e.spec.ollama_tag,
                "param_size_b": e.spec.param_size_b,
                "required_ram_gb": e.spec.required_ram_gb,
                "context_window_k": e.spec.context_window_k,
                "specialization": e.spec.specialization,
                "status": e.status.value,
                "est_headroom_gb": e.est_headroom_gb,
                "reason": e.reason,
                "pull_command": f"ollama run {e.spec.ollama_tag}",
            }
            for e in evaluations
        ],
    }


@router.post("/api/test-route")
async def test_route_prompt(payload: TestRouteRequest, request: Request):
    """Simulates prompt routing without invoking LLM inference."""
    deps = get_dependencies(request)
    dev_router: DevRouter = deps["dev_router"]

    messages = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    messages.append({"role": "user", "content": payload.prompt})

    decision = dev_router.route_task(messages)

    return {
        "score": decision.complexity.score,
        "tier": decision.tier.value,
        "target_provider": decision.provider_id,
        "target_model": decision.model,
        "reason": decision.reason,
        "is_air_gapped": decision.is_air_gapped,
        "signals": decision.complexity.signals,
        "estimated_tokens": decision.complexity.estimated_tokens,
        "code_block_count": decision.complexity.code_block_count,
    }
