#!/usr/bin/env python
"""
DevMoE Parallel Evaluation & Benchmark Script
Tests Local Ollama Models in parallel with Gemini (Google AI Studio) and Groq APIs.
"""
import os
import time
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from devmoe.config import load_config
from devmoe.profiler import get_hardware_profile, MemoryAdvisor
from devmoe.providers import ProviderManager
from devmoe.router import QuotaTracker, DevRouter

load_dotenv()
console = Console()

TEST_PROMPTS = [
    {
        "title": "Task 1: Regex & Type Hints (Easy)",
        "prompt": "Write a Python regex pattern that matches valid semver version strings (e.g., 1.2.3, 2.0.0-beta.1) with full type annotations.",
    },
    {
        "title": "Task 2: REST API & SQL Pagination (Medium)",
        "prompt": "Write a FastAPI endpoint '/api/v1/projects' that queries PostgreSQL with offset pagination, filter by status, and order by updated_at.",
    },
    {
        "title": "Task 3: Concurrency & Mutex Deadlock Prevention (Complex)",
        "prompt": "Explain and write a thread-safe in-memory cache in Python using reader-writer locks or mutexes to prevent deadlocks and race conditions under 100 concurrent workers.",
    },
]


async def run_single_evaluation(provider_id: str, model: str, prompt: str, manager: ProviderManager):
    """Executes a single model request and tracks latency and tokens."""
    provider = manager.providers.get(provider_id)
    if not provider:
        return {
            "provider": provider_id,
            "model": model,
            "status": "NOT FOUND",
            "latency_ms": 0,
            "tokens": 0,
            "snippet": "Provider not registered in manager.",
        }

    start = time.time()
    try:
        res = await asyncio.wait_for(
            provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.7,
                max_tokens=256,
            ),
            timeout=8.0,
        )
        elapsed_ms = round((time.time() - start) * 1000, 1)
        return {
            "provider": provider_id,
            "model": res.model,
            "status": "SUCCESS" if not res.was_fallback else "FALLBACK",
            "latency_ms": res.latency_ms or elapsed_ms,
            "tokens": res.usage.total_tokens,
            "snippet": res.content.strip().splitlines()[0][:60] + "...",
            "was_fallback": res.was_fallback,
            "fallback_reason": res.fallback_reason,
        }
    except asyncio.TimeoutError:
        return {
            "provider": provider_id,
            "model": model,
            "status": "TIMEOUT",
            "latency_ms": 8000.0,
            "tokens": 0,
            "snippet": "Network timeout after 8s",
        }
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 1)
        return {
            "provider": provider_id,
            "model": model,
            "status": "ERROR",
            "latency_ms": elapsed_ms,
            "tokens": 0,
            "snippet": str(e)[:60],
        }


async def main():
    config = load_config()
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(hw)
    tracker = QuotaTracker()
    manager = ProviderManager(config, tracker, advisor)
    router = DevRouter(config, manager, tracker, advisor)

    console.print(
        Panel.fit(
            "[bold cyan]DevMoE Parallel Benchmark: Local Models vs Gemini vs Groq[/bold cyan]\n"
            f"Hardware: {hw.platform} | RAM: {hw.ram_available_gb} GB free | VRAM: {hw.primary_vram_free_gb} GB",
            border_style="cyan",
        )
    )

    # Detect which local models are installed or fallback
    installed_models = await manager.local_ollama.list_installed_models()
    best_local = advisor.get_highest_param_local_fit(installed_models)
    local_target = best_local.ollama_tag if best_local else "qwen2.5-coder:1.5b"

    # Targets to compare in parallel
    targets = [
        ("ollama_local", "qwen2.5-coder:1.5b", "Local Qwen 1.5B (Active GPU)"),
        ("groq_cloud", "qwen/qwen3.8-27b", "Groq Cloud Qwen 27B (Free Tier)"),
        ("gemini_gemma_cloud", "gemini-3.8-flash", "Google AI Studio (Gemini Free)"),
    ]

    for item in TEST_PROMPTS:
        console.print(f"\n[bold yellow]=== {item['title']} ===[/bold yellow]")
        console.print(f"[dim]Prompt: {item['prompt'][:90]}...[/dim]\n")

        # First, query how the DevMoE intelligent router classifies & decides
        decision = router.route_task([{"role": "user", "content": item["prompt"]}])
        console.print(
            f"[bold green]Router Decision:[/bold green] Score {decision.complexity.score}/100 "
            f"({decision.tier.value.upper()}) -> [cyan]{decision.provider_id}[/cyan] ({decision.model})"
        )

        # Launch parallel requests across all candidates
        console.print("[dim]Executing parallel test calls...[/dim]")
        tasks = [
            run_single_evaluation(t[0], t[1], item["prompt"], manager)
            for t in targets
        ]
        results = await asyncio.gather(*tasks)

        # Render Comparison Table
        table = Table(border_style="blue")
        table.add_column("Provider / Endpoint", style="bold cyan")
        table.add_column("Model", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Latency", justify="right", style="magenta")
        table.add_column("Tokens", justify="right", style="green")
        table.add_column("Output Preview", style="dim")

        for r in results:
            if r["status"] == "SUCCESS":
                status_styled = "[bold green]OK[/bold green]"
            elif r["status"] == "FALLBACK":
                status_styled = "[bold yellow]FALLBACK[/bold yellow]"
            else:
                status_styled = "[bold red]FAIL[/bold red]"

            table.add_row(
                r["provider"],
                r["model"],
                status_styled,
                f"{r['latency_ms']} ms",
                str(r["tokens"]),
                r["snippet"],
            )

        console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
