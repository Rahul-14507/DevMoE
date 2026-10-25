"""
DevMoE live demo.

Everything printed here is measured on the machine running it: hardware is
probed, installed Ollama models are listed, cloud models are discovered from
each provider's /models endpoint, and prompts are routed and executed for real.

Part 3 injects a rate-limit error to demonstrate failover deterministically.
It is labelled as an injection in the output; it is not a real provider outage.
"""
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from devmoe.config import load_config
from devmoe.profiler import MemoryAdvisor, get_hardware_profile
from devmoe.providers import ProviderManager
from devmoe.providers.base import ProviderRateLimitError
from devmoe.router import DevRouter, QuotaTracker

console = Console()

PROMPTS = [
    ("Easy", "Write a Python regex that matches semantic version strings like 1.2.3 or 2.0.0-beta.1, with type hints."),
    ("Medium", "Write a FastAPI endpoint /projects that reads from PostgreSQL with offset pagination and a pytest test with a mocked session."),
    ("Complex", "Design a thread-safe in-memory cache in Python using reader-writer locks. Explain how you prevent deadlock and race conditions under 100 concurrent workers."),
]


def preview(text: str, n: int = 70) -> str:
    cleaned = text
    if "</thought>" in text:
        after = text.split("</thought>", 1)[1].strip()
        cleaned = after if after else text.replace("<thought>", "").replace("</thought>", "").strip()
    else:
        cleaned = text.replace("<thought>", "").strip()

    for line in cleaned.splitlines():
        l = line.strip()
        if not l or l.startswith("```"):
            continue
        l_clean = l.lstrip("*#- ").strip()
        if l_clean:
            return (l_clean[:n] + "...") if len(l_clean) > n else l_clean
    return (cleaned.strip()[:n] + "...") if len(cleaned.strip()) > n else cleaned.strip()


async def run(router: DevRouter, title: str, prompt: str, max_tokens: int = 500):
    decision = router.route_task([{"role": "user", "content": prompt}])
    res, _ = await router.execute([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return decision, res


async def main():
    config = load_config()
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(hw, config.hardware.ram_headroom_gb, config.hardware.vram_headroom_gb)
    tracker = QuotaTracker(default_cooldown_seconds=config.routing.failover.retry_delay_seconds)
    mgr = ProviderManager(config, tracker, advisor)
    router = DevRouter(config, mgr, tracker, advisor)

    # ---- Part 1: detected environment -------------------------------------------------
    await mgr.refresh_models()
    gpu = ", ".join(f"{g.name} ({g.vram_total_gb} GB, {g.vram_free_gb} GB free)" for g in hw.gpus) or "none detected (CPU only)"
    env = Table(title="1. Detected environment (live, nothing hardcoded)", show_header=False, border_style="cyan")
    env.add_row("Platform", hw.platform)
    env.add_row("RAM", f"{hw.ram_total_gb} GB total, {hw.ram_available_gb} GB available")
    env.add_row("GPU", gpu)
    env.add_row("Ollama models installed", ", ".join(mgr.installed_local) or "none")
    fallback = mgr.get_highest_local_model_target()
    env.add_row("Failover target (largest installed that fits now)", fallback or "none")
    for pid, ids in mgr.model_cache.items():
        fast, strong = mgr.pick_model(pid, "medium"), mgr.pick_model(pid, "complex")
        env.add_row(f"{pid} (discovered {len(ids)} models)", f"medium -> {fast}   complex -> {strong}")
    console.print(env)

    # ---- Part 2: routing + real execution ---------------------------------------------
    table = Table(title="2. Routing and real execution", border_style="blue")
    for col in ("Task", "Score", "Routed to", "Ran on", "Latency", "Output"):
        table.add_column(col)
    for title, prompt in PROMPTS:
        decision, res = await run(router, title, prompt)
        ran = f"{res.provider_id} / {res.model}" + ("  [fallback]" if res.was_fallback else "")
        table.add_row(
            title,
            f"{decision.complexity.score}",
            f"{decision.provider_id} / {decision.model}",
            ran,
            f"{res.latency_ms:.0f} ms",
            preview(res.content),
        )
        if res.was_fallback:
            console.print(f"[yellow]{title}: {res.fallback_reason}[/yellow]")
    console.print(table)

    # ---- Part 3: injected rate limits -------------------------------------------------
    console.print(Panel(
        "Injecting HTTP 429 into every configured cloud provider to show failover.\n"
        "This is a simulated outage, not a real one.",
        title="3. Failover demo (injected)", border_style="yellow"))

    async def always_429(*a, **k):
        raise ProviderRateLimitError("injected for demo", status_code=429, provider_id="demo")

    cloud = [p for p in mgr.providers.values() if not p.is_local and getattr(p, "is_configured", False)]
    originals = [(p, p.complete) for p in cloud]
    for p in cloud:
        p.complete = always_429
    mgr.quota_tracker = QuotaTracker(default_cooldown_seconds=1)  # fresh breaker state
    router.quota_tracker = mgr.quota_tracker
    try:
        _, prompt = PROMPTS[2]
        res, decision = await router.execute([{"role": "user", "content": prompt}], max_tokens=300)
    finally:
        for p, fn in originals:
            p.complete = fn
    console.print(f"Routed to:  {decision.provider_id} / {decision.model}")
    console.print(f"Ran on:     {res.provider_id} / {res.model}  ({res.latency_ms:.0f} ms)")
    console.print(f"Why:        {res.fallback_reason}")
    console.print(f"Answer:     {preview(res.content, 100)}")


if __name__ == "__main__":
    asyncio.run(main())
