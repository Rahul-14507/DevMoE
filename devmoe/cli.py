import argparse
import sys
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import load_config
from .profiler import get_hardware_profile, MemoryAdvisor
from .router import CodeComplexityClassifier, QuotaTracker, DevRouter
from .providers import ProviderManager

console = Console()


def cmd_status(args):
    """Inspect system hardware, memory, and provider connectivity."""
    config = load_config(args.config if hasattr(args, "config") else None)
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(hw)
    tracker = QuotaTracker()
    mgr = ProviderManager(config, tracker, advisor)

    console.print(
        Panel.fit(
            f"[bold cyan]DevMoE System Status[/bold cyan]\n"
            f"Platform: [green]{hw.platform}[/green] | CPU Cores: {hw.cpu_cores_physical} physical, {hw.cpu_cores_logical} logical",
            border_style="cyan",
        )
    )

    # Hardware Table
    hw_table = Table(title="Hardware & Memory Profile", border_style="blue")
    hw_table.add_column("Resource", style="cyan", no_wrap=True)
    hw_table.add_column("Total", style="magenta")
    hw_table.add_column("Available / Free", style="green")
    hw_table.add_column("Utilization", style="yellow")

    hw_table.add_row(
        "Host RAM",
        f"{hw.ram_total_gb} GB",
        f"{hw.ram_available_gb} GB",
        f"{hw.ram_used_percent}%",
    )
    if hw.has_gpu and hw.gpus:
        for g in hw.gpus:
            hw_table.add_row(
                f"GPU ({g.name[:20]})",
                f"{g.vram_total_gb} GB",
                f"{g.vram_free_gb} GB",
                f"{round((1 - g.vram_free_gb/g.vram_total_gb)*100, 1)}%",
            )
    else:
        hw_table.add_row("GPU VRAM", "None (CPU Mode)", "--", "--")

    console.print(hw_table)

    # Best local model
    best_spec = advisor.get_highest_param_local_fit()
    console.print(
        f"\n[bold yellow]Auto-Failover Target Model:[/bold yellow] "
        f"[bold green]{best_spec.name}[/bold green] (tag: `[cyan]ollama run {best_spec.ollama_tag}[/cyan]`) "
        f"— Requires {best_spec.required_ram_gb} GB RAM.\n"
    )

    # Providers Table
    p_table = Table(title="LLM Providers & Curated Free Tiers", border_style="magenta")
    p_table.add_column("Provider ID", style="cyan")
    p_table.add_column("Type", style="yellow")
    p_table.add_column("Curated Free Tier", style="green")
    p_table.add_column("Status", style="bold")

    configured = mgr.get_configured_providers()
    for pid, p in mgr.providers.items():
        is_cfg = pid in configured
        status_text = "[bold green]Ready[/bold green]" if is_cfg else "[dim]Unconfigured (Needs API Key)[/dim]"
        tier_text = "Yes (Indefinite)" if p.is_free_tier else "Frontier / Paid"
        type_text = "Local MoE" if p.is_local else "Cloud"
        p_table.add_row(pid, type_text, tier_text, status_text)

    console.print(p_table)


def cmd_recommend(args):
    """Dynamic local model recommendations based on host memory."""
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(hw)
    evals = advisor.get_all_evaluations()

    console.print(
        Panel.fit(
            f"[bold cyan]Local Model Memory Advisor[/bold cyan]\n"
            f"Available Host RAM: [bold green]{hw.ram_available_gb} GB[/bold green] | "
            f"VRAM: [bold green]{hw.primary_vram_free_gb} GB[/bold green]",
            border_style="cyan",
        )
    )

    table = Table(title="Model Suitability Matrix", border_style="cyan")
    table.add_column("Model Name", style="bold white")
    table.add_column("Params", justify="right")
    table.add_column("RAM Req", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Suitability", justify="center")
    table.add_column("Ollama Command", style="dim")

    for e in evals:
        if e.status.value == "optimal":
            status_style = "[bold green]OPTIMAL[/bold green]"
        elif e.status.value == "recommended":
            status_style = "[green]RECOMMENDED[/green]"
        elif e.status.value == "tight":
            status_style = "[yellow]TIGHT FIT[/yellow]"
        else:
            status_style = "[red]EXCEEDS RAM[/red]"

        table.add_row(
            e.spec.name,
            f"{e.spec.param_size_b}B",
            f"{e.spec.required_ram_gb} GB",
            f"{e.spec.context_window_k}k",
            status_style,
            f"ollama run {e.spec.ollama_tag}",
        )

    console.print(table)


def cmd_route(args):
    """Test routing decision on a prompt."""
    config = load_config(args.config if hasattr(args, "config") else None)
    hw = get_hardware_profile()
    advisor = MemoryAdvisor(hw)
    tracker = QuotaTracker()
    mgr = ProviderManager(config, tracker, advisor)
    router = DevRouter(config, mgr, tracker, advisor)

    prompt = args.prompt
    messages = [{"role": "user", "content": prompt}]
    decision = router.route_task(messages)

    console.print(
        Panel.fit(
            f"[bold cyan]Query:[/bold cyan] {prompt}\n\n"
            f"[bold magenta]Complexity Score:[/bold magenta] [bold yellow]{decision.complexity.score}/100[/bold yellow] ([bold]{decision.tier.value.upper()}[/bold])\n"
            f"[bold green]Routed Destination:[/bold green] [bold cyan]{decision.provider_id}[/bold cyan] -> [bold white]{decision.model}[/bold white]\n"
            f"[bold blue]Decision Rationale:[/bold blue] {decision.reason}\n"
            f"[bold]Air-Gapped Local Mode:[/bold] {decision.is_air_gapped}",
            title="[bold green]DevMoE Routing Trace[/bold green]",
            border_style="green",
        )
    )

    if decision.complexity.signals:
        console.print("[dim]Signals detected:[/dim] " + ", ".join(decision.complexity.signals))


def cmd_serve(args):
    """Launch the DevMoE gateway server and web dashboard."""
    config = load_config(args.config if hasattr(args, "config") else None)
    host = args.host or config.server.host
    port = args.port or config.server.port

    console.print(
        Panel.fit(
            f"[bold cyan]DevMoE Gateway Server[/bold cyan]\n"
            f"API Endpoint: [bold green]http://{host}:{port}/v1[/bold green]\n"
            f"Web Dashboard: [bold green]http://{host}:{port}/dashboard[/bold green]\n"
            f"OpenAI Models: [bold green]http://{host}:{port}/v1/models[/bold green]",
            border_style="cyan",
        )
    )

    from .server.app import create_app
    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        prog="devmoe",
        description="DevMoE - Intelligent Software Engineering LLM Router",
    )
    parser.add_argument("--config", "-c", help="Path to custom config.yaml", default=None)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Run the gateway and web dashboard")
    serve_parser.add_argument("--host", help="Host binding (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, help="Port binding (default: 8080)")

    # status
    subparsers.add_parser("status", help="Inspect hardware, memory, and provider status")

    # recommend
    subparsers.add_parser("recommend", help="Display memory-based model suitability matrix")

    # route
    route_parser = subparsers.add_parser("route", help="Analyze prompt complexity and test routing")
    route_parser.add_argument("prompt", help="Code or task prompt string to evaluate")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "route":
        cmd_route(args)
    else:
        # Default: show help or status
        parser.print_help()


if __name__ == "__main__":
    main()
