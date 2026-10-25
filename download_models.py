#!/usr/bin/env python
"""
DevMoE Model Downloader Helper
Downloads local models via Ollama with live progress bars.
"""
import sys
import shutil
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

MODELS = [
    {
        "tag": "qwen2.5-coder:1.5b",
        "size": "~986 MB",
        "role": "Routine Coding, Syntax & Easy Tasks (Ultra-Fast)",
    },
    {
        "tag": "gemma2:2b",
        "size": "~1.6 GB",
        "role": "Official Google Gemma 2 Local Expert",
    },
    {
        "tag": "deepseek-coder:6.7b",
        "size": "~3.8 GB",
        "role": "Highest-Param VRAM Fit for RTX 3050 6GB GPU (Optimal Failover Target)",
    },
    {
        "tag": "qwen2.5-coder:7b",
        "size": "~4.7 GB",
        "role": "High-Capability 7B Coding Specialist",
    },
]


def check_ollama():
    return shutil.which("ollama") is not None


def main():
    console.print(
        Panel.fit(
            "[bold cyan]DevMoE Local Model Downloader[/bold cyan]\n"
            "Prepares local coding experts optimized for your hardware.",
            border_style="cyan",
        )
    )

    if not check_ollama():
        console.print("[bold red]Ollama is not installed or not in your PATH.[/bold red]\n")
        console.print("To install Ollama on Windows, run:")
        console.print("  [bold yellow]winget install Ollama.Ollama[/bold yellow]\n")
        console.print("Or download the Windows installer directly from:")
        console.print("  [bold cyan]https://ollama.com/download/OllamaSetup.exe[/bold cyan]\n")
        console.print("After installing, open a new terminal, make sure [bold green]ollama serve[/bold green] is running, and re-run this script.")
        return

    console.print("[bold green]Ollama CLI detected![/bold green]\n")
    console.print("Models to pull:")
    for m in MODELS:
        console.print(f" - [bold white]{m['tag']}[/bold white] ({m['size']}): {m['role']}")

    console.print("\nStarting downloads with live progress...\n")

    for m in MODELS:
        tag = m["tag"]
        console.print(f"[bold cyan]>>> Pulling {tag} ({m['size']})...[/bold cyan]")
        try:
            # Run directly inheriting stdio so Ollama's interactive progress bar is displayed
            res = subprocess.run(["ollama", "pull", tag])
            if res.returncode == 0:
                console.print(f"[bold green]Successfully pulled {tag}![/bold green]\n")
            else:
                console.print(f"[yellow]Ollama returned code {res.returncode} for {tag}[/yellow]\n")
        except Exception as e:
            console.print(f"[bold red]Failed to pull {tag}: {e}[/bold red]\n")

    console.print("[bold green]All model downloads completed![/bold green]")
    console.print("You can verify installed models anytime with: [cyan]ollama list[/cyan]")


if __name__ == "__main__":
    main()
