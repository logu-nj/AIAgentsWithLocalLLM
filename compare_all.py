"""
=============================================================
 Cross-Framework Comparison Runner
 Runs the SAME task across all 4 frameworks and compares:
  • Quality   (content length, structure)
  • Latency   (wall-clock time)
  • Stability (did it complete without errors?)
  • Messages  (number of LLM calls)
=============================================================
"""

import sys, os, time, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

console = Console()

TASK = "Create a blog about AI agents and include latest trends"


def run_framework(name: str, run_fn, color: str) -> dict:
    """Run a framework and capture metrics."""
    console.print(Rule(f"[{color}]Running {name}[/{color}]"))

    start = time.time()
    error = None
    output = ""

    try:
        # Capture stdout to measure output
        result = run_fn(TASK)
        if isinstance(result, dict):
            output = result.get("final_output", str(result)) or ""
        else:
            output = str(result) or ""
    except Exception as e:
        error = str(e)
        console.print(f"[red]❌ {name} crashed: {e}[/red]")

    elapsed = time.time() - start

    metrics = {
        "framework":     name,
        "latency_s":     round(elapsed, 1),
        "output_length": len(output),
        "word_count":    len(output.split()) if output else 0,
        "stable":        error is None,
        "error":         error or "None",
    }

    return metrics


def compare_all():
    console.print(Panel(
        f"[bold white]Task: {TASK}[/bold white]\n"
        "[dim]Running across LangGraph → AutoGen → CrewAI → ADK[/dim]",
        title="[bold white]⚡ Cross-Framework Comparison[/bold white]",
        border_style="white"
    ))

    all_metrics = []

    # ── LangGraph ──────────────────────────────────────────
    try:
        import lg_agent.main as lg
        metrics = run_framework("LangGraph", lg.run, "cyan")
        all_metrics.append(metrics)
    except Exception as e:
        console.print(f"[red]LangGraph import error: {e}[/red]")
        all_metrics.append({"framework": "LangGraph", "latency_s": 0, "output_length": 0,
                             "word_count": 0, "stable": False, "error": str(e)})

    # ── AutoGen ────────────────────────────────────────────
    try:
        import ag_agent.main as ag
        metrics = run_framework("AutoGen", ag.run, "blue")
        all_metrics.append(metrics)
    except Exception as e:
        console.print(f"[red]AutoGen import error: {e}[/red]")
        all_metrics.append({"framework": "AutoGen", "latency_s": 0, "output_length": 0,
                             "word_count": 0, "stable": False, "error": str(e)})

    # ── CrewAI ─────────────────────────────────────────────
    try:
        import crew_agent.main as cr
        metrics = run_framework("CrewAI", cr.run, "yellow")
        all_metrics.append(metrics)
    except Exception as e:
        console.print(f"[red]CrewAI import error: {e}[/red]")
        all_metrics.append({"framework": "CrewAI", "latency_s": 0, "output_length": 0,
                             "word_count": 0, "stable": False, "error": str(e)})

    # ── Google ADK ─────────────────────────────────────────
    try:
        import adk_agent.main as adk
        metrics = run_framework("ADK", adk.run, "magenta")
        all_metrics.append(metrics)
    except Exception as e:
        console.print(f"[red]ADK import error: {e}[/red]")
        all_metrics.append({"framework": "ADK", "latency_s": 0, "output_length": 0,
                             "word_count": 0, "stable": False, "error": str(e)})

    # ── Results Table ──────────────────────────────────────
    console.print()
    console.print(Rule("[bold white]📊 COMPARISON RESULTS[/bold white]"))

    table = Table(title=f"Framework Comparison — '{TASK[:50]}...'", border_style="dim")
    table.add_column("Framework",    style="bold", min_width=12)
    table.add_column("Latency (s)",  justify="right", style="cyan")
    table.add_column("Word Count",   justify="right", style="green")
    table.add_column("Stable?",      justify="center")
    table.add_column("Errors",       style="dim", max_width=30)

    for m in all_metrics:
        stable_badge = "✅" if m["stable"] else "❌"
        table.add_row(
            m["framework"],
            str(m["latency_s"]),
            str(m["word_count"]),
            stable_badge,
            m["error"][:30] if m["error"] != "None" else "—",
        )

    console.print(table)

    # ── Qualitative Comparison ─────────────────────────────
    console.print()
    qual_table = Table(title="Qualitative Feature Comparison", border_style="dim")
    qual_table.add_column("Feature",          style="bold")
    qual_table.add_column("LangGraph",        justify="center")
    qual_table.add_column("AutoGen",          justify="center")
    qual_table.add_column("CrewAI",           justify="center")
    qual_table.add_column("ADK",              justify="center")

    features = [
        ("Flow Control",    "Graph edges",     "Conversation",  "Task deps",      "Tool pipeline"),
        ("Loop Handling",   "✅ Controlled",   "⚠️  Risky",     "⚠️  Limited",    "✅ Controlled"),
        ("Debugging",       "✅ Easy",         "❌ Hard",        "⚠️  Medium",     "✅ Easy"),
        ("Flexibility",     "🔵 High",         "🔵 Very High",  "🟡 Medium",      "🟡 Medium"),
        ("Schema Safety",   "⚠️  TypedDict",   "❌ None",        "⚠️  Pydantic",   "✅ Pydantic"),
        ("Production Ready","✅ Yes",          "❌ Not Ideal",   "⚠️  Moderate",   "✅ Strong"),
        ("Chaos Level",     "🟢 Low",          "🔴 High",        "🟡 Medium",      "🟢 Low"),
    ]

    for feat in features:
        qual_table.add_row(*feat)

    console.print(qual_table)

    # ── Key Takeaways ──────────────────────────────────────
    console.print(Panel(
        """[bold cyan]LangGraph[/bold cyan]  → Engineering mindset. Explicit graph = full control.
[bold blue]AutoGen[/bold blue]    → Research/experimentation. Agents converse freely = powerful but chaotic.
[bold yellow]CrewAI[/bold yellow]     → Structured teamwork. Role-based design = professional, less chaos.
[bold magenta]ADK[/bold magenta]        → Production systems. Schema-first, tool-based = reliable, less flexible.""",
        title="💡 Key Takeaways",
        border_style="white"
    ))

    return all_metrics


if __name__ == "__main__":
    compare_all()
