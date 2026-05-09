import sys, os
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_TASK = "Create a blog about AI agents and include latest trends"


def main():
    args = sys.argv[1:]

    if not args:
        console.print(
            Panel(
                """[bold]Usage:[/bold]
  python run.py langgraph          [dim]→ Run LangGraph framework[/dim]
  python run.py autogen            [dim]→ Run AutoGen framework[/dim]
  python run.py crewai             [dim]→ Run CrewAI framework[/dim]
  python run.py adk                [dim]→ Run Google ADK framework[/dim]

[bold]Custom task:[/bold]
  python run.py langgraph "Your custom task here"

[bold]Frameworks:[/bold]  LangGraph | AutoGen | CrewAI | ADK
[bold]Model:[/bold]       gemma4:e2b via Ollama (http://localhost:11434)""",
                title="[bold magenta]Multi-Agent Learning Project[/bold magenta]",
                border_style="magenta",
            )
        )
        return

    command = args[0].lower()
    task = args[1] if len(args) > 1 else DEFAULT_TASK

    if command == "langgraph":
        import lg_agent.main as lg

        lg.run(task)

    elif command == "autogen":
        import ag_agent.main as ag

        ag.run(task)

    elif command == "crewai":
        import crew_agent.main as cr

        cr.run(task)

    elif command == "adk":
        import adk_agent.main as adk

        adk.run(task)

    elif command == "compare":
        # pyrefly: ignore [missing-import]
        import compare_all

        compare_all.TASK = task
        compare_all.compare_all()

    elif command == "test":
        import importlib.util

        test_path = os.path.join(
            os.path.dirname(__file__), "test-cases", "edge_case_tests.py"
        )
        spec = importlib.util.spec_from_file_location("edge_case_tests", test_path)
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        tests.run_all_tests()

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Run [bold]python run.py[/bold] for usage.")


if __name__ == "__main__":
    main()
