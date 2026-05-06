"""
=============================================================
 FRAMEWORK 1 — LangGraph  |  Smart Task Executor
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from shared.ollama_client import get_langchain_llm
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()
llm = get_langchain_llm(temperature=0.7)

# ─────────────────────────────────────────────
#  1. SHARED STATE  (typed dict shared by all nodes)
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    user_input:     str
    plan:           Optional[str]           # Planner output
    content:        Optional[str]           # Executor output
    critic_verdict: Optional[str]           # "PASS" or "FAIL:<reason>"
    refined_content:Optional[str]           # Refiner output
    iterations:     int                     # Safety counter
    final_output:   Optional[str]           # Resolved output


MAX_ITERATIONS = 3   # ← Edge-case guard: prevent infinite refine loops

# ─────────────────────────────────────────────
#  2. NODES (each is a pure function: state → state)
# ─────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """
    Planner Agent
    → Breaks the user task into numbered, actionable steps.
    Edge-case handled: if LLM returns an empty plan we inject a fallback.
    """
    console.print(Rule("[bold cyan]PLANNER AGENT[/bold cyan]"))

    prompt = f"""You are a strategic planner.
Given this user task: "{state['user_input']}"

Break it into clear, numbered steps that a writer should follow.
Keep it concise – maximum 5 steps. Output ONLY the numbered list."""

    result = llm.invoke(prompt)
    plan = result.content.strip()

    # Edge-case: empty plan
    if not plan:
        plan = "1. Understand the topic\n2. Write an introduction\n3. Cover key points\n4. Add examples\n5. Write a conclusion"
        console.print("[yellow]Planner returned empty - using fallback plan.[/yellow]")

    console.print(Panel(plan, title="Plan", border_style="cyan"))
    return {**state, "plan": plan}


def executor_node(state: AgentState) -> AgentState:
    """
    Executor Agent
    → Generates content following the plan.
    Edge-case handled: empty/very-short content triggers retry flag.
    """
    console.print(Rule("[bold green]EXECUTOR AGENT[/bold green]"))

    prompt = f"""You are a skilled content writer.
Task: "{state['user_input']}"

Follow this plan:
{state['plan']}

Write a detailed, high-quality response. Be specific and informative."""

    result = llm.invoke(prompt)
    content = result.content.strip()

    # Edge-case: suspiciously short content
    if len(content) < 100:
        console.print(f"[yellow]Executor produced very short content ({len(content)} chars). Marking for refinement.[/yellow]")
        content = content or "[EXECUTOR FAILED TO GENERATE CONTENT]"

    console.print(Panel(content[:500] + ("..." if len(content) > 500 else ""),
                        title="Draft Content", border_style="green"))
    return {**state, "content": content}


def critic_node(state: AgentState) -> AgentState:
    """
    Critic Agent
    → Reviews content and returns PASS or FAIL:<reason>.
    """
    console.print(Rule("[bold red]CRITIC AGENT[/bold red]"))

    # Use refined content if available, otherwise original
    content_to_check = state.get("refined_content") or state.get("content", "")

    prompt = f"""You are a strict quality reviewer.
Evaluate this content for the task: "{state['user_input']}"

CONTENT TO REVIEW:
{content_to_check}

Your verdict MUST be exactly one of:
- "PASS" — if content is comprehensive, accurate, and well-structured
- "FAIL:<specific reason>" — if it needs improvement

Respond with ONLY the verdict, nothing else."""

    result = llm.invoke(prompt)
    verdict = result.content.strip()

    # Normalise – LLM might add extra text
    if "PASS" in verdict.upper() and "FAIL" not in verdict.upper():
        verdict = "PASS"
    elif "FAIL" in verdict.upper():
        # Extract everything from FAIL onwards
        idx = verdict.upper().find("FAIL")
        verdict = verdict[idx:]
        if ":" not in verdict:
            verdict = "FAIL: Content quality insufficient"

    console.print(Panel(verdict,
                        title="Critic Verdict",
                        border_style="red" if verdict.startswith("FAIL") else "green"))
    return {**state, "critic_verdict": verdict}


def refiner_node(state: AgentState) -> AgentState:
    """
    Refiner Agent
    → Fixes issues identified by the Critic.
    """
    console.print(Rule("[bold yellow]REFINER AGENT[/bold yellow]"))

    iterations = state.get("iterations", 0) + 1
    console.print(f"[yellow]Refinement iteration: {iterations}/{MAX_ITERATIONS}[/yellow]")

    content_to_fix = state.get("refined_content") or state.get("content", "")
    reason = state.get("critic_verdict", "unknown issue")

    prompt = f"""You are an expert editor.
The following content was rejected by a critic because: {reason}

ORIGINAL CONTENT:
{content_to_fix}

TASK: "{state['user_input']}"

Rewrite and improve the content to fix the stated issues.
Make it comprehensive, accurate, and well-structured."""

    result = llm.invoke(prompt)
    refined = result.content.strip()

    console.print(Panel(refined[:500] + ("..." if len(refined) > 500 else ""),
                        title="Refined Content", border_style="yellow"))
    return {**state, "refined_content": refined, "iterations": iterations}


# ─────────────────────────────────────────────
#  3. ROUTING FUNCTIONS (conditional edges)
# ─────────────────────────────────────────────

def route_after_critic(state: AgentState) -> str:
    """
    Decision point after Critic:
      • PASS        → END
      • FAIL + within limit → refiner
      • FAIL + limit hit   → END (with warning)
    """
    verdict = state.get("critic_verdict", "FAIL")
    iterations = state.get("iterations", 0)

    if verdict == "PASS":
        console.print("[bold green]Critic PASSED - finalising output.[/bold green]")
        return "finalize"

    if iterations >= MAX_ITERATIONS:
        console.print(f"[bold red]Max iterations ({MAX_ITERATIONS}) reached - exiting loop.[/bold red]")
        return "finalize"

    console.print(f"[bold yellow]Critic FAILED - routing to Refiner (attempt {iterations+1}/{MAX_ITERATIONS}).[/bold yellow]")
    return "refine"


def finalize_node(state: AgentState) -> AgentState:
    """
    Merges the best available content into final_output.
    """
    best = state.get("refined_content") or state.get("content", "No content generated.")
    return {**state, "final_output": best}


# ─────────────────────────────────────────────
#  4. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",  planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic",   critic_node)
    graph.add_node("refiner",  refiner_node)
    graph.add_node("finalize", finalize_node)

    # Linear flow: start → planner → executor → critic
    graph.set_entry_point("planner")
    graph.add_edge("planner",  "executor")
    graph.add_edge("executor", "critic")

    # Conditional routing after critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finalize": "finalize",
            "refine":   "refiner",
        }
    )

    # After refiner → back to critic (loop)
    graph.add_edge("refiner", "critic")

    # finalize → END
    graph.add_edge("finalize", END)

    return graph.compile()


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

def run(user_input: str):
    console.print(Panel(
        f"[bold white]{user_input}[/bold white]",
        title="[bold magenta]LangGraph - Smart Task Executor[/bold magenta]",
        border_style="magenta"
    ))

    initial_state: AgentState = {
        "user_input":     user_input,
        "plan":           None,
        "content":        None,
        "critic_verdict": None,
        "refined_content":None,
        "iterations":     0,
        "final_output":   None,
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    console.print(Rule("[bold magenta]FINAL OUTPUT[/bold magenta]"))
    console.print(Panel(
        final_state["final_output"],
        title="Result",
        border_style="magenta"
    ))
    return final_state


if __name__ == "__main__":
    task = "Create a blog about AI agents and include latest trends"
    run(task)
