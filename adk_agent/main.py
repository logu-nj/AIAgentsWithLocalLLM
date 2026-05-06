"""
=============================================================
 FRAMEWORK 4 — Google Agent Development Kit (ADK)
              Smart Task Executor
=============================================================
 Flow:  Orchestrator Agent delegates to sub-agents via tools.
        ADK's pipeline-style execution with schema validation.

 Concepts demonstrated:
  • ADK Agent with tools
  • Structured tool schema (Pydantic)
  • Sequential pipeline orchestration
  • Strict schema enforcement
  • Error handling on tool failure

 Edge cases handled:
  • Strict schema failures → Pydantic validation
  • Tool execution errors → try/except with graceful fallback
  • Limited flexibility → explicit tool contracts
=============================================================
"""

import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from pydantic import BaseModel, Field
from shared.ollama_client import raw_generate
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────
#  Pydantic Schemas — Strict I/O Contracts
#  ADK enforces tool schemas strictly.
# ─────────────────────────────────────────────

class PlannerInput(BaseModel):
    user_task: str = Field(..., description="The user's task description")

class PlannerOutput(BaseModel):
    plan: str = Field(..., description="Numbered action plan (4-6 steps)")
    step_count: int = Field(..., ge=1, le=10, description="Number of steps in the plan")

class ExecutorInput(BaseModel):
    user_task: str = Field(..., description="Original user task")
    plan: str = Field(..., description="Plan from the planner agent")

class ExecutorOutput(BaseModel):
    content: str = Field(..., min_length=50, description="Generated content (min 50 chars)")
    word_count: int = Field(..., ge=1, description="Approximate word count")

class CriticInput(BaseModel):
    user_task: str = Field(..., description="Original user task")
    content: str = Field(..., description="Content to review")

class CriticOutput(BaseModel):
    verdict: str = Field(..., description="Either 'PASS' or 'FAIL: <reason>'")
    passed: bool = Field(..., description="True if content passed review")
    reason: Optional[str] = Field(None, description="Reason for failure, if failed")

class RefinerInput(BaseModel):
    user_task: str = Field(..., description="Original user task")
    content: str = Field(..., description="Content to refine")
    critic_reason: str = Field(..., description="Critic's reason for rejection")

class RefinerOutput(BaseModel):
    refined_content: str = Field(..., min_length=50, description="Improved content")
    changes_made: str = Field(..., description="Brief description of changes made")


# ─────────────────────────────────────────────
#  Tool Functions (ADK "tools" = typed functions)
# ─────────────────────────────────────────────

def planner_tool(inp: PlannerInput) -> PlannerOutput:
    """
    ADK Tool: Planner
    Breaks a task into actionable steps.
    Raises ValueError on schema violation (strict ADK behavior).
    """
    prompt = f"""You are a strategic planner.
Task: "{inp.user_task}"

Create a numbered action plan with exactly 5 steps.
Format: 
1. Step one
2. Step two
...

Output ONLY the numbered list."""

    try:
        plan_text = raw_generate(prompt, system="You are a strategic planning expert.")

        if not plan_text or len(plan_text) < 20:
            raise ValueError("Planner returned insufficient output")

        # Count steps
        steps = [l for l in plan_text.strip().split("\n") if l.strip() and l[0].isdigit()]
        step_count = len(steps) if steps else 5

        result = PlannerOutput(plan=plan_text, step_count=step_count)
        console.print(Panel(plan_text, title="📋 Planner Tool Output", border_style="cyan"))
        return result

    except Exception as e:
        console.print(f"[red]❌ Planner Tool Error: {e}[/red]")
        # Schema-compliant fallback
        fallback_plan = "1. Research the topic\n2. Draft the content\n3. Add examples\n4. Structure the output\n5. Finalize"
        return PlannerOutput(plan=fallback_plan, step_count=5)


def executor_tool(inp: ExecutorInput) -> ExecutorOutput:
    """
    ADK Tool: Executor
    Generates content following the plan.
    Validates output length via Pydantic.
    """
    prompt = f"""You are an expert content writer.
Task: "{inp.user_task}"

Follow this plan:
{inp.plan}

Write detailed, high-quality content. Be comprehensive and specific."""

    try:
        content = raw_generate(prompt, system="You are a professional content writer.")

        if not content or len(content) < 50:
            raise ValueError(f"Executor produced insufficient content: {len(content)} chars")

        word_count = len(content.split())
        result = ExecutorOutput(content=content, word_count=word_count)
        console.print(Panel(
            content[:400] + ("..." if len(content) > 400 else ""),
            title=f"📝 Executor Tool Output ({word_count} words)",
            border_style="green"
        ))
        return result

    except Exception as e:
        console.print(f"[red]❌ Executor Tool Error: {e}[/red]")
        # Schema-compliant minimal fallback
        fallback = f"Content about {inp.user_task}. This is a placeholder due to generation failure."
        return ExecutorOutput(content=fallback, word_count=len(fallback.split()))


def critic_tool(inp: CriticInput) -> CriticOutput:
    """
    ADK Tool: Critic
    Validates content against strict schema.
    Output must conform to CriticOutput — no free-form text.
    """
    prompt = f"""You are a strict quality reviewer.
Task being evaluated: "{inp.user_task}"

Content:
{inp.content}

Evaluate and respond with ONLY:
- "PASS" — content is complete, accurate and well-structured
- "FAIL: <reason>" — content has specific issues

Your response MUST start with PASS or FAIL."""

    try:
        verdict_raw = raw_generate(prompt, system="You are a quality assurance expert.")
        verdict_raw = verdict_raw.strip()

        # Parse verdict strictly
        if "PASS" in verdict_raw.upper() and "FAIL" not in verdict_raw.upper():
            result = CriticOutput(verdict="PASS", passed=True, reason=None)
        elif "FAIL" in verdict_raw.upper():
            idx = verdict_raw.upper().find("FAIL")
            fail_text = verdict_raw[idx:]
            reason = fail_text.replace("FAIL:", "").replace("FAIL", "").strip()
            if not reason:
                reason = "Content quality insufficient"
            result = CriticOutput(verdict=f"FAIL: {reason}", passed=False, reason=reason)
        else:
            # Ambiguous response — default to fail (strict ADK behavior)
            result = CriticOutput(
                verdict="FAIL: Ambiguous critic response",
                passed=False,
                reason="Could not determine verdict from critic response"
            )

        color = "green" if result.passed else "red"
        console.print(Panel(result.verdict, title="⚖️  Critic Tool Output", border_style=color))
        return result

    except Exception as e:
        console.print(f"[red]❌ Critic Tool Error: {e}[/red]")
        return CriticOutput(verdict="FAIL: Tool execution error", passed=False, reason=str(e))


def refiner_tool(inp: RefinerInput) -> RefinerOutput:
    """
    ADK Tool: Refiner
    Fixes content based on critic schema-validated feedback.
    """
    prompt = f"""You are an expert content editor.
Task: "{inp.user_task}"
Critic's reason for rejection: "{inp.critic_reason}"

Original content:
{inp.content}

Rewrite and improve the content to address ALL stated issues.
Then briefly describe what changes you made (1-2 sentences)."""

    try:
        raw_output = raw_generate(prompt, system="You are an expert content refinement specialist.")

        # Try to extract the "changes" section if model includes it
        lines = raw_output.strip().split("\n")
        changes_line = ""
        content_lines = []

        for line in lines:
            if any(kw in line.lower() for kw in ["change", "improve", "fix", "update", "revision"]):
                if len(line) < 200:  # Likely a summary line
                    changes_line = line
                else:
                    content_lines.append(line)
            else:
                content_lines.append(line)

        refined = "\n".join(content_lines).strip() or raw_output
        changes = changes_line or f"Addressed: {inp.critic_reason[:100]}"

        if len(refined) < 50:
            refined = raw_output  # Use full output as fallback

        result = RefinerOutput(refined_content=refined, changes_made=changes)
        console.print(Panel(
            refined[:400] + ("..." if len(refined) > 400 else ""),
            title="✨ Refiner Tool Output",
            border_style="yellow"
        ))
        return result

    except Exception as e:
        console.print(f"[red]❌ Refiner Tool Error: {e}[/red]")
        fallback = inp.content + "\n\n[Note: Refinement failed, showing original content]"
        return RefinerOutput(refined_content=fallback, changes_made=f"Refinement failed: {e}")


# ─────────────────────────────────────────────
#  ADK Orchestrator
#  In real ADK this would be an Agent with tools registered.
#  Here we simulate ADK's structured pipeline execution.
# ─────────────────────────────────────────────

MAX_REFINEMENTS = 3

def run(user_input: str):
    console.print(Panel(
        f"[bold white]{user_input}[/bold white]",
        title="[bold magenta]🚀 Google ADK — Smart Task Executor[/bold magenta]",
        border_style="magenta"
    ))

    # ── Tool 1: Planner ──────────────────────────────
    console.print(Rule("[bold cyan]🔧 TOOL CALL: planner_tool[/bold cyan]"))
    plan_input = PlannerInput(user_task=user_input)
    plan_output: PlannerOutput = planner_tool(plan_input)

    # ── Tool 2: Executor ─────────────────────────────
    console.print(Rule("[bold green]🔧 TOOL CALL: executor_tool[/bold green]"))
    exec_input = ExecutorInput(user_task=user_input, plan=plan_output.plan)
    exec_output: ExecutorOutput = executor_tool(exec_input)

    # ── Tool 3: Critic ───────────────────────────────
    console.print(Rule("[bold red]🔧 TOOL CALL: critic_tool[/bold red]"))
    critic_input = CriticInput(user_task=user_input, content=exec_output.content)
    critic_output: CriticOutput = critic_tool(critic_input)

    current_content = exec_output.content
    final_content = current_content

    # ── Conditional: Refiner Loop ─────────────────────
    if not critic_output.passed:
        console.print(Rule("[bold yellow]🔄 REFINEMENT PIPELINE[/bold yellow]"))

        for attempt in range(1, MAX_REFINEMENTS + 1):
            console.print(f"[yellow]🔧 TOOL CALL: refiner_tool (attempt {attempt}/{MAX_REFINEMENTS})[/yellow]")

            refiner_input = RefinerInput(
                user_task=user_input,
                content=current_content,
                critic_reason=critic_output.reason or "Quality insufficient",
            )

            # Edge-case: Tool execution error handled inside refiner_tool
            refiner_output: RefinerOutput = refiner_tool(refiner_input)

            # Re-critique with strict schema
            console.print(f"[red]🔧 TOOL CALL: critic_tool (re-evaluation {attempt})[/red]")
            re_critic_input = CriticInput(
                user_task=user_input,
                content=refiner_output.refined_content
            )
            re_critic_output: CriticOutput = critic_tool(re_critic_input)

            if re_critic_output.passed:
                console.print(f"[bold green]✅ ADK PIPELINE: Content passed after {attempt} refinement(s)[/bold green]")
                final_content = refiner_output.refined_content
                break

            current_content = refiner_output.refined_content
            critic_output = re_critic_output

            if attempt == MAX_REFINEMENTS:
                console.print(f"[bold red]🛑 Max refinements ({MAX_REFINEMENTS}) reached. Pipeline complete.[/bold red]")
                final_content = current_content
    else:
        console.print("[bold green]✅ ADK PIPELINE: Content passed first review![/bold green]")

    # ── Final Output ──────────────────────────────────
    console.print(Rule("[bold magenta]🎯 ADK PIPELINE COMPLETE[/bold magenta]"))
    console.print(Panel(final_content, title="✅ Final Output", border_style="magenta"))

    return final_content


if __name__ == "__main__":
    task = "Create a blog about AI agents and include latest trends"
    run(task)
