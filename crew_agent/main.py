"""
=============================================================
 FRAMEWORK 3 — CrewAI  |  Smart Task Executor
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crewai import Agent, Task, Crew, Process, LLM
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────
#  LLM Setup — CrewAI native LLM wrapper
# ─────────────────────────────────────────────

ollama_llm = LLM(
    model="ollama/gemma4:e2b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# ─────────────────────────────────────────────
#  1. DEFINE AGENTS
# ─────────────────────────────────────────────

planner_agent = Agent(
    role="Strategic Task Planner",
    goal="Break complex tasks into clear, numbered action plans that guide content creation",
    backstory="""You are a seasoned project manager who has planned hundreds of
content creation projects. You are known for producing crystal-clear, 
numbered step-by-step plans that leave no room for ambiguity. 
You ONLY output the plan — no commentary, no preamble.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,   # No role confusion — planner plans only
    max_iter=3,
)

executor_agent = Agent(
    role="Expert Content Writer",
    goal="Execute the given plan and produce high-quality, detailed written content",
    backstory="""You are a professional writer with expertise in technology, AI,
and business writing. You follow plans precisely and produce content that is 
informative, engaging, and well-structured. You never deviate from the plan.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

critic_agent = Agent(
    role="Quality Assurance Critic",
    goal="Rigorously evaluate content quality and provide a clear PASS or FAIL verdict with specific reasons",
    backstory="""You are a strict quality assurance reviewer with high standards.
You evaluate content for completeness, accuracy, structure, and engagement.
You respond with EXACTLY 'PASS' or 'FAIL: <specific reason>'.
You NEVER pass mediocre content.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,
)

refiner_agent = Agent(
    role="Content Refinement Expert",
    goal="Fix and improve content based on critic feedback to achieve PASS status",
    backstory="""You are an expert editor who specializes in taking good content
and making it excellent. You carefully address every criticism raised and rewrite
content to be comprehensive, accurate, and compelling.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)

# ─────────────────────────────────────────────
#  2. BUILD TASKS (with context dependencies)
# ─────────────────────────────────────────────

def build_tasks(user_input: str):
    planning_task = Task(
        description=f"""Create a detailed action plan for this task: "{user_input}"  
Output a numbered list of 4-6 clear steps that a writer should follow.
Each step should be specific and actionable.
Output ONLY the numbered list.""",
        expected_output="A numbered list of 4-6 actionable steps for completing the task",
        agent=planner_agent,
    )

    execution_task = Task(
        description=f"""Write comprehensive content for: "{user_input}"
Follow the plan provided by the Strategic Task Planner exactly.
Produce detailed, high-quality content that covers all aspects of the topic.
Include specific examples, data points, and practical insights where relevant.""",
        expected_output="A comprehensive, well-structured piece of content (at least 400 words)",
        agent=executor_agent,
        context=[planning_task],  # ← Task dependency: waits for plan
    )

    critic_task = Task(
        description=f"""Review the content written for: "{user_input}"
Evaluate based on:
1. Completeness - Does it cover the topic thoroughly?
2. Accuracy - Is the information correct?
3. Structure - Is it well-organized?
4. Engagement - Is it interesting and readable?

Respond with EXACTLY:
- "PASS" if all criteria are met
- "FAIL: <specific improvement needed>" if any criterion fails""",
        expected_output="Either 'PASS' or 'FAIL: <reason>'",
        agent=critic_agent,
        context=[execution_task],  # ← Depends on executor output
    )

    refiner_task = Task(
        description=f"""Improve the content for: "{user_input}"
You have received feedback from the Quality Assurance Critic.
Address ALL the stated issues and rewrite the content to be excellent.
The improved version should clearly pass a quality review.""",
        expected_output="An improved, comprehensive version of the content",
        agent=refiner_agent,
        context=[execution_task, critic_task],  # ← Both content + verdict
    )

    return planning_task, execution_task, critic_task, refiner_task


# ─────────────────────────────────────────────
#  3. ORCHESTRATE: Run with conditional refinement
# ─────────────────────────────────────────────

MAX_REFINEMENTS = 3

def run(user_input: str):
    console.print(Panel(
        f"[bold white]{user_input}[/bold white]",
        title="[bold yellow]CrewAI - Smart Task Executor[/bold yellow]",
        border_style="yellow"
    ))

    planning_task, execution_task, critic_task, refiner_task = build_tasks(user_input)

    # ── Phase 1: Plan + Execute + Critique ──────────────
    console.print(Rule("[bold yellow]Phase 1: Plan > Execute > Critique[/bold yellow]"))

    phase1_crew = Crew(
        agents=[planner_agent, executor_agent, critic_agent],
        tasks=[planning_task, execution_task, critic_task],
        process=Process.sequential,
        verbose=True,
    )

    phase1_result = phase1_crew.kickoff()
    critic_output = str(phase1_result.raw).strip()

    console.print(Panel(critic_output, title="Critic Verdict", border_style="red"))

    # ── Phase 2: Conditional Refinement ─────────────────
    current_content = str(execution_task.output.raw if execution_task.output else "")
    final_content = current_content

    if "FAIL" in critic_output.upper():
        console.print(Rule("[bold orange3]Phase 2: Refinement Loop[/bold orange3]"))

        for attempt in range(1, MAX_REFINEMENTS + 1):
            console.print(f"[yellow]Refinement attempt {attempt}/{MAX_REFINEMENTS}[/yellow]")

            # Build a fresh refiner task with updated context
            fresh_refiner_task = Task(
                description=f"""Improve this content for: "{user_input}"
                
Critic feedback: {critic_output}

Current content:
{current_content}

Fix ALL issues and produce excellent content.""",
                expected_output="Improved content that addresses all critic concerns",
                agent=refiner_agent,
            )

            fresh_critic_task = Task(
                description=f"""Re-evaluate the refined content for: "{user_input}"
                
Respond with EXACTLY "PASS" or "FAIL: <reason>". Be strict.""",
                expected_output="Either 'PASS' or 'FAIL: <reason>'",
                agent=critic_agent,
                context=[fresh_refiner_task],
            )

            refinement_crew = Crew(
                agents=[refiner_agent, critic_agent],
                tasks=[fresh_refiner_task, fresh_critic_task],
                process=Process.sequential,
                verbose=True,
            )

            ref_result = refinement_crew.kickoff()
            new_verdict = str(ref_result.raw).strip()
            new_content = str(fresh_refiner_task.output.raw if fresh_refiner_task.output else current_content)

            console.print(Panel(new_verdict, title=f"Re-Critic Verdict (attempt {attempt})", border_style="red"))

            if "PASS" in new_verdict.upper() and "FAIL" not in new_verdict.upper():
                console.print(f"[bold green]PASSED after {attempt} refinement(s)![/bold green]")
                final_content = new_content
                break

            current_content = new_content
            critic_output = new_verdict

            if attempt == MAX_REFINEMENTS:
                console.print(f"[bold red]Max refinements ({MAX_REFINEMENTS}) reached. Using best version.[/bold red]")
                final_content = current_content
    else:
        console.print("[bold green]Content PASSED on first try![/bold green]")

    # ── Final Output ──────────────────────────────────
    console.print(Rule("[bold yellow]FINAL OUTPUT[/bold yellow]"))
    console.print(Panel(final_content, title="Result", border_style="yellow"))

    return final_content


if __name__ == "__main__":
    task = "Create a blog about AI agents and include latest trends"
    run(task)
