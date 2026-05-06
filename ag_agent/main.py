"""
=============================================================
 FRAMEWORK 2 — AutoGen  |  Smart Task Executor
=============================================================
 Flow:   User Input → Group Chat (Planner → Executor → Critic)
                                              ↓
                                         (FAIL signal)
                                              ↓
                                          Refiner → Critic (loop)

 Concepts demonstrated:
  • Conversational multi-agent (GroupChat)
  • Custom model client for Ollama (no OpenAI needed)
  • Role-based prompting
  • Message limit guard (cost explosion prevention)
  • Conversation termination via TERMINATE keyword

 Edge cases handled:
  • Agents arguing endlessly → max_round limit
  • Cost explosion → message count tracked
  • Hallucinated corrections → Critic re-validates
=============================================================
"""

import sys, os, re
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.ollama_client import OLLAMA_BASE_URL, OLLAMA_MODEL, raw_generate
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

# ─────────────────────────────────────────────
#  Custom Ollama Client for AutoGen
#  AutoGen 0.4+ uses a model_client interface.
#  We implement a thin wrapper around raw ollama.
# ─────────────────────────────────────────────

class OllamaModelClient:
    """Minimal AutoGen-compatible model client for Ollama."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._message_count = 0  # Token-explosion tracker

    def create(self, messages: list, **kwargs) -> dict:
        self._message_count += 1
        # Flatten messages into a prompt
        prompt_parts = []
        system_msg = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_msg = content
            else:
                prompt_parts.append(f"{role.upper()}: {content}")

        full_prompt = "\n".join(prompt_parts)
        response_text = raw_generate(full_prompt, system=system_msg)

        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "usage": {"total_tokens": len(response_text.split()) * 2},
        }

    @property
    def message_count(self):
        return self._message_count


# ─────────────────────────────────────────────
#  Agent Implementation (Manual conversation loop)
#  We use a manual loop (not AutoGen GroupChat) to stay
#  compatible with pyautogen 0.10 + local Ollama.
# ─────────────────────────────────────────────

class SimpleAgent:
    """A minimal agent with a name, system prompt, and Ollama backend."""

    def __init__(self, name: str, system_prompt: str, client: OllamaModelClient):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client

    def respond(self, conversation_history: list[dict]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)

        result = self.client.create(messages)
        reply = result["choices"][0]["message"]["content"].strip()
        return reply


# ─────────────────────────────────────────────
#  Define Agents
# ─────────────────────────────────────────────

def build_agents(client: OllamaModelClient):
    planner = SimpleAgent(
        name="Planner",
        system_prompt="""You are a strategic planning agent.
When given a task, output a clear numbered action plan (max 5 steps).
Output ONLY the plan. Nothing else.""",
        client=client,
    )

    executor = SimpleAgent(
        name="Executor",
        system_prompt="""You are a skilled content writer.
You will receive a plan and must execute it to produce high-quality content.
Write detailed, informative, and well-structured content.""",
        client=client,
    )

    critic = SimpleAgent(
        name="Critic",
        system_prompt="""You are a strict quality critic.
Evaluate content and respond with EXACTLY:
- "PASS" if the content is complete, accurate and well-structured
- "FAIL: <reason>" if it has issues

Your response MUST start with either PASS or FAIL.""",
        client=client,
    )

    refiner = SimpleAgent(
        name="Refiner",
        system_prompt="""You are an expert editor and content refiner.
You receive rejected content and the critic's reason for rejection.
Rewrite and improve the content to fix all stated issues.
Output ONLY the improved content.""",
        client=client,
    )

    return planner, executor, critic, refiner


# ─────────────────────────────────────────────
#  Orchestrator — Manual Conversation Loop
# ─────────────────────────────────────────────

MAX_ROUNDS = 4          # Edge-case: prevent infinite argument loops
MESSAGE_LIMIT = 20      # Edge-case: cost explosion guard

def run(user_input: str):
    console.print(Panel(
        f"[bold white]{user_input}[/bold white]",
        title="[bold blue]🚀 AutoGen — Smart Task Executor[/bold blue]",
        border_style="blue"
    ))

    client = OllamaModelClient()
    planner, executor, critic, refiner = build_agents(client)

    conversation: list[dict] = []  # Shared conversation history
    total_messages = 0

    def agent_turn(agent: SimpleAgent, user_msg: str) -> str:
        nonlocal total_messages
        total_messages += 1

        # Cost explosion guard
        if total_messages > MESSAGE_LIMIT:
            console.print(f"[bold red]💸 MESSAGE LIMIT ({MESSAGE_LIMIT}) REACHED — Stopping to prevent cost explosion![/bold red]")
            return "[TERMINATED: message limit reached]"

        conversation.append({"role": "user", "content": user_msg})
        reply = agent.respond(conversation)
        conversation.append({"role": "assistant", "content": f"[{agent.name}]: {reply}"})

        console.print(Panel(
            reply[:400] + ("..." if len(reply) > 400 else ""),
            title=f"[bold]{agent.name}[/bold]",
            border_style="blue"
        ))
        return reply

    # ── Step 1: Planner ──────────────────────────────
    console.print(Rule("[bold cyan]🧠 PLANNER[/bold cyan]"))
    plan = agent_turn(planner, f"Create a plan for this task: {user_input}")

    if not plan or len(plan) < 20:
        plan = "1. Research topic\n2. Draft content\n3. Review\n4. Refine\n5. Finalize"
        console.print("[yellow]⚠️  Bad planner output — using fallback plan.[/yellow]")

    # ── Step 2: Executor ─────────────────────────────
    console.print(Rule("[bold green]✍️  EXECUTOR[/bold green]"))
    content = agent_turn(executor, f"Execute this plan and write the content:\n{plan}")

    if not content or len(content) < 50:
        content = "[EXECUTOR PRODUCED INSUFFICIENT CONTENT]"
        console.print("[red]❌ Executor failure — content too short.[/red]")

    # ── Step 3: Critic + Refiner Loop ────────────────
    current_content = content
    final_content = content

    for round_num in range(1, MAX_ROUNDS + 1):
        console.print(Rule(f"[bold red]🔍 CRITIC — Round {round_num}/{MAX_ROUNDS}[/bold red]"))
        verdict = agent_turn(
            critic,
            f"Review this content for the task '{user_input}':\n{current_content}"
        )

        if "PASS" in verdict.upper() and "FAIL" not in verdict.upper():
            console.print(f"[bold green]✅ CRITIC PASSED at round {round_num}[/bold green]")
            final_content = current_content
            break

        # Extract failure reason
        reason = verdict
        if "FAIL" in verdict.upper():
            idx = verdict.upper().find("FAIL")
            reason = verdict[idx:]

        console.print(f"[bold yellow]🔁 CRITIC FAILED — round {round_num}: {reason}[/bold yellow]")

        if round_num == MAX_ROUNDS:
            console.print(f"[bold red]🛑 Max rounds ({MAX_ROUNDS}) reached. Keeping best content.[/bold red]")
            final_content = current_content
            break

        # Refine
        console.print(Rule(f"[bold yellow]🔧 REFINER — Round {round_num}[/bold yellow]"))
        refined = agent_turn(
            refiner,
            f"Fix this content (reason: {reason}):\n{current_content}"
        )
        current_content = refined if refined and len(refined) > 50 else current_content

    # ── Final Output ──────────────────────────────────
    console.print(Rule("[bold blue]🎯 FINAL OUTPUT[/bold blue]"))
    console.print(Panel(final_content, title="✅ Result", border_style="blue"))
    console.print(f"\n[dim]📊 Total messages exchanged: {total_messages} | Ollama calls: {client.message_count}[/dim]")

    return final_content


if __name__ == "__main__":
    task = "Create a blog about AI agents and include latest trends"
    run(task)
