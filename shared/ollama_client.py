"""
Shared Ollama client configuration for all frameworks.
Model: gemma4:e2b via http://localhost:11434
"""

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b"

# For LangChain / LangGraph / CrewAI via LangChain
from langchain_ollama import ChatOllama

def get_langchain_llm(temperature: float = 0.7) -> ChatOllama:
    """Returns a LangChain-compatible ChatOllama LLM."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )

# For raw Ollama calls (AutoGen custom model client)
import ollama as _ollama

def raw_generate(prompt: str, system: str = "") -> str:
    """Direct Ollama call, returns string response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    return response.message.content
