# Multi-Agent Framework Comparison

This project is a comparative multi-agent learning study that demonstrates the architectural differences between four leading AI agent frameworks: **LangGraph**, **AutoGen**, **CrewAI**, and **Google ADK**. 

The primary goal is to build a "Smart Task Executor with Validation" using each framework, backed by a shared local Ollama client. It systematically tests edge cases like infinite loops, task dependency failures, and token explosions to evaluate framework stability, control flow, and production readiness.

## 🏗️ Project Structure

```text
multi-agent-learning/
├── adk_agent/          # Google ADK implementation
├── ag_agent/           # AutoGen implementation
├── crew_agent/         # CrewAI implementation
├── lg_agent/           # LangGraph implementation
├── shared/             # Shared utilities and Ollama client
├── prompts/            # Centralized system and agent prompts
├── run.py              # Main entry point runner
└── pyproject.toml / requirements.txt
```

## 🚀 Prerequisites

1. **Python 3.12+**
2. **Ollama**: Installed and running locally.
3. **Local Model**: Pull the required model via Ollama:
   ```bash
   ollama run gemma4:e2b
   ```

## 🛠️ Setup & Installation

This project uses a virtual environment to manage dependencies.

1. **Navigate to the project directory:**
   ```bash
   cd multi-agent-learning
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows:** `.venv\Scripts\activate`
   - **Mac/Linux:** `source .venv/bin/activate`

4. **Install dependencies:**
   *(If you are using `uv` which is configured in this project)*
   ```bash
   uv sync
   ```
   *(Or using pip)*
   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Running the Comparisons

To run a specific agent framework or the full comparative test, you can use the provided runner scripts:

```bash
# Run a specific framework's implementation
python run.py
```