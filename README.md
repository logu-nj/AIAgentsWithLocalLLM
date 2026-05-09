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
<<<<<<< HEAD
=======
├── test-cases/         # Edge case definitions (loops, failures, etc.)
├── compare_all.py      # Script to run and compare all frameworks
>>>>>>> a3b2a3ff65d09dda48a2b4756fbb30c14156401b
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

<<<<<<< HEAD
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows:** `.venv\Scripts\activate`
   - **Mac/Linux:** `source .venv/bin/activate`

4. **Install dependencies:**
=======
2. **Install dependencies:**
>>>>>>> a3b2a3ff65d09dda48a2b4756fbb30c14156401b
   *(If you are using `uv` which is configured in this project)*
   ```bash
   uv sync
   ```
   *(Or using pip)*
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
=======
3. **Activate the virtual environment:**
   - **Windows:** `.venv\Scripts\activate`
   - **Mac/Linux:** `source .venv/bin/activate`

## 🐛 Debugging in VS Code

A custom `.vscode/launch.json` is provided to ensure smooth debugging within your virtual environment.

1. **Select the correct Python Interpreter:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac).
   - Type and select **`Python: Select Interpreter`**.
   - Choose the interpreter located in your project's `.venv` folder (e.g., `./.venv/Scripts/python.exe`).
   
2. **Start Debugging:**
   - Open the specific python file you want to debug (for example, `lg_agent/main.py`).
   - **IMPORTANT:** Make sure the Python script is the active tab in VS Code.
   - Press **`F5`** or go to `Run > Start Debugging`. 
   - *Note: Avoid the "Play" button in the top right corner as it might bypass the launch configuration.*

>>>>>>> a3b2a3ff65d09dda48a2b4756fbb30c14156401b
## 🧪 Running the Comparisons

To run a specific agent framework or the full comparative test, you can use the provided runner scripts:

```bash
# Run a specific framework's implementation
python run.py
<<<<<<< HEAD
```
=======

# Run the comprehensive comparison across all frameworks
python compare_all.py
```

## 🎯 Testing Edge Cases
The framework includes built-in stress testing located in the `test-cases` directory. These tests evaluate how each framework handles:
- Circular dependencies and infinite loops
- LLM hallucinations and token limit explosions
- Task validation failures and retry limits
>>>>>>> a3b2a3ff65d09dda48a2b4756fbb30c14156401b
