# Code Fixer

> An AI-powered Python code debugging assistant that automatically runs, diagnoses, and fixes broken code — all through an interactive chat UI.

## Preview

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/LLM-Mistral-orange.svg" alt="Mistral AI">
  <img src="https://img.shields.io/badge/Framework-LangGraph-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/UI-Streamlit-red.svg" alt="Streamlit">
</p>

## Description

Code Fixer tackles the frustration of hunting down bugs in Python code. Instead of manually tracing errors, paste your broken code or upload a `.py`/`.txt` file, and the system will:

1. **Execute** it in an isolated sandbox to capture the real error output.
2. **Analyze** the failure and iteratively generate fixes using Mistral's Codestral model.
3. **Verify** each fix by re-running the corrected code until it works.

Built for developers, students, and non-technical users who need quick, reliable code repair without setting up complex debugging environments.

## Features

- **Sandboxed Code Execution** — runs Python code in a Docker-isolated environment, outputting real error traces safely.
- **Automated Code Repair** — leverages Mistral Codestral for targeted, structured fix generation (`CodeFix` Pydantic schema).
- **Iterative Debugging** — up to 3 automatic fix–verify cycles per error, installing missing packages on demand.
- **Tool-Use Graph** — LangGraph orchestrates a ReAct-style agent loop (execute → generate_fix → verify) with chat memory.
- **Multi-Thread Chat** — sidebar supports creating, renaming, and deleting conversations; all history persists in SQLite via LangGraph checkpoints.
- **File Upload** — accepts `.py` and `.txt` files alongside inline text input.
- **Clean UI** — streaming token response with collapsible tool-call detail panels (execution logs, install status, generated fixes).

## Tech Stack

| Layer           | Technology                                                |
| --------------- | --------------------------------------------------------- |
| **Language**    | Python 3.12+                                              |
| **LLM API**     | Mistral AI (Mistral Large for agent, Codestral for fixes) |
| **Agent**       | LangChain + LangGraph (StateGraph, tools, checkpoints)    |
| **Sandbox**     | llm-sandbox (Docker backend)                              |
| **UI**          | Streamlit                                                 |
| **Persistence** | SQLite (LangGraph SqliteSaver)                            |
| **Package Mgmt**| uv                                                        |

## Prerequisites

- **Python 3.12+** (specified in `.python-version`)
- **uv** — fast Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker** — required for sandboxed code execution (the server must have Docker running)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ASL_Bug_Fixer.git
cd ASL_Bug_Fixer

# Install dependencies with uv
uv sync
```

## Configuration

Create a `.env` file in the project root with your Mistral API key:

```env
MISTRAL_API_KEY=your_api_key_here
```

Get your key from the [Mistral AI console](https://console.mistral.ai/).

## Usage

```bash
# Launch the Streamlit app
uv run streamlit run src/frontend.py
```

1. **Pick a language** — the web UI opens in your default browser (usually `localhost:8501`).
2. **Paste or upload** — paste code in the chat input or upload a `.py`/`.txt` file.
3. **Watch the fix** — the status bar shows live tool calls: execution, fix generation, package installation.
4. **Review** — collapsed expanders reveal execution logs, the generated fix code, and reasoning.

## Project Structure

```
.
├── pyproject.toml          # Project metadata and dependencies (uv)
├── .env                    # API keys (git-ignored)
│
├── src/
│   ├── backend.py          # LangGraph agent, tools, checkpoint DB, LLM config
│   └── frontend.py         # Streamlit UI, chat management, sidebar threads
│
└── notebooks/
    └── test.ipynb          # Experimentation playground
```

### Backend (`src/backend.py`)

| Component            | Role                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `execute_code`       | Runs Python in a Docker sandbox, returns stdout/stderr (truncated to 3000 chars)         |
| `generate_fix`       | Calls Codestral with structured output to produce a `CodeFix` (updated code + reasoning) |
| `install_package`    | Installs missing pip packages inside sandbox and re-runs code                            |
| `StateGraph`         | Agent → Tools → Agent loop with `tools_condition` routing                                |
| `SqliteSaver`        | Persists conversation state and thread labels in `code_fixer.db`                         |

### Frontend (`src/frontend.py`)

| Feature              | Details                                                             |
| -------------------- | ------------------------------------------------------------------- |
| Chat threads         | Create, rename, and delete conversations with auto-generated labels |
| File upload          | `.py` and `.txt` preview with inline code display                   |
| Streaming response   | Token-by-token rendering with collapsible tool-call panels          |
| History persistence  | Rebuilds message history from LangGraph checkpoint state            |
