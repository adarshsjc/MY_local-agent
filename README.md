# MY_local-agent

A terminal-based personal assistant that runs locally with [Ollama](https://ollama.com/) and Pydantic AI.

## Features

- Chat with the locally hosted `qwen3.5:4b-q4_K_M` model.
- Get the current date and time and calculate arithmetic expressions.
- Create, read, list, and delete files inside the assistant's `workspace/` folder.
- Keep conversation history for the current run; session history is cleared when you exit.
- Display response timing and token-usage information when the model provides it.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The `qwen3.5:4b-q4_K_M` model available in Ollama

## Setup

Open PowerShell in the project directory and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependency and download the model:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
```

Start Ollama if it is not already running, then launch the assistant:

```powershell
python main.py
```

Type `quit` or `exit` to close the assistant. The application attempts to unload the model when it exits.

## Project layout

```text
MY_local-agent/
├── main.py
├── tools.py
├── security.py
├── requirements.txt
├── README.md
└── workspace/
```

The Python modules remain in the project root so the existing imports and `python main.py` command continue to work. The assistant's file tools are restricted to `workspace/`; files in that folder may be created, changed, or deleted by the assistant.

`project.md` is a local project note and is intentionally excluded from Git. The root `.gitignore` also excludes Python caches, virtual environments, environment files, and local notes.
