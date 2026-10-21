# MY_local-agent

A small personal AI assistant that runs locally with [Ollama](https://ollama.com/) and Pydantic AI.

## Features

- Answers questions using a locally hosted language model.
- Provides tools to get the current date and time, calculate arithmetic expressions, and save or read notes.
- Stores notes in `notes.txt` in the project directory. This file is ignored by Git because notes may contain personal information.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The `qwen3.5:2b` model downloaded in Ollama

## Setup

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependency and download the model:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3.5:2b
```

Make sure Ollama is running, then start the assistant:

```powershell
python main.py
```

Type `quit` or `exit` to close the assistant.

## Project files

```text
MY_local-agent/
├── main.py
├── .gitignore
├── requirements.txt
└── README.md
```

`notes.txt` is created automatically the first time the assistant saves a note and is intentionally not included in Git.
