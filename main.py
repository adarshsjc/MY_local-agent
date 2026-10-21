import os
import sys

# 1. Suppress the Pydantic AI startup ASCII banner
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

import gc
import json
import subprocess
import time
import urllib.request

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from security import init_workspace
from tools import (
    calculate,
    create_or_write_file,
    get_current_time,
    list_workspace_files,
    read_file,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "qwen3.5:4b"
OLLAMA_BASE_URL = "http://localhost:11434"
CONTEXT_WINDOW_LIMIT = 8192


def check_ollama_running() -> bool:
    """Verify that Ollama server is responsive."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def unload_model_from_vram(model_name: str) -> bool:
    """Forces Ollama to release the model from GPU/VRAM immediately."""
    try:
        payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    try:
        subprocess.run(
            ["ollama", "stop", model_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# Agent Definition with Thinking Instructions
# ---------------------------------------------------------
model = OllamaModel(
    MODEL_NAME,
    provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
)

agent = Agent(
    model,
    tools=[
        get_current_time,
        calculate,
        create_or_write_file,
        read_file,
        list_workspace_files,
    ],
    instructions=(
        f"You are a helpful, secure local assistant running on {MODEL_NAME}.\n"
        "Always structure your response with reasoning: first write your thoughts "
        "inside <think>...</think> tags explaining what tools you plan to use and why.\n"
        "After closing </think>, provide your final user-facing response.\n"
        "You can read, create, and list files ONLY within the workspace folder.\n"
        "Never attempt to access directories outside of it. Keep final answers concise."
    ),
)


# ---------------------------------------------------------
# Telemetry Display (Fixed for Property / Callable usage)
# ---------------------------------------------------------
def get_usage_object(source):
    """Safely extracts the RunUsage object whether it's a property or callable."""
    u = getattr(source, "usage", None)
    if callable(u):
        try:
            return u()
        except TypeError:
            return u
    return u


def display_telemetry(usage, elapsed_seconds: float):
    if not usage:
        return

    # Supports both Pydantic AI v1 (request/response) and v2 (input/output) fields
    input_tokens = getattr(usage, "input_tokens", getattr(usage, "request_tokens", 0)) or 0
    output_tokens = getattr(usage, "output_tokens", getattr(usage, "response_tokens", 0)) or 0
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens)

    tps = output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    fill_ratio = min(1.0, total_tokens / CONTEXT_WINDOW_LIMIT)
    fill_pct = fill_ratio * 100.0

    bar_width = 24
    filled_blocks = int(round(fill_ratio * bar_width))
    progress_bar = "█" * filled_blocks + "░" * (bar_width - filled_blocks)

    print("\n" + "─" * 68)
    print(f" ⚡ Speed:     {tps:.2f} tokens/sec ({output_tokens} tokens in {elapsed_seconds:.2f}s)")
    print(f" 🧠 Context:   [{progress_bar}] {total_tokens:,}/{CONTEXT_WINDOW_LIMIT:,} tokens ({fill_pct:.1f}% filled)")
    print(f" 📊 Breakdown: Input: {input_tokens:,} tokens | Output: {output_tokens:,} tokens")
    print("─" * 68 + "\n")


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    # 1. Check workspace existence
    workspace_path, existed = init_workspace()

    # 2. Check Ollama
    if not check_ollama_running():
        print("[Warning] Ollama server is unreachable at http://localhost:11434.")
        print("Make sure 'ollama serve' is running.\n")

    print("=" * 68)
    print(f" Local Agent Initialized | Model: {MODEL_NAME}")
    if existed:
        print(f" Workspace: Found existing folder at: {workspace_path}")
    else:
        print(f" Workspace: Created new workspace at: {workspace_path}")
    print(" Type 'quit' or 'exit' to wipe memory and unload the model.")
    print("=" * 68 + "\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "quit"

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\n[Exiting] Purging session memory...")
            history.clear()
            del history
            gc.collect()
            print("[Memory] All active session history wiped from RAM.")

            print(f"[Ollama] Unloading '{MODEL_NAME}' from GPU/VRAM...")
            unload_model_from_vram(MODEL_NAME)
            print("[Ollama] Model unloaded. System resources freed. Goodbye!\n")
            break

        start_time = time.perf_counter()

        try:
            # Use run_stream_sync with live thinking token formatting
            if hasattr(agent, "run_stream_sync"):
                with agent.run_stream_sync(user_input, message_history=history) as stream:
                    in_thinking = False
                    for chunk in stream.stream_text(delta=True):
                        if "<think>" in chunk:
                            in_thinking = True
                            chunk = chunk.replace("<think>", "\n🤔 [Thinking]:\n\033[90m")
                        if "</think>" in chunk:
                            in_thinking = False
                            chunk = chunk.replace("</think>", "\033[0m\n\n💬 Agent:\n")
                        print(chunk, end="", flush=True)

                    if in_thinking:
                        print("\033[0m", end="")

                    raw_usage = get_usage_object(stream)
                    history = stream.all_messages()
            else:
                result = agent.run_sync(user_input, message_history=history)
                output_text = result.output
                if "<think>" in output_text and "</think>" in output_text:
                    think_part = output_text.split("<think>")[1].split("</think>")[0].strip()
                    clean_output = output_text.split("</think>")[1].strip()
                    print(f"\n🤔 [Thinking]:\n\033[90m{think_part}\033[0m\n")
                    print(f"\n💬 Agent:\n{clean_output}")
                else:
                    print(f"\n💬 Agent:\n{output_text}")

                raw_usage = get_usage_object(result)
                history = result.all_messages()

            elapsed = time.perf_counter() - start_time
            display_telemetry(raw_usage, elapsed)

        except Exception as e:
            print(f"\n[Execution Error]: {e}\n")


if __name__ == "__main__":
    main()