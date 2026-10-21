import os
import sys

# Suppress Pydantic AI startup ASCII banner
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

import gc
import json
import re
import subprocess
import threading
import time
import urllib.request

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from security import init_workspace
from tools import (
    calculate,
    create_or_write_file,
    delete_file,
    get_current_time,
    list_workspace_files,
    read_file,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "qwen3.5:4b-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"
CONTEXT_WINDOW_LIMIT = 8192


class BlinkingThinkingIndicator:
    """Displays an instant animated 'Thinking . . .' spinner in the terminal."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self):
        frames = ["Thinking .  ", "Thinking .. ", "Thinking ...", "Thinking    "]
        idx = 0
        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            sys.stdout.write(f"\r\033[93m🤔 {frame}\033[0m")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.35)

    def stop(self):
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            sys.stdout.write("\r\033[K")  # Clear the spinner line
            sys.stdout.flush()


def check_ollama_running() -> bool:
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
# Agent Definition
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
        delete_file,
        list_workspace_files,
    ],
    instructions=(
        f"You are a local assistant running on {MODEL_NAME}.\n"
        "Always structure your response with reasoning: first write your thoughts "
        "inside <think>...</think> tags explaining what you plan to do.\n"
        "After closing </think>, provide your direct user-facing response.\n"
        "You can read, write, list, and delete files ONLY within the workspace folder.\n"
        "Never attempt to access or modify directories outside of it."
    ),
)


def get_usage_object(source):
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
    workspace_path, existed = init_workspace()

    if not check_ollama_running():
        print("[Warning] Ollama server is unreachable at http://localhost:11434.")
        print("Make sure 'ollama serve' is running.\n")

    print("=" * 68)
    print(f" Local Agent Initialized | Model: {MODEL_NAME}")
    if existed:
        print(f" Workspace: Found existing folder at: {workspace_path}")
    else:
        print(f" Workspace: Created new workspace at: {workspace_path}")
    print(" Tips: Type 't' or 'think' anytime to expand the model's thoughts.")
    print("       Type 'quit' or 'exit' to wipe memory and unload the model.")
    print("=" * 68 + "\n")

    history = []
    last_thought = ""
    indicator = BlinkingThinkingIndicator()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "quit"

        if not user_input:
            continue

        # Toggle to view last thought process
        if user_input.lower() in ("t", "think", "thinking"):
            if last_thought:
                print("\n┌── 🧠 Model Thought Process ────────────────────────────────────")
                for line in last_thought.splitlines():
                    print(f"│  {line}")
                print("└── ─────────────────────────────────────────────────────────────\n")
            else:
                print("\n[Notice] No previous thought process captured yet.\n")
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

        # Start instant blinking indicator
        indicator.start()
        start_time = time.perf_counter()

        try:
            result = agent.run_sync(user_input, message_history=history)
            elapsed = time.perf_counter() - start_time
            indicator.stop()

            raw_text = result.output

            # Extract <think> content
            thought_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
            if thought_match:
                last_thought = thought_match.group(1).strip()
                final_answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            else:
                last_thought = ""
                final_answer = raw_text.strip()

            # Display ChatGPT-like collapsible bar
            if last_thought:
                word_count = len(last_thought.split())
                print(f"\n\033[90m▶ Thought for {elapsed:.1f}s ({word_count} words) [Type 't' to expand]\033[0m")

            print(f"\nAgent: {final_answer}")

            raw_usage = get_usage_object(result)
            history = result.all_messages()
            display_telemetry(raw_usage, elapsed)

        except Exception as e:
            indicator.stop()
            print(f"\n[Execution Error]: {e}\n")


if __name__ == "__main__":
    main()