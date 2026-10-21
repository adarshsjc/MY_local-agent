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
    """
    Forces Ollama to release the model from GPU/RAM immediately (keep_alive = 0).
    Falls back to `ollama stop` command if needed.
    """
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
# Model & Agent Definition
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
        f"You are a sandboxed assistant running locally on {MODEL_NAME}. "
        "You can read, create, and list files ONLY within the workspace folder. "
        "Never attempt to access directories outside of it. "
        "Maintain the ongoing context of the user's requests and answer directly in English."
    ),
)


# ---------------------------------------------------------
# Telemetry Display
# ---------------------------------------------------------
def display_telemetry(usage, elapsed_seconds: float):
    input_tokens = usage.request_tokens or 0
    output_tokens = usage.response_tokens or 0
    total_tokens = usage.total_tokens or (input_tokens + output_tokens)

    tps = output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    fill_ratio = min(1.0, total_tokens / CONTEXT_WINDOW_LIMIT)
    fill_pct = fill_ratio * 100.0

    bar_width = 24
    filled_blocks = int(round(fill_ratio * bar_width))
    progress_bar = "█" * filled_blocks + "░" * (bar_width - filled_blocks)

    print("\n" + "─" * 68)
    print(f" ⚡ Speed:     {tps:.2f} tokens/sec ({output_tokens} tokens in {elapsed_seconds:.2f}s)")
    print(f" 🧠 Context:   [{progress_bar}] {total_tokens:,}/{CONTEXT_WINDOW_LIMIT:,} tokens ({fill_pct:.1f}% filled)")
    print(f" 📊 Breakdown: Prompt: {input_tokens:,} tokens | Output: {output_tokens:,} tokens")
    print("─" * 68 + "\n")


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    # 1. Initialize dedicated AI workspace directory
    workspace_path = init_workspace()

    # 2. Verify Ollama service
    if not check_ollama_running():
        print("[Warning] Could not reach Ollama at http://localhost:11434.")
        print("Please ensure Ollama is installed and running (`ollama serve`).\n")

    print("=" * 68)
    print(f" Local Agent Initialized | Model: {MODEL_NAME}")
    print(f" Secure Workspace:        {workspace_path}")
    print(" Type 'quit' or 'exit' to wipe memory and unload the model.")
    print("=" * 68 + "\n")

    # In-memory session history (keeps your story/thread active during chat)
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "quit"

        if not user_input:
            continue

        # Clean Exit: Wipe session history & unload model from memory
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
        print("\nAgent: ", end="", flush=True)

        try:
            # Live token streaming
            if hasattr(agent, "run_stream_sync"):
                with agent.run_stream_sync(user_input, message_history=history) as stream:
                    for chunk in stream.stream_text(delta=True):
                        print(chunk, end="", flush=True)
                    usage = stream.usage()
                    history = stream.all_messages()
            else:
                result = agent.run_sync(user_input, message_history=history)
                print(result.output)
                usage = result.usage()
                history = result.all_messages()

            elapsed = time.perf_counter() - start_time
            display_telemetry(usage, elapsed)

        except Exception as e:
            print(f"\n[Execution Error]: {e}\n")


if __name__ == "__main__":
    main()