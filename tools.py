from datetime import datetime
from security import WORKSPACE_DIR, safe_eval_expression, safe_resolve_path


def get_current_time() -> str:
    """Get the current date, day of week, and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def calculate(expression: str) -> str:
    """Safely evaluate a basic math expression like '14 * 23 + 5'."""
    return safe_eval_expression(expression)


def create_or_write_file(filename: str, content: str) -> str:
    """Create or overwrite a file strictly inside the sandboxed project workspace."""
    try:
        target = safe_resolve_path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel_path = target.relative_to(WORKSPACE_DIR).as_posix()
        return f"Success: File '{rel_path}' saved successfully ({len(content)} characters)."
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Error writing file: {e}"


def read_file(filename: str) -> str:
    """Read the content of a file strictly inside the sandboxed project workspace."""
    try:
        target = safe_resolve_path(filename)
        if not target.exists() or not target.is_file():
            return f"Error: File '{filename}' does not exist in workspace."
        return target.read_text(encoding="utf-8")
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Error reading file: {e}"


def list_workspace_files() -> str:
    """List all available files currently stored inside the sandboxed workspace."""
    try:
        files = [
            f.relative_to(WORKSPACE_DIR).as_posix()
            for f in WORKSPACE_DIR.rglob("*")
            if f.is_file()
        ]
        if not files:
            return "The workspace is currently empty."
        return "Files in workspace:\n" + "\n".join(f"- {f}" for f in files)
    except Exception as e:
        return f"Error listing workspace files: {e}"