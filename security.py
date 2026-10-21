from pathlib import Path

# Workspace directory pinned strictly to a 'workspace' subfolder
WORKSPACE_DIR = (Path(__file__).parent / "workspace").resolve()


def init_workspace() -> Path:
    """Ensures the dedicated AI workspace directory exists."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def safe_resolve_path(filename: str) -> Path:
    """
    Resolves a relative filename and ensures it remains strictly
    inside WORKSPACE_DIR. Prevents '../' traversal and absolute paths.
    """
    path_obj = Path(filename)

    # Reject absolute paths (e.g. C:\Windows, /etc/passwd)
    if path_obj.is_absolute():
        raise PermissionError("Security Violation: Absolute paths are strictly forbidden.")

    resolved = (WORKSPACE_DIR / path_obj).resolve()

    # Verify that the resolved path does not escape the workspace boundary
    try:
        resolved.relative_to(WORKSPACE_DIR)
    except ValueError:
        raise PermissionError("Security Violation: Path escapes the authorized workspace sandbox.")

    return resolved


def safe_eval_expression(expression: str) -> str:
    """Evaluates basic math expressions using a strict character allowlist and no builtins."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Security Error: Only digits and operators (+, -, *, /, (), .) are allowed."
    try:
        # Evaluate with zero globals/builtins to prevent arbitrary execution
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"