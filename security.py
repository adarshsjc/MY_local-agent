from pathlib import Path

# Sandboxed workspace directory
WORKSPACE_DIR = (Path(__file__).parent / "workspace").resolve()


def init_workspace() -> tuple[Path, bool]:
    """
    Checks if the workspace folder already exists.
    Reuses existing directory if present; creates it only if missing.
    Returns (WORKSPACE_DIR, already_existed).
    """
    already_existed = WORKSPACE_DIR.is_dir()
    if not already_existed:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR, already_existed


def safe_resolve_path(filename: str) -> Path:
    """
    Resolves relative path and verifies it stays strictly inside WORKSPACE_DIR.
    Prevents path climbing (../) and absolute system paths.
    """
    path_obj = Path(filename)

    if path_obj.is_absolute():
        raise PermissionError("Security Violation: Absolute paths are strictly forbidden.")

    resolved = (WORKSPACE_DIR / path_obj).resolve()

    try:
        resolved.relative_to(WORKSPACE_DIR)
    except ValueError:
        raise PermissionError("Security Violation: Path escapes the authorized workspace sandbox.")

    return resolved


def safe_eval_expression(expression: str) -> str:
    """Evaluates basic math expressions using a strict character allowlist and no builtins."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Security Error: Only numbers and operators (+, -, *, /, (), .) are allowed."
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"