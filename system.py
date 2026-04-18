import os
from pathlib import Path
from typing import NamedTuple

# Paths and configuration
REPOSITORIES_DIR = os.getenv('REPOSITORIES_DIR', os.path.expanduser('~/.myai/repositories'))
REPOSITORIES_VROOT = "/repositories"
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/.myai/workspace'))
WORKSPACE_VROOT = "/workspace"

os.makedirs(REPOSITORIES_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def is_safe_vpath(vpath: Path, expected_vroot: Path) -> tuple[bool, str]:
    try:
        parts = vpath.parts
        if ".." in parts:
            return False, "Path traversal detected: '..' not allowed"
        if not vpath.is_relative_to(expected_vroot):
            return False, f"Path must be under {expected_vroot}"
        return True, ""
    except Exception as e:
        return False, f"Path validation error: {str(e)}"

def vpath_to_realpath(vpath: Path, vroot: str, base_dir: str) -> Path:
    parts = list(vpath.parts)
    if parts[0] == "/":
        parts = parts[1:]
    if parts and parts[0] == vroot.lstrip("/"):
        parts = parts[1:]

    return Path(base_dir) / Path(*parts)


class ShellResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str

def run_sandboxed_command(command: str) -> ShellResult:
    import subprocess
    # Resolve absolute paths for safety
    workspace_path = os.path.abspath(WORKSPACE_DIR)
    reference_path = os.path.abspath(REPOSITORIES_DIR)

    # Build the bwrap command arguments
    bwrap_args = [
        "bwrap",
        # Mount essential system directories as read‑only
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        # Unshare namespaces for isolation
        "--unshare-all",
        # Writable workspace
        "--bind", workspace_path, WORKSPACE_VROOT,
        # Set working directory inside the sandbox
        "--chdir", "/",
    ]

    for entry in os.scandir(reference_path):
        # bwrap --ro-bind follows symlinks
        bwrap_args.extend(["--ro-bind", entry.path, f"{REPOSITORIES_VROOT}/{entry.name}"])

    # Execute the provided command via bash
    bwrap_args.extend([
        "bash", "-c", command
    ])

    try:
        result = subprocess.run(
            bwrap_args,
            capture_output=True,
            text=True,
            timeout=30,  # safety timeout
        )
        return ShellResult(
            result.returncode,
            result.stdout,
            result.stderr
        )
    except subprocess.TimeoutExpired:
        return ShellResult(
            returncode=None,
            stdout='',
            stderr=str("Command timed out"),
        )
    except Exception as e:
        return ShellResult(
            returncode=None,
            stdout='',
            stderr=str(e),
        )
