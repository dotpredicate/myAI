import os
from pathlib import Path
from typing import NamedTuple, Optional, List

# Paths and configuration
REPOSITORIES_DIR = os.getenv('REPOSITORIES_DIR', os.path.expanduser('~/.myai/repositories'))
REPOSITORIES_VROOT = "/repositories"
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/.myai/workspace'))
WORKSPACE_VROOT = "/workspace"

os.makedirs(REPOSITORIES_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def get_repositories() -> List[str]:
    try:
        return [d.name for d in os.scandir(REPOSITORIES_DIR) if d.is_dir() and not d.name.startswith('.')]
    except Exception:
        return []

def is_safe_vpath(vpath: Path, expected_vroot: Path, allowed_scopes: Optional[List[str]] = None) -> tuple[bool, str]:
    try:
        parts = vpath.parts
        if ".." in parts:
            return False, "Path traversal detected: '..' not allowed"
        
        if not vpath.is_relative_to(expected_vroot):
            return False, f"Path must be under {expected_vroot}"

        if allowed_scopes:
            # The vpath is relative to the root (e.g., /repositories/repo_name/file.py)
            # We need to check if the first part after the root is in allowed_scopes
            vpath_str = str(vpath)
            
            # Check if any allowed scope is a prefix of the path
            # We check against "/repositories/<scope_name>"
            match_found = False
            for scope in allowed_scopes:
                prefix = f"{expected_vroot}/{scope}"
                if vpath_str.startswith(prefix):
                    match_found = True
                    break
            
            if not match_found:
                return False, f"Access denied: path is not in allowed scopes {allowed_scopes}"

        return True, ""
    except Exception as e:
        return False, f"Path validation error: {str(e)}"

def vpath_to_realpath(vpath: Path, vroot: str, base_dir: str) -> Path:
    parts = list(vpath.parts)
    if parts and parts[0] == "/":
        parts = parts[1:]
    
    if parts and parts[0] == vroot.lstrip("/"):
        parts = parts[1:]

    return Path(base_dir) / Path(*parts)


class ShellResult(NamedTuple):
    returncode: Optional[int]
    stdout: str
    stderr: str

def run_sandboxed_command(command: str, scopes: Optional[List[str]] = None) -> ShellResult:
    import subprocess
    workspace_path = os.path.abspath(WORKSPACE_DIR)
    reference_path = os.path.abspath(REPOSITORIES_DIR)

    bwrap_args = [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--unshare-all",
        "--bind", workspace_path, WORKSPACE_VROOT,
        "--chdir", "/",
    ]

    if scopes:
        for scope in scopes:
            scope_path = os.path.join(reference_path, scope)
            if os.path.isdir(scope_path):
                bwrap_args.extend(["--ro-bind", scope_path, f"{REPOSITORIES_VROOT}/{scope}"])
    else:
        for entry in os.scandir(reference_path):
            if entry.is_dir() and not entry.name.startswith('.'):
                bwrap_args.extend(["--ro-bind", entry.path, f"{REPOSITORIES_VROOT}/{entry.name}"])

    bwrap_args.extend([
        "bash", "-c", command
    ])

    try:
        result = subprocess.run(
            bwrap_args,
            capture_output=True,
            text=True,
            timeout=30,
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
            stderr="Command timed out",
        )
    except Exception as e:
        return ShellResult(
            returncode=None,
            stdout='',
            stderr=str(e),
        )
