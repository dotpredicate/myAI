import os
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional, List
from repositories import get_repo_by_name, RepositoryConfig, SecurityPolicy
from domain import ScopeSpec


# Paths and configuration
REPOSITORIES_VROOT = "/repositories"
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/.myai/workspace'))
WORKSPACE_VROOT = "/workspace"

os.makedirs(WORKSPACE_DIR, exist_ok=True)


def is_safe_vpath(vpath: Path, expected_vroot: Path, allowed_scopes: Optional[List[ScopeSpec]] = None) -> tuple[bool, str]:
    try:
        parts = vpath.parts
        if ".." in parts:
            return False, "Path traversal detected: '..' not allowed"

        if not vpath.is_relative_to(expected_vroot):
            return False, f"Path must be under {expected_vroot}"

        if allowed_scopes:
            vpath_str = str(vpath)
            match_found = False
            for scope in allowed_scopes:
                prefix = f"{expected_vroot}/{scope.internal_name}"
                if vpath_str.startswith(prefix):
                    match_found = True
                    break

            if not match_found:
                names = [s.internal_name for s in allowed_scopes]
                return False, f"Access denied: path is not in allowed scopes {names}"

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


def get_repo_from_vpath(vpath: Path) -> Optional[RepositoryConfig]:
    """Extract repo internal_name from a vpath like /repositories/myproject/... and return its RepositoryConfig."""

    try:
        vroot = Path(REPOSITORIES_VROOT)
        rel = vpath.relative_to(vroot)
        repo_name = rel.parts[0]
        return get_repo_by_name(repo_name)
    except (ValueError, IndexError):
        return None


def resolve_repo_vpath(vpath: Path) -> Optional[Path]:
    """Resolve a vpath under /repositories/ to a real filesystem path using the repo's stored path."""

    repo = get_repo_from_vpath(vpath)
    if not repo:
        return None
    try:
        vroot = Path(REPOSITORIES_VROOT)
        rel = vpath.relative_to(vroot)
        # rel looks like: repo_internal_name/path/to/file, strip repo_internal_name
        rel_parts = rel.parts[1:]
        return Path(repo.path) / Path(*rel_parts)
    except (ValueError, IndexError):
        return None


class ShellResult(NamedTuple):
    returncode: Optional[int]
    stdout: str
    stderr: str


def run_sandboxed_command(command: str, scopes: Optional[List[ScopeSpec]] = None) -> ShellResult:
    workspace_path = os.path.abspath(WORKSPACE_DIR)

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

    if not scopes:
        scopes = []
    
    for scope in scopes:
        repo = get_repo_by_name(scope.internal_name)
        if repo is None:
            continue
        scope_path = repo.path
        if not os.path.isdir(scope_path):
            continue
        effective_policy = scope.security_policy_override if scope.security_policy_override is not None else repo.security_policy
        if effective_policy == SecurityPolicy.WRITE:
            bwrap_args.extend(["--bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])
        else:
            bwrap_args.extend(["--ro-bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])

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