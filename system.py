import asyncio
from contextlib import suppress
import os
import shutil
from pathlib import Path
from typing import NamedTuple, Optional, List
from domain import ScopeSpec, SecurityPolicy, RepositoryConfig
from repositories import get_repo_by_name


# Paths and configuration
REPOSITORIES_VROOT = "/repositories"
WORKSPACES_ROOT = Path(os.getenv('MYAI_WORKSPACES_DIR', os.path.expanduser('~/.myai/workspaces')))
WORKSPACE_VROOT = "/workspace"


def _workspace_dir_for(conversation_id: int) -> Path:
    path = WORKSPACES_ROOT / str(conversation_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_workspace(conversation_id: int) -> None:
    shutil.rmtree(WORKSPACES_ROOT / str(conversation_id), ignore_errors=True)


def is_safe_vpath(vpath: Path, expected_vroot: Path, allowed_scopes: Optional[List[ScopeSpec]] = None) -> tuple[bool, str]:
    try:
        parts = vpath.parts
        if ".." in parts:
            return False, "Path traversal detected: '..' not allowed"

        if not vpath.is_relative_to(expected_vroot):
            return False, f"Path must be under {expected_vroot}"

        if allowed_scopes is not None:
            for scope in allowed_scopes:
                scope_root = expected_vroot / scope.internal_name
                if vpath == scope_root or vpath.is_relative_to(scope_root):
                    return True, ""

            names = [s.internal_name for s in allowed_scopes]
            return False, f"Access denied: path is not in allowed scopes {names}"

        return True, ""
    except Exception as e:
        return False, f"Path validation error: {str(e)}"


def is_safe_realpath(realpath: Path, expected_root: Path) -> bool:
    try:
        return realpath.resolve(strict=False).is_relative_to(expected_root.resolve(strict=False))
    except (OSError, RuntimeError):
        return False


def vpath_to_realpath(vpath: Path, vroot: str, base_dir: str) -> Path:
    parts = list(vpath.parts)
    if parts and parts[0] == "/":
        parts = parts[1:]

    if parts and parts[0] == vroot.lstrip("/"):
        parts = parts[1:]

    return Path(base_dir) / Path(*parts)


def resolve_workspace_vpath(vpath: Path, conversation_id: int) -> Optional[Path]:
    safe, _ = is_safe_vpath(vpath, Path(WORKSPACE_VROOT))
    if not safe:
        return None

    root = _workspace_dir_for(conversation_id)
    realpath = vpath_to_realpath(vpath, WORKSPACE_VROOT, str(root))
    if not is_safe_realpath(realpath, root):
        return None
    return realpath


async def get_repo_from_vpath(vpath: Path) -> Optional[RepositoryConfig]:
    """Extract repo internal_name from a vpath like /repositories/myproject/... and return its RepositoryConfig."""

    try:
        vroot = Path(REPOSITORIES_VROOT)
        rel = vpath.relative_to(vroot)
        repo_name = rel.parts[0]
        return await get_repo_by_name(repo_name)
    except (ValueError, IndexError):
        return None


async def resolve_repo_vpath(vpath: Path) -> Optional[Path]:
    """Resolve a vpath under /repositories/ to a real filesystem path using the repo's stored path."""

    repo = await get_repo_from_vpath(vpath)
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


async def run_sandboxed_command(command: str, scopes: Optional[List[ScopeSpec]], conversation_id: int) -> ShellResult:
    workspace_path = os.path.abspath(str(_workspace_dir_for(conversation_id)))

    bwrap_args = [
        "bwrap",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "HOME", WORKSPACE_VROOT,
        "--setenv", "TMPDIR", "/tmp",
        "--unshare-all",
        "--bind", workspace_path, WORKSPACE_VROOT,
        "--chdir", "/",
    ]

    if not scopes:
        scopes = []
    
    for scope in scopes:
        repo = await get_repo_by_name(scope.internal_name)
        if repo is None:
            continue
        scope_path = repo.path
        if not os.path.isdir(scope_path):
            continue
        effective_policy = scope.security_policy if scope.security_policy is not None else repo.security_policy
        if effective_policy == SecurityPolicy.WRITE:
            bwrap_args.extend(["--bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])
        else:
            bwrap_args.extend(["--ro-bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])

    bwrap_args.extend([
        "bash", "-c", command
    ])

    try:
        process = await asyncio.create_subprocess_exec(
            *bwrap_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.communicate()
            return ShellResult(
                returncode=None,
                stdout="",
                stderr="Command timed out",
            )
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.communicate()
            raise

        return ShellResult(
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )
    except Exception as e:
        return ShellResult(
            returncode=None,
            stdout='',
            stderr=str(e),
        )
