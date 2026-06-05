import os
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple, Literal

from pydantic import BaseModel
import database


# Paths and configuration
REPOSITORIES_VROOT = "/repositories"
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/.myai/workspace'))
WORKSPACE_VROOT = "/workspace"

os.makedirs(WORKSPACE_DIR, exist_ok=True)


class RepositoryConfig(BaseModel):
    id: int
    display_name: str
    internal_name: str
    repo_type: Literal['plain', 'git']
    path: str
    security_policy: Literal['read-only', 'privileged-write', 'write']


def get_repositories() -> List[RepositoryConfig]:
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories ORDER BY display_name")
        rows = cur.fetchall()
        return [RepositoryConfig(id=r[0], display_name=r[1], internal_name=r[2], repo_type=r[3], path=r[4], security_policy=r[5]) for r in rows]


def get_repo_by_name(name: str) -> Optional[RepositoryConfig]:
    """Look up a repository by its internal_name."""
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories WHERE internal_name = %s", (name,))
        row = cur.fetchone()
        if row:
            return RepositoryConfig(id=row[0], display_name=row[1], internal_name=row[2], repo_type=row[3], path=row[4], security_policy=row[5])
        return None


def _auto_detect_repo_type(path: str) -> Literal['plain', 'git']:
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return 'git'
    except Exception:
        pass
    return 'plain'


def get_repo_documents(repo_name: str) -> List[Tuple[str, Path]]:
    """Returns list of (relative_path, full_path) for every file in a repository.
    Supports both git repos (using ls-files) and non-git repos (os.walk)."""
    repo = get_repo_by_name(repo_name)
    if not repo:
        return []
    repo_path = Path(repo.path)
    if not repo_path.is_dir():
        return []

    try:
        if repo.repo_type == 'git':
            result = subprocess.run(
                ["git", "-C", str(repo_path), "ls-files"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                return []
            rel_names = [f for f in result.stdout.splitlines() if f]
        else:
            rel_names = []
            for root, _, fnames in os.walk(str(repo_path)):
                for fname in fnames:
                    rel = Path(root) / fname
                    rel_names.append(str(rel.relative_to(repo_path)))

        return [(name, repo_path / name) for name in sorted(rel_names)]
    except Exception:
        return []


def get_repository_files(repo_name: str) -> List[str]:
    """List file names in a repository directory (relative paths)."""
    return [rel for rel, _ in get_repo_documents(repo_name)]


def is_safe_vpath(vpath: Path, expected_vroot: Path, allowed_scopes: Optional[List[str]] = None) -> tuple[bool, str]:
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
            repo = get_repo_by_name(scope)
            if repo is None:
                continue
            scope_path = repo.path
            if not os.path.isdir(scope_path):
                continue
            if repo.security_policy == 'write':
                bwrap_args.extend(["--bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])
            else:
                bwrap_args.extend(["--ro-bind", scope_path, f"{REPOSITORIES_VROOT}/{repo.internal_name}"])
    else:
        # No scopes defined
        pass

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