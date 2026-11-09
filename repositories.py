import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Literal
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
import database
from log_config import get_logger
from domain import RepositoryAccessEntry, SecurityPolicy, RepositoryConfig, scope_policy_is_escalation

logger = get_logger(__name__)

router = APIRouter()

async def get_repositories() -> List[RepositoryConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories ORDER BY display_name")
        rows = await cur.fetchall()
        return [RepositoryConfig(id=r[0], display_name=r[1], internal_name=r[2], repo_type=r[3], path=r[4], security_policy=r[5]) for r in rows]


async def get_repo_by_name(name: str) -> Optional[RepositoryConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories WHERE internal_name = %s", (name,))
        row = await cur.fetchone()
        if row:
            return RepositoryConfig(id=row[0], display_name=row[1], internal_name=row[2], repo_type=row[3], path=row[4], security_policy=row[5])
        return None


async def get_repo_by_id(repo_id: int) -> Optional[RepositoryConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories WHERE id = %s", (repo_id,))
        row = await cur.fetchone()
        if row:
            return RepositoryConfig(id=row[0], display_name=row[1], internal_name=row[2], repo_type=row[3], path=row[4], security_policy=row[5])
        return None


async def validate_repository_access(entries: list[RepositoryAccessEntry]) -> None:
    """Reject unknown, duplicated or escalating repository access entries."""
    seen: set[int] = set()
    for entry in entries:
        if not entry.repository_id:
            continue
        if entry.repository_id in seen:
            raise ValueError(f"Repository {entry.repository_id} is listed more than once")
        seen.add(entry.repository_id)
        repo = await get_repo_by_id(entry.repository_id)
        if repo is None:
            raise ValueError(f"Repository {entry.repository_id} not found")
        if entry.security_policy_override is not None:
            if scope_policy_is_escalation(entry.security_policy_override, repo.security_policy):
                raise ValueError(
                    f"Security policy override '{entry.security_policy_override}' is not allowed "
                    f"for repository '{repo.internal_name}' (base policy: {repo.security_policy.value})"
                )


async def _auto_detect_repo_type(path: str) -> Literal['plain', 'git']:
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["git", "-C", path, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return 'git'
    except Exception:
        pass
    return 'plain'


@dataclass(frozen=True)
class RepoScan:
    status: Literal['ok', 'missing', 'error']
    files: List[Tuple[str, Path]]
    error: str = ''


async def scan_repo_documents(repo: RepositoryConfig) -> RepoScan:
    repo_path = Path(repo.path)
    if not repo_path.is_dir():
        return RepoScan('missing', [])

    try:
        if repo.repo_type == 'git':
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", str(repo_path), "ls-files"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return RepoScan('error', [], result.stderr.strip() or 'git ls-files failed')
            rel_names = [f for f in result.stdout.splitlines() if f]
        else:
            errors: list[OSError] = []
            rel_names = []
            for root, _, fnames in os.walk(str(repo_path), onerror=errors.append):
                for fname in fnames:
                    rel = Path(root) / fname
                    rel_names.append(str(rel.relative_to(repo_path)))
            if errors:
                return RepoScan('error', [], str(errors[0]))

        files = [(name, repo_path / name) for name in sorted(rel_names)]
        return RepoScan('ok', [(name, path) for name, path in files if path.is_file()])
    except Exception as exc:
        return RepoScan('error', [], str(exc))


async def get_repo_documents(repo_name: str) -> List[Tuple[str, Path]]:
    repo = await get_repo_by_name(repo_name)
    if not repo:
        return []
    scan = await scan_repo_documents(repo)
    return scan.files if scan.status == 'ok' else []


async def get_repository_files(repo_name: str) -> List[str]:
    """List file names in a repository directory (relative paths)."""
    return [rel for rel, _ in await get_repo_documents(repo_name)]


@router.get('/api/repositories')
async def list_repositories():
    repos = await get_repositories()
    return JSONResponse(content={
        "repositories": [r.model_dump() for r in repos]
    })


@router.post('/api/repositories')
async def create_repository(payload: dict = Body(...)):
    display_name = payload.get('display_name', '').strip()
    internal_name = payload.get('internal_name', '').strip()
    path = payload.get('path', '').strip()

    if not display_name or not internal_name or not path:
        return JSONResponse(status_code=400, content={'error': 'display_name, internal_name and path are required'})

    if not re.match(r'^[a-zA-Z0-9_-]+$', internal_name):
        return JSONResponse(status_code=400, content={'error': 'internal_name must contain only letters, numbers, hyphens and underscores'})

    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return JSONResponse(status_code=400, content={'error': f'path does not exist or is not a directory: {path}'})

    existing = await get_repo_by_name(internal_name)
    if existing:
        return JSONResponse(status_code=409, content={'error': f'repository with internal_name "{internal_name}" already exists'})

    repo_type = payload.get('type')
    if repo_type not in (None, 'plain', 'git'):
        return JSONResponse(status_code=400, content={'error': 'type must be "plain" or "git" if provided'})
    if not repo_type:
        repo_type = await _auto_detect_repo_type(path)

    security_policy = payload.get('security', 'read-only')
    if security_policy not in list(SecurityPolicy):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO repositories (display_name, internal_name, repo_type, path, security_policy) VALUES (%s, %s, %s, %s, %s)",
            (display_name, internal_name, repo_type, path, security_policy)
        )

    repo = await get_repo_by_name(internal_name)
    assert repo is not None
    return JSONResponse(content=repo.model_dump(), status_code=201)


@router.put('/api/repositories/{name}')
async def update_repository(name: str, payload: dict = Body(...)):
    existing = await get_repo_by_name(name)
    if not existing:
        return JSONResponse(status_code=404, content={'error': 'repository not found'})

    display_name = payload.get('display_name', existing.display_name)
    security_policy = payload.get('security', existing.security_policy)

    if security_policy not in list(SecurityPolicy):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "UPDATE repositories SET display_name = %s, security_policy = %s WHERE id = %s",
            (display_name, security_policy, existing.id)
        )

    updated = await get_repo_by_name(name)
    assert updated is not None
    return JSONResponse(content=updated.model_dump())


@router.delete('/api/repositories/{name}')
async def delete_repository(name: str):
    existing = await get_repo_by_name(name)
    if not existing:
        return JSONResponse(status_code=404, content={'error': 'repository not found'})

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM repositories WHERE id = %s", (existing.id,))

    return JSONResponse(content={'status': 'deleted', 'internal_name': name})


@router.get('/api/browse')
async def browse_directory(path: str, include: list[str] = Query(['files', 'folders'])):
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(abs_path):
        return JSONResponse(status_code=400, content={'error': 'path does not exist or is not a directory'})

    want_files = 'files' in include
    want_folders = 'folders' in include
    entries = []
    try:
        with os.scandir(abs_path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and want_folders:
                    entries.append({'name': entry.name, 'type': 'folder', 'path': entry.path})
                elif entry.is_file(follow_symlinks=False) and want_files:
                    entries.append({'name': entry.name, 'type': 'file', 'path': entry.path})
    except PermissionError:
        return JSONResponse(status_code=403, content={'error': 'permission denied'})

    entries.sort(key=lambda e: (0 if e['type'] == 'folder' else 1, e['name'].lower()))
    return JSONResponse(content={'path': abs_path, 'entries': entries})


@router.get('/api/repositories/{repo_name}/files')
async def list_repository_files(repo_name: str):
    files = await get_repository_files(repo_name)
    return JSONResponse(content={"files": files})
