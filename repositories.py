import os
import re
import subprocess
from pathlib import Path
from enum import StrEnum
from typing import Optional, List, Tuple, Literal
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database
from log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

class SecurityPolicy(StrEnum):
    READ_ONLY = 'read-only'
    PRIVILEGED_WRITE = 'privileged-write'
    WRITE = 'write'

class RepositoryConfig(BaseModel):
    id: int
    display_name: str
    internal_name: str
    repo_type: Literal['plain', 'git']
    path: str
    security_policy: SecurityPolicy

class ScopeSpec(BaseModel):
    internal_name: str
    security_policy: SecurityPolicy

def get_repositories() -> List[RepositoryConfig]:
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories ORDER BY display_name")
        rows = cur.fetchall()
        return [RepositoryConfig(id=r[0], display_name=r[1], internal_name=r[2], repo_type=r[3], path=r[4], security_policy=r[5]) for r in rows]


def get_repo_by_name(name: str) -> Optional[RepositoryConfig]:
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories WHERE internal_name = %s", (name,))
        row = cur.fetchone()
        if row:
            return RepositoryConfig(id=row[0], display_name=row[1], internal_name=row[2], repo_type=row[3], path=row[4], security_policy=row[5])
        return None


def get_repo_by_id(repo_id: int) -> Optional[RepositoryConfig]:
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, display_name, internal_name, repo_type, path, security_policy FROM repositories WHERE id = %s", (repo_id,))
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


@router.get('/api/repositories')
async def list_repositories():
    repos = get_repositories()
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

    existing = get_repo_by_name(internal_name)
    if existing:
        return JSONResponse(status_code=409, content={'error': f'repository with internal_name "{internal_name}" already exists'})

    repo_type = payload.get('type')
    if repo_type not in (None, 'plain', 'git'):
        return JSONResponse(status_code=400, content={'error': 'type must be "plain" or "git" if provided'})
    if not repo_type:
        repo_type = _auto_detect_repo_type(path)

    security_policy = payload.get('security', 'read-only')
    if security_policy not in list(SecurityPolicy):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (display_name, internal_name, repo_type, path, security_policy) VALUES (%s, %s, %s, %s, %s)",
            (display_name, internal_name, repo_type, path, security_policy)
        )
        row = cur.fetchone()
        assert row is not None
        conn.commit()

    repo = get_repo_by_name(internal_name)
    assert repo is not None
    return JSONResponse(content=repo.model_dump(), status_code=201)


@router.put('/api/repositories/{name}')
async def update_repository(name: str, payload: dict = Body(...)):
    existing = get_repo_by_name(name)
    if not existing:
        return JSONResponse(status_code=404, content={'error': 'repository not found'})

    display_name = payload.get('display_name', existing.display_name)
    security_policy = payload.get('security', existing.security_policy)

    if security_policy not in list(SecurityPolicy):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE repositories SET display_name = %s, security_policy = %s WHERE id = %s",
            (display_name, security_policy, existing.id)
        )
        conn.commit()

    updated = get_repo_by_name(name)
    assert updated is not None
    return JSONResponse(content=updated.model_dump())


@router.delete('/api/repositories/{name}')
async def delete_repository(name: str):
    existing = get_repo_by_name(name)
    if not existing:
        return JSONResponse(status_code=404, content={'error': 'repository not found'})

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM repositories WHERE id = %s", (existing.id,))
        conn.commit()

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
    files = get_repository_files(repo_name)
    return JSONResponse(content={"files": files})