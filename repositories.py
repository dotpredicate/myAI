import os
import re
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
import database
import system
from system import RepositoryConfig, REPOSITORIES_VROOT
from log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def get_repo_from_vpath(vpath: Path) -> Optional[RepositoryConfig]:
    """Extract repo internal_name from a vpath like /repositories/myproject/... and return its RepositoryConfig."""
    try:
        vroot = Path(REPOSITORIES_VROOT)
        rel = vpath.relative_to(vroot)
        repo_name = rel.parts[0]
        return system.get_repo_by_name(repo_name)
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


@router.get('/api/repositories')
async def list_repositories():
    repos = system.get_repositories()
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

    existing = system.get_repo_by_name(internal_name)
    if existing:
        return JSONResponse(status_code=409, content={'error': f'repository with internal_name "{internal_name}" already exists'})

    repo_type = payload.get('type')
    if repo_type not in (None, 'plain', 'git'):
        return JSONResponse(status_code=400, content={'error': 'type must be "plain" or "git" if provided'})
    if not repo_type:
        repo_type = system._auto_detect_repo_type(path)

    security_policy = payload.get('security', 'read-only')
    if security_policy not in ('read-only', 'privileged-write', 'write'):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (display_name, internal_name, repo_type, path, security_policy) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (display_name, internal_name, repo_type, path, security_policy)
        )
        row = cur.fetchone()
        assert row is not None
        repo_id = row[0]
        conn.commit()

    repo = system.get_repo_by_name(internal_name)
    assert repo is not None
    return JSONResponse(content=repo.model_dump(), status_code=201)


@router.put('/api/repositories/{name}')
async def update_repository(name: str, payload: dict = Body(...)):
    existing = system.get_repo_by_name(name)
    if not existing:
        return JSONResponse(status_code=404, content={'error': 'repository not found'})

    display_name = payload.get('display_name', existing.display_name)
    security_policy = payload.get('security', existing.security_policy)

    if security_policy not in ('read-only', 'privileged-write', 'write'):
        return JSONResponse(status_code=400, content={'error': 'security must be "read-only", "privileged-write", or "write"'})

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE repositories SET display_name = %s, security_policy = %s WHERE id = %s",
            (display_name, security_policy, existing.id)
        )
        conn.commit()

    updated = system.get_repo_by_name(name)
    assert updated is not None
    return JSONResponse(content=updated.model_dump())


@router.delete('/api/repositories/{name}')
async def delete_repository(name: str):
    existing = system.get_repo_by_name(name)
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
async def get_repository_files(repo_name: str):
    files = system.get_repository_files(repo_name)
    return JSONResponse(content={"files": files})
