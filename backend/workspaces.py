import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from backend import database
from backend import paths
from backend import system
from backend.domain import WorkspaceConfig, WorkspaceSource
from backend.log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', value).strip('-').lower()
    return slug or 'workspace'


async def _unique_repo_name(cur, base: str) -> str:
    name = _slug(base)
    candidate = name
    suffix = 2
    while True:
        await cur.execute("SELECT 1 FROM repositories WHERE internal_name = %s", (candidate,))
        if await cur.fetchone() is None:
            return candidate
        candidate = f"{name}-{suffix}"
        suffix += 1


def _source_from_row(row) -> WorkspaceSource:
    return WorkspaceSource(
        repository_id=row[0],
        internal_name=row[1],
        display_name=row[2],
        repo_type=row[3],
        path=row[4],
    )


async def _workspace_from_row(cur, row) -> WorkspaceConfig:
    await cur.execute(
        """
        SELECT r.id, r.internal_name, r.display_name, r.repo_type, r.path
        FROM workspace_sources ws
        JOIN repositories r ON r.id = ws.repository_id
        WHERE ws.workspace_id = %s
        ORDER BY r.display_name
        """,
        (row[0],),
    )
    sources = [_source_from_row(source) for source in await cur.fetchall()]
    return WorkspaceConfig(
        id=row[0],
        name=row[1],
        created_at=row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
        sources=sources,
    )


async def get_workspaces() -> list[WorkspaceConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id, name, created_at FROM workspaces ORDER BY name")
        rows = await cur.fetchall()
        return [await _workspace_from_row(cur, row) for row in rows]


async def get_workspace(workspace_id: int) -> Optional[WorkspaceConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id, name, created_at FROM workspaces WHERE id = %s", (workspace_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return await _workspace_from_row(cur, row)


async def _find_repository(cur, canonical_path: str):
    await cur.execute(
        "SELECT id, display_name, internal_name, repo_type, path FROM repositories WHERE path = %s",
        (canonical_path,),
    )
    return await cur.fetchone()


async def _find_workspace_for_repository(cur, repository_id: int):
    await cur.execute(
        """
        SELECT w.id, w.name, w.created_at
        FROM workspaces w
        JOIN workspace_sources ws ON ws.workspace_id = w.id
        WHERE ws.repository_id = %s
        ORDER BY w.created_at DESC, w.id DESC
        LIMIT 1
        """,
        (repository_id,),
    )
    return await cur.fetchone()


def _guard_path(path: str) -> None:
    home = os.path.realpath(os.path.expanduser('~'))
    myai = os.path.realpath(str(paths.MYAI_DIR))
    if path == '/' or path == home:
        raise ValueError('refusing to register the home or filesystem root as a workspace source')
    if path == myai or Path(path).is_relative_to(Path(myai)):
        raise ValueError('refusing to register the myAI data directory as a workspace source')


def _git_root(path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', '-C', path, 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return os.path.realpath(result.stdout.strip())


async def resolve_workspace(raw_path: str) -> tuple[WorkspaceConfig, bool, bool]:
    """Resolve a path to a workspace, creating the repo and workspace if needed.

    Returns (workspace, workspace_created, repository_created).
    """
    path = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path)))
    if not os.path.isdir(path):
        raise ValueError(f'path is not a directory: {path}')
    _guard_path(path)
    root = _git_root(path) or path
    _guard_path(root)

    workspace_created = False
    repository_created = False
    async with database.mk_conn() as conn:
        await conn.begin_immediate()
        async with conn.cursor() as cur:
            repo = await _find_repository(cur, root)
            if repo is None:
                display_name = Path(root).name or 'source'
                internal_name = await _unique_repo_name(cur, display_name)
                repo_type = 'git' if _git_root(root) else 'plain'
                await cur.execute(
                    "INSERT INTO repositories (display_name, internal_name, repo_type, path) VALUES (%s, %s, %s, %s)",
                    (display_name, internal_name, repo_type, root),
                )
                repo = await _find_repository(cur, root)
                assert repo is not None
                repository_created = True

            workspace_row = await _find_workspace_for_repository(cur, repo[0])
            if workspace_row is None:
                name = repo[1] or Path(root).name or 'Workspace'
                await cur.execute(
                    "INSERT INTO workspaces (name) VALUES (%s) RETURNING id, name, created_at",
                    (name,),
                )
                workspace_row = await cur.fetchone()
                assert workspace_row is not None
                await cur.execute(
                    "INSERT INTO workspace_sources (workspace_id, repository_id) VALUES (%s, %s)",
                    (workspace_row[0], repo[0]),
                )
                workspace_created = True
        await conn.commit()

    workspace = await get_workspace(workspace_row[0])
    assert workspace is not None
    system.space_dir_for(workspace.id)
    try:
        from backend.search import schedule_sync
        schedule_sync()
    except Exception:
        logger.exception('failed to start index synchronization')
    return workspace, workspace_created, repository_created


async def update_workspace(workspace_id: int, name: str) -> Optional[WorkspaceConfig]:
    name = name.strip()
    if not name:
        raise ValueError('name cannot be empty')
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "UPDATE workspaces SET name = %s WHERE id = %s RETURNING id",
            (name, workspace_id),
        )
        if await cur.fetchone() is None:
            return None
    return await get_workspace(workspace_id)


async def delete_workspace(workspace_id: int) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,))
        if await cur.fetchone() is None:
            return False
        await cur.execute(
            "SELECT id FROM workspaces WHERE id != %s ORDER BY id LIMIT 1",
            (workspace_id,),
        )
        replacement = await cur.fetchone()
        if replacement is None:
            raise ValueError('cannot delete the last workspace')
        await cur.execute(
            "UPDATE conversations SET workspace_id = %s WHERE workspace_id = %s",
            (replacement[0], workspace_id),
        )
        await cur.execute("DELETE FROM workspaces WHERE id = %s", (workspace_id,))
        return True


async def add_source(workspace_id: int, repository_id: int) -> WorkspaceConfig:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,))
        if await cur.fetchone() is None:
            raise ValueError(f'workspace {workspace_id} not found')
        await cur.execute("SELECT 1 FROM repositories WHERE id = %s", (repository_id,))
        if await cur.fetchone() is None:
            raise ValueError(f'repository {repository_id} not found')
        await cur.execute(
            "INSERT OR IGNORE INTO workspace_sources (workspace_id, repository_id) VALUES (%s, %s)",
            (workspace_id, repository_id),
        )
    workspace = await get_workspace(workspace_id)
    assert workspace is not None
    return workspace


async def remove_source(workspace_id: int, repository_id: int) -> WorkspaceConfig:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "DELETE FROM workspace_sources WHERE workspace_id = %s AND repository_id = %s",
            (workspace_id, repository_id),
        )
    workspace = await get_workspace(workspace_id)
    if workspace is None:
        raise ValueError(f'workspace {workspace_id} not found')
    return workspace


@router.get('/api/workspaces')
async def list_workspaces():
    return JSONResponse(content={'workspaces': [w.model_dump() for w in await get_workspaces()]})


@router.get('/api/workspaces/{workspace_id}')
async def get_workspace_endpoint(workspace_id: int):
    workspace = await get_workspace(workspace_id)
    if workspace is None:
        return JSONResponse(status_code=404, content={'error': 'workspace not found'})
    return JSONResponse(content=workspace.model_dump())


@router.post('/api/workspaces/resolve')
async def resolve_workspace_endpoint(payload: dict = Body(...)):
    raw_path = str(payload.get('path', '')).strip()
    if not raw_path:
        return JSONResponse(status_code=400, content={'error': 'path is required'})
    try:
        workspace, workspace_created, repository_created = await resolve_workspace(raw_path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})
    return JSONResponse(content={
        'workspace': workspace.model_dump(),
        'created': workspace_created,
        'repository_created': repository_created,
        'sync_started': True,
    })


@router.patch('/api/workspaces/{workspace_id}')
async def update_workspace_endpoint(workspace_id: int, payload: dict = Body(...)):
    try:
        workspace = await update_workspace(workspace_id, str(payload.get('name', '')))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})
    if workspace is None:
        return JSONResponse(status_code=404, content={'error': 'workspace not found'})
    return JSONResponse(content=workspace.model_dump())


@router.delete('/api/workspaces/{workspace_id}')
async def delete_workspace_endpoint(workspace_id: int):
    try:
        deleted = await delete_workspace(workspace_id)
    except ValueError as exc:
        return JSONResponse(status_code=409, content={'error': str(exc)})
    if not deleted:
        return JSONResponse(status_code=404, content={'error': 'workspace not found'})
    return JSONResponse(content={'status': 'deleted', 'id': workspace_id})


@router.post('/api/workspaces/{workspace_id}/sources')
async def add_source_endpoint(workspace_id: int, payload: dict = Body(...)):
    repository_id = payload.get('repository_id')
    if not isinstance(repository_id, int):
        return JSONResponse(status_code=400, content={'error': 'repository_id is required'})
    try:
        workspace = await add_source(workspace_id, repository_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})
    return JSONResponse(content=workspace.model_dump())


@router.delete('/api/workspaces/{workspace_id}/sources/{repository_id}')
async def remove_source_endpoint(workspace_id: int, repository_id: int):
    try:
        workspace = await remove_source(workspace_id, repository_id)
    except ValueError as exc:
        return JSONResponse(status_code=404, content={'error': str(exc)})
    return JSONResponse(content=workspace.model_dump())
