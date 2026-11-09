import json
import re
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database
from domain import ProjectConfig, ProjectRepositoryAccess, RepositoryAccessEntry
from inference.registry import registry
from log_config import get_logger
from repositories import validate_repository_access

logger = get_logger(__name__)
router = APIRouter()


class CreateProjectRequest(BaseModel):
    display_name: str
    internal_name: str
    description: str = ''
    instructions: Optional[str] = None
    default_agent: Optional[int] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: dict[str, Any] = {}
    repository_access: list[RepositoryAccessEntry] = []


class UpdateProjectRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    default_agent: Optional[int] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: Optional[dict[str, Any]] = None
    repository_access: Optional[list[RepositoryAccessEntry]] = None


def _access_from_row(row: tuple[Any, ...]) -> ProjectRepositoryAccess:
    return ProjectRepositoryAccess(
        id=row[0],
        project_id=row[1],
        repository_id=row[2],
        repository_internal_name=row[3],
        security_policy_override=row[4],
    )


async def _fetch_repository_access(
    cur: Any,
    project_id: int,
) -> list[ProjectRepositoryAccess]:
    await cur.execute(
        """
        SELECT pra.id, pra.project_id, pra.repository_id, r.internal_name, pra.security_policy_override
        FROM project_repository_access pra
        JOIN repositories r ON r.id = pra.repository_id
        WHERE pra.project_id = %s
        """,
        (project_id,),
    )
    return [_access_from_row(row) for row in await cur.fetchall()]


def _project_from_row(
    row: tuple[Any, ...],
    repository_access: list[ProjectRepositoryAccess],
) -> ProjectConfig:
    return ProjectConfig(
        id=row[0],
        display_name=row[1],
        internal_name=row[2],
        description=row[3],
        instructions=row[4],
        default_agent=row[5],
        provider_key=row[6],
        model_id=row[7],
        inference_config=row[8] if row[8] else {},
        repository_access=repository_access,
    )


async def _validate_default_agent(agent_id: Optional[int]) -> None:
    if agent_id is None:
        return
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM agents WHERE id = %s", (agent_id,))
        if await cur.fetchone() is None:
            raise ValueError(f"Agent {agent_id} not found")


async def _replace_repository_access(
    cur: Any,
    project_id: int,
    entries: list[RepositoryAccessEntry],
) -> None:
    await cur.execute("DELETE FROM project_repository_access WHERE project_id = %s", (project_id,))
    for access in entries:
        if not access.repository_id:
            continue
        await cur.execute(
            """
            INSERT INTO project_repository_access
                (project_id, repository_id, security_policy_override)
            VALUES (%s, %s, %s)
            """,
            (project_id, access.repository_id, access.security_policy_override),
        )


def _validate_model_settings(provider_key: Optional[str], model_id: Optional[str]) -> None:
    if (provider_key is None) != (model_id is None):
        raise ValueError('provider_key and model_id must be provided together')
    if provider_key is None or model_id is None:
        return
    if not provider_key or not model_id:
        raise ValueError('provider_key and model_id cannot be empty')
    try:
        registry.get(provider_key)
    except KeyError:
        raise ValueError(f"Provider '{provider_key}' is not registered")


async def get_projects() -> list[ProjectConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT p.id, p.display_name, p.internal_name, p.description, p.instructions,
                   p.default_agent_id, p.provider_key, p.model_id, p.inference_config
            FROM projects p
            ORDER BY p.display_name
            """
        )
        rows = await cur.fetchall()
        await cur.execute(
            """
            SELECT pra.id, pra.project_id, pra.repository_id, r.internal_name, pra.security_policy_override
            FROM project_repository_access pra
            JOIN repositories r ON r.id = pra.repository_id
            ORDER BY pra.project_id, pra.id
            """
        )
        access_by_project: dict[int, list[ProjectRepositoryAccess]] = {}
        for access_row in await cur.fetchall():
            access = _access_from_row(access_row)
            access_by_project.setdefault(access.project_id, []).append(access)
        return [
            _project_from_row(row, access_by_project.get(row[0], []))
            for row in rows
        ]


async def get_project_by_name(name: str) -> Optional[ProjectConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT p.id, p.display_name, p.internal_name, p.description, p.instructions,
                   p.default_agent_id, p.provider_key, p.model_id, p.inference_config
            FROM projects p
            WHERE p.internal_name = %s
            """,
            (name,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return _project_from_row(row, await _fetch_repository_access(cur, row[0]))


async def resolve_project_id(name: str) -> Optional[int]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id FROM projects WHERE internal_name = %s", (name,))
        row = await cur.fetchone()
        return row[0] if row else None


async def create_project(payload: CreateProjectRequest) -> ProjectConfig:
    display_name = payload.display_name.strip()
    internal_name = payload.internal_name.strip()
    description = payload.description.strip()
    instructions = payload.instructions.strip() if payload.instructions else None
    provider_key = payload.provider_key.strip() if payload.provider_key else None
    model_id = payload.model_id.strip() if payload.model_id else None

    if not display_name or not internal_name:
        raise ValueError('display_name and internal_name are required')
    if not re.match(r'^[a-zA-Z0-9_-]+$', internal_name):
        raise ValueError('internal_name must contain only letters, numbers, hyphens and underscores')

    await _validate_default_agent(payload.default_agent)
    _validate_model_settings(provider_key, model_id)
    await validate_repository_access(payload.repository_access)

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO projects
                (display_name, internal_name, description, instructions, default_agent_id,
                 provider_key, model_id, inference_config)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                display_name,
                internal_name,
                description,
                instructions,
                payload.default_agent,
                provider_key,
                model_id,
                json.dumps(payload.inference_config),
            ),
        )
        row = await cur.fetchone()
        assert row is not None
        project_id = row[0]

        await _replace_repository_access(cur, project_id, payload.repository_access)

        await conn.commit()

    result = await get_project_by_name(internal_name)
    assert result is not None
    return result


async def update_project(name: str, payload: UpdateProjectRequest) -> ProjectConfig:
    existing = await get_project_by_name(name)
    if not existing:
        raise ValueError(f"Project '{name}' not found")

    fields = payload.model_fields_set

    display_name = payload.display_name if payload.display_name is not None else existing.display_name
    display_name = display_name.strip()
    if not display_name:
        raise ValueError('display_name cannot be empty')

    description = payload.description if payload.description is not None else existing.description
    instructions = payload.instructions if 'instructions' in fields else existing.instructions
    if instructions is not None:
        instructions = instructions.strip() or None

    default_agent = payload.default_agent if 'default_agent' in fields else existing.default_agent
    await _validate_default_agent(default_agent)

    if 'provider_key' in fields or 'model_id' in fields:
        provider_key = payload.provider_key if 'provider_key' in fields else existing.provider_key
        model_id = payload.model_id if 'model_id' in fields else existing.model_id
        provider_key = provider_key.strip() if provider_key else None
        model_id = model_id.strip() if model_id else None
    else:
        provider_key = existing.provider_key
        model_id = existing.model_id
    _validate_model_settings(provider_key, model_id)

    inference_config = payload.inference_config if payload.inference_config is not None else existing.inference_config

    if payload.repository_access is not None:
        repository_access = payload.repository_access
        await validate_repository_access(repository_access)
    else:
        repository_access = [
            RepositoryAccessEntry(
                repository_id=entry.repository_id,
                security_policy_override=entry.security_policy_override,
            )
            for entry in existing.repository_access
        ]

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            UPDATE projects
            SET display_name = %s, description = %s, instructions = %s, default_agent_id = %s,
                provider_key = %s, model_id = %s, inference_config = %s
            WHERE id = %s
            """,
            (
                display_name,
                description,
                instructions,
                default_agent,
                provider_key,
                model_id,
                json.dumps(inference_config),
                existing.id,
            ),
        )

        await _replace_repository_access(cur, existing.id, repository_access)

        await conn.commit()

    result = await get_project_by_name(name)
    assert result is not None
    return result


async def delete_project(name: str) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM projects WHERE internal_name = %s RETURNING id", (name,))
        row = await cur.fetchone()
        return row is not None


@router.get('/api/projects')
async def list_projects():
    projects = await get_projects()
    return JSONResponse(content={
        "projects": [project.model_dump() for project in projects]
    })


@router.post('/api/projects')
async def create_project_endpoint(payload: CreateProjectRequest):
    internal_name = payload.internal_name.strip()
    if internal_name and await resolve_project_id(internal_name) is not None:
        return JSONResponse(
            status_code=409,
            content={'error': f"project with internal_name '{internal_name}' already exists"},
        )
    try:
        project = await create_project(payload)
        return JSONResponse(content=project.model_dump(), status_code=201)
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})


@router.put('/api/projects/{name}')
async def update_project_endpoint(name: str, payload: UpdateProjectRequest):
    try:
        project = await update_project(name, payload)
        return JSONResponse(content=project.model_dump())
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})


@router.delete('/api/projects/{name}')
async def delete_project_endpoint(name: str):
    deleted = await delete_project(name)
    if not deleted:
        return JSONResponse(status_code=404, content={'error': 'project not found'})
    return JSONResponse(content={'status': 'deleted', 'internal_name': name})
