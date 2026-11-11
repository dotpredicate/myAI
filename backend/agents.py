import json
import re
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import database
from backend.domain import AgentConfig
from backend.inference.registry import registry
from backend.log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CreateAgentRequest(BaseModel):
    display_name: str
    internal_name: str
    description: str = ''
    instructions: Optional[str] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: dict[str, Any] = {}
    spawnable_agents: list[str] = []


class UpdateAgentRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: Optional[dict[str, Any]] = None
    spawnable_agents: Optional[list[str]] = None


def _agent_from_row(row: tuple[Any, ...]) -> AgentConfig:
    return AgentConfig(
        id=row[0],
        display_name=row[1],
        internal_name=row[2],
        description=row[3],
        instructions=row[4],
        provider_key=row[5],
        model_id=row[6],
        inference_config=row[7] if row[7] else {},
        spawnable_agents=row[8] if row[8] else [],
    )


def _validate_model(provider_key: Optional[str], model_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    provider_key = provider_key.strip() if provider_key else None
    model_id = model_id.strip() if model_id else None
    if (provider_key is None) != (model_id is None):
        raise ValueError('provider_key and model_id must be provided together')
    if provider_key is None:
        return None, None
    try:
        registry.get(provider_key)
    except KeyError:
        raise ValueError(f"Provider '{provider_key}' is not registered")
    return provider_key, model_id


async def get_agents() -> list[AgentConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, display_name, internal_name, description, instructions,
                   provider_key, model_id, inference_config, spawnable_agents
            FROM agents
            ORDER BY display_name
            """
        )
        return [_agent_from_row(row) for row in await cur.fetchall()]


async def get_agent_by_name(name: str) -> Optional[AgentConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, display_name, internal_name, description, instructions,
                   provider_key, model_id, inference_config, spawnable_agents
            FROM agents WHERE internal_name = %s
            """,
            (name,),
        )
        row = await cur.fetchone()
        return _agent_from_row(row) if row else None


async def provider_in_use(provider_key: str) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT 1 FROM agents WHERE provider_key = %s
            UNION ALL
            SELECT 1 FROM conversations WHERE provider_key = %s
            LIMIT 1
            """,
            (provider_key, provider_key),
        )
        return await cur.fetchone() is not None


async def create_agent(payload: CreateAgentRequest) -> AgentConfig:
    display_name = payload.display_name.strip()
    internal_name = payload.internal_name.strip()
    description = payload.description.strip()
    instructions = payload.instructions.strip() if payload.instructions else ''
    provider_key, model_id = _validate_model(payload.provider_key, payload.model_id)
    spawnable_agents = list(dict.fromkeys(payload.spawnable_agents))

    if not display_name or not internal_name:
        raise ValueError('display_name and internal_name are required')
    if not re.match(r'^[a-zA-Z0-9_-]+$', internal_name):
        raise ValueError('internal_name must contain only letters, numbers, hyphens and underscores')

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM agents WHERE internal_name = %s", (internal_name,))
        if await cur.fetchone():
            raise ValueError(f"Agent '{internal_name}' already exists")
        await cur.execute(
            """
            INSERT INTO agents
                (display_name, internal_name, description, instructions,
                 provider_key, model_id, inference_config, spawnable_agents)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                display_name,
                internal_name,
                description,
                instructions,
                provider_key,
                model_id,
                json.dumps(payload.inference_config),
                json.dumps(spawnable_agents),
            ),
        )
        row = await cur.fetchone()
        assert row is not None
        await conn.commit()

    result = await get_agent_by_name(internal_name)
    assert result is not None
    return result


async def update_agent(name: str, payload: UpdateAgentRequest) -> AgentConfig:
    existing = await get_agent_by_name(name)
    if not existing:
        raise ValueError(f"Agent '{name}' not found")

    fields = payload.model_fields_set
    display_name = (payload.display_name if payload.display_name is not None else existing.display_name).strip()
    if not display_name:
        raise ValueError('display_name cannot be empty')
    description = payload.description if payload.description is not None else existing.description
    instructions = payload.instructions if 'instructions' in fields else existing.instructions
    if instructions is not None:
        instructions = instructions.strip() or None

    if 'provider_key' in fields or 'model_id' in fields:
        provider_key, model_id = _validate_model(payload.provider_key, payload.model_id)
    else:
        provider_key, model_id = existing.provider_key, existing.model_id

    inference_config = (
        payload.inference_config if payload.inference_config is not None else existing.inference_config
    )
    if provider_key is None:
        inference_config = {}
    requested_spawnable = (
        payload.spawnable_agents if payload.spawnable_agents is not None else existing.spawnable_agents
    )
    spawnable_agents = list(dict.fromkeys(requested_spawnable))

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            UPDATE agents
            SET display_name = %s, description = %s, instructions = %s,
                provider_key = %s, model_id = %s, inference_config = %s, spawnable_agents = %s
            WHERE id = %s
            """,
            (
                display_name,
                description,
                instructions,
                provider_key,
                model_id,
                json.dumps(inference_config),
                json.dumps(spawnable_agents),
                existing.id,
            ),
        )
        await conn.commit()

    result = await get_agent_by_name(name)
    assert result is not None
    return result


async def delete_agent(name: str) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM agents WHERE internal_name = %s RETURNING id", (name,))
        return await cur.fetchone() is not None


@router.get('/api/agents')
async def list_agents():
    return JSONResponse(content={'agents': [agent.model_dump() for agent in await get_agents()]})


@router.post('/api/agents')
async def create_agent_endpoint(payload: CreateAgentRequest):
    try:
        agent = await create_agent(payload)
        return JSONResponse(content=agent.model_dump(), status_code=201)
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})


@router.put('/api/agents/{name}')
async def update_agent_endpoint(name: str, payload: UpdateAgentRequest):
    try:
        agent = await update_agent(name, payload)
        return JSONResponse(content=agent.model_dump())
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})


@router.delete('/api/agents/{name}')
async def delete_agent_endpoint(name: str):
    deleted = await delete_agent(name)
    if not deleted:
        return JSONResponse(status_code=404, content={'error': 'agent not found'})
    return JSONResponse(content={'status': 'deleted', 'internal_name': name})
