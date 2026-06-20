import json
import re
from typing import Any, Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database
from domain import SecurityPolicy, AgentConfig, AgentRepositoryAccess, scope_policy_is_escalation
from repositories import get_repo_by_id
from inference import registry
from log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

class RepoAccessEntry(BaseModel):
    repository_id: int
    security_policy_override: SecurityPolicy | None = None


class CreateAgentRequest(BaseModel):
    display_name: str
    internal_name: str
    description: str = ''
    instructions: Optional[str] = None
    provider_key: str
    model_id: str
    inference_config: dict[str, Any] = {}
    repository_access: list[RepoAccessEntry] = []


class UpdateAgentRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    instructions: str | None = None
    provider_key: str | None = None
    model_id: str | None = None
    inference_config: dict[str, Any] | None = None
    repository_access: list[RepoAccessEntry] | None = None


async def _fetch_repository_access(agent_id: int) -> list[AgentRepositoryAccess]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT ara.id, ara.agent_id, ara.repository_id, r.internal_name, ara.security_policy_override
            FROM agent_repository_access ara
            JOIN repositories r ON r.id = ara.repository_id
            WHERE ara.agent_id = %s
            """,
            (agent_id,)
        )
        rows = await cur.fetchall()
        return [
            AgentRepositoryAccess(
                id=row[0],
                agent_id=row[1],
                repository_id=row[2],
                repository_internal_name=row[3],
                security_policy_override=row[4],
            )
            for row in rows
        ]


async def get_agents() -> list[AgentConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, display_name, internal_name, description, instructions, provider_key, model_id, inference_config FROM agents"
        )
        rows = await cur.fetchall()
        agents = []
        for row in rows:
            agent_id = row[0]
            agents.append(AgentConfig(
                id=agent_id,
                display_name=row[1],
                internal_name=row[2],
                description=row[3],
                instructions=row[4],
                provider_key=row[5],
                model_id=row[6],
                inference_config=row[7] if row[7] else {},
                repository_access=await _fetch_repository_access(agent_id),
            ))
        return agents


async def get_agent_by_name(name: str) -> AgentConfig | None:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, display_name, internal_name, description, instructions, provider_key, model_id, inference_config FROM agents WHERE internal_name = %s",
            (name,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        agent_id = row[0]
        return AgentConfig(
            id=agent_id,
            display_name=row[1],
            internal_name=row[2],
            description=row[3],
            instructions=row[4],
            provider_key=row[5],
            model_id=row[6],
            inference_config=row[7] if row[7] else {},
            repository_access=await _fetch_repository_access(agent_id),
        )


async def create_agent(payload: CreateAgentRequest) -> AgentConfig:
    display_name = payload.display_name.strip()
    internal_name = payload.internal_name.strip()
    description = payload.description.strip()
    instructions = payload.instructions.strip() if payload.instructions else None
    provider_key = payload.provider_key.strip()
    model_id = payload.model_id.strip()
    inference_config = payload.inference_config
    repository_access = payload.repository_access

    if not display_name or not internal_name or not provider_key or not model_id:
        raise ValueError('display_name, internal_name, provider_key and model_id are required')

    if not re.match(r'^[a-zA-Z0-9_-]+$', internal_name):
        raise ValueError('internal_name must contain only letters, numbers, hyphens and underscores')

    # Validate provider_key exists in registry
    try:
        registry.get(provider_key)
    except KeyError:
        raise ValueError(f"Provider '{provider_key}' is not registered")

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await conn.set_autocommit(False)
        await cur.execute(
            "INSERT INTO agents (display_name, internal_name, description, instructions, provider_key, model_id, inference_config) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (display_name, internal_name, description, instructions, provider_key, model_id, json.dumps(inference_config)),
        )
        row = await cur.fetchone()
        assert row is not None
        agent_id = row[0]

        # Validate and insert repository access entries
        for access in repository_access:
            if access.repository_id:
                # Validate no policy escalation
                repo = await get_repo_by_id(access.repository_id)
                if repo and access.security_policy_override:
                    if scope_policy_is_escalation(access.security_policy_override, repo.security_policy):
                        raise ValueError(f"Security policy override '{access.security_policy_override}' is not allowed for repository '{repo.internal_name}' (base policy: {repo.security_policy.value})")
                await cur.execute(
                    "INSERT INTO agent_repository_access (agent_id, repository_id, security_policy_override) VALUES (%s, %s, %s)",
                    (agent_id, access.repository_id, access.security_policy_override),
                )

        await conn.commit()

    result = await get_agent_by_name(internal_name)
    assert result is not None
    return result


async def update_agent(name: str, payload: UpdateAgentRequest) -> AgentConfig:
    existing = await get_agent_by_name(name)
    if not existing:
        raise ValueError(f"Agent '{name}' not found")

    display_name = payload.display_name if payload.display_name is not None else existing.display_name
    description = payload.description if payload.description is not None else existing.description
    instructions = payload.instructions if payload.instructions is not None else existing.instructions
    provider_key = payload.provider_key if payload.provider_key is not None else existing.provider_key
    model_id = payload.model_id if payload.model_id is not None else existing.model_id
    inference_config = payload.inference_config if payload.inference_config is not None else existing.inference_config
    repository_access = payload.repository_access if payload.repository_access is not None else []

    # Validate provider_key exists in registry
    try:
        registry.get(provider_key)
    except KeyError:
        raise ValueError(f"Provider '{provider_key}' is not registered")

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await conn.set_autocommit(False)
        await cur.execute(
            "UPDATE agents SET display_name = %s, description = %s, instructions = %s, provider_key = %s, model_id = %s, inference_config = %s WHERE id = %s",
            (display_name, description, instructions, provider_key, model_id, json.dumps(inference_config), existing.id),
        )

        await cur.execute("DELETE FROM agent_repository_access WHERE agent_id = %s", (existing.id,))
        for access in repository_access:
            if access.repository_id:
                # Validate no policy escalation
                repo = await get_repo_by_id(access.repository_id)
                if repo and access.security_policy_override:
                    if scope_policy_is_escalation(access.security_policy_override, repo.security_policy):
                        raise ValueError(f"Security policy override '{access.security_policy_override}' is not allowed for repository '{repo.internal_name}' (base policy: {repo.security_policy.value})")
                await cur.execute(
                    "INSERT INTO agent_repository_access (agent_id, repository_id, security_policy_override) VALUES (%s, %s, %s)",
                    (existing.id, access.repository_id, access.security_policy_override),
                )

        await conn.commit()

    result = await get_agent_by_name(name)
    assert result is not None
    return result


async def delete_agent(name: str) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM agents WHERE internal_name = %s RETURNING id", (name,))
        row = await cur.fetchone()
        return row is not None


@router.get('/api/agents')
async def list_agents():
    agents = await get_agents()
    return JSONResponse(content={
        "agents": [a.model_dump() for a in agents]
    })


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