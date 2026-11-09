import json
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import database
from domain import (
    LLAMA_CPP_PROVIDER_KEY,
    InferenceProviderConfig,
    InferenceProviderInput,
    InferenceProviderUpdate,
)
from inference.openai_chat import OpenAICompatibleProvider
from inference.registry import registry
from log_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

_FIELDS = (
    "key, display_name, description, base_url, api_key, models, "
    "timeout_seconds, headers, extra_body, include_reasoning_content"
)


class CreateProviderRequest(InferenceProviderInput):
    key: str


def _config_from_row(row: Any) -> InferenceProviderConfig:
    return InferenceProviderConfig(
        key=row[0],
        display_name=row[1],
        description=row[2] or '',
        base_url=row[3],
        api_key=row[4],
        models=row[5] or [],
        timeout_seconds=row[6],
        headers=row[7] or {},
        extra_body=row[8] or {},
        include_reasoning_content=row[9],
    )


def _provider_response(config: InferenceProviderConfig) -> dict[str, object]:
    return {
        'key': config.key,
        'display_name': config.display_name,
        'description': config.description,
        'base_url': config.base_url,
        'has_api_key': config.has_api_key,
        'models': config.models,
        'timeout_seconds': config.timeout_seconds,
        'extra_body': config.extra_body,
        'include_reasoning_content': config.include_reasoning_content,
    }


async def list_provider_configs() -> list[InferenceProviderConfig]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(f"SELECT {_FIELDS} FROM inference_providers ORDER BY key")
        rows = await cur.fetchall()
    return [_config_from_row(row) for row in rows]


async def get_provider_config(key: str) -> InferenceProviderConfig | None:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            f"SELECT {_FIELDS} FROM inference_providers WHERE key = %s",
            (key,),
        )
        row = await cur.fetchone()
    return _config_from_row(row) if row else None


async def create_provider_config(config: InferenceProviderConfig) -> None:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO inference_providers
                (key, display_name, description, base_url, api_key, models,
                 timeout_seconds, headers, extra_body, include_reasoning_content)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                config.key,
                config.display_name,
                config.description,
                config.base_url,
                config.api_key,
                json.dumps(config.models),
                config.timeout_seconds,
                json.dumps(config.headers),
                json.dumps(config.extra_body),
                config.include_reasoning_content,
            ),
        )
        await conn.commit()


async def update_provider_config(key: str, update: InferenceProviderUpdate) -> None:
    fields: dict[str, object] = update.model_dump(exclude_unset=True)
    if not fields:
        return
    columns: list[str] = []
    values: list[object] = []
    for name, value in fields.items():
        if name in ('models', 'headers', 'extra_body'):
            value = json.dumps(value)
        columns.append(f"{name} = %s")
        values.append(value)
    values.append(key)
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            f"UPDATE inference_providers SET {', '.join(columns)} WHERE key = %s",
            tuple(values),
        )
        await conn.commit()


async def delete_provider_config(key: str) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "DELETE FROM inference_providers WHERE key = %s RETURNING key",
            (key,),
        )
        deleted = await cur.fetchone() is not None
        await conn.commit()
    return deleted


async def reload_providers() -> None:
    """Load OpenAI-compatible providers from the database into the registry."""
    configs = await list_provider_configs()
    for registration in registry.list_registrations():
        if registration.key != LLAMA_CPP_PROVIDER_KEY:
            registry.unregister(registration.key)
    for config in configs:
        if config.key == LLAMA_CPP_PROVIDER_KEY:
            continue
        registry.register(
            config.key,
            config.display_name,
            config.description,
            OpenAICompatibleProvider(config),
        )


@router.get('/api/providers')
async def get_providers():
    configs = {config.key: config for config in await list_provider_configs()}
    providers = []
    for registration in registry.list_registrations():
        config = configs.get(registration.key)
        providers.append({
            "key": registration.key,
            "display_name": config.display_name if config else registration.display_name,
            "description": config.description if config else registration.description,
            "kind": "openai_compatible" if config else "builtin",
            "has_api_key": config.has_api_key if config else False,
        })
    return JSONResponse(content={"providers": providers})


@router.get('/api/providers/{provider_key}')
async def get_provider(provider_key: str):
    config = await get_provider_config(provider_key)
    if config is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    return JSONResponse(content=_provider_response(config))


@router.post('/api/providers')
async def create_provider(payload: CreateProviderRequest):
    key = payload.key.strip()
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', key):
        return JSONResponse(status_code=400, content={'error': 'key must contain only letters, numbers, dots, hyphens and underscores'})
    if key == LLAMA_CPP_PROVIDER_KEY or any(r.key == key for r in registry.list_registrations()):
        return JSONResponse(status_code=409, content={'error': f"provider '{key}' already exists"})
    if await get_provider_config(key) is not None:
        return JSONResponse(status_code=409, content={'error': f"provider '{key}' already exists"})

    config = InferenceProviderConfig(key=key, **payload.model_dump(exclude={'key'}))
    await create_provider_config(config)
    await reload_providers()
    return JSONResponse(content=_provider_response(config), status_code=201)


@router.patch('/api/providers/{provider_key}')
async def update_provider(provider_key: str, payload: InferenceProviderUpdate):
    existing = await get_provider_config(provider_key)
    if existing is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    if not payload.model_fields_set:
        return JSONResponse(content=_provider_response(existing))

    await update_provider_config(provider_key, payload)
    await reload_providers()
    result = await get_provider_config(provider_key)
    if result is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    return JSONResponse(content=_provider_response(result))


@router.delete('/api/providers/{provider_key}')
async def delete_provider(provider_key: str):
    from agents import provider_in_use

    existing = await get_provider_config(provider_key)
    if existing is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    if await provider_in_use(provider_key):
        return JSONResponse(status_code=409, content={'error': 'provider is used by one or more agents or projects'})
    await delete_provider_config(provider_key)
    await reload_providers()
    return JSONResponse(content={'status': 'deleted', 'key': provider_key})


@router.get('/api/providers/{provider_key}/models')
async def get_models(provider_key: str):
    try:
        provider = registry.get(provider_key)
        models = await provider.list_models()
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"Provider '{provider_key}' not found"})
    except Exception as e:
        logger.error("Error fetching models for provider '%s': %s", provider_key, e)
        models = []
    return JSONResponse(content={"models": [{'id': m.id, 'name': m.id} for m in models]})


@router.get('/api/providers/{provider_key}/inference-params')
async def get_inference_params(provider_key: str):
    try:
        provider = registry.get(provider_key)
        return provider.get_inference_params()
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"Provider '{provider_key}' not found"})
