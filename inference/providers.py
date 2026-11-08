from typing import Any

from psycopg.types.json import Jsonb

import database
from domain import LLAMA_CPP_PROVIDER_KEY, InferenceProviderConfig, InferenceProviderUpdate
from inference.openai_chat import OpenAICompatibleProvider
from inference.registry import registry

_FIELDS = (
    "key, display_name, description, base_url, api_key, models, "
    "timeout_seconds, headers, extra_body, include_reasoning_content"
)


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
                Jsonb(config.models),
                config.timeout_seconds,
                Jsonb(config.headers),
                Jsonb(config.extra_body),
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
            value = Jsonb(value)
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
