from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from psycopg.types.json import Jsonb

import database


@dataclass(frozen=True)
class StoredImageModel:
    id: int
    name: str
    recipe: str
    files: dict[str, str]
    created_at: datetime


def _row_to_model(row: tuple) -> StoredImageModel:
    return StoredImageModel(
        id=row[0],
        name=row[1],
        recipe=row[2],
        files=row[3],
        created_at=row[4],
    )


async def list_image_models() -> list[StoredImageModel]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, name, recipe, files, created_at FROM sd_cpp_image_models ORDER BY name"
        )
        rows = await cur.fetchall()
    return [_row_to_model(row) for row in rows]


async def get_image_model(model_id: int) -> Optional[StoredImageModel]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, name, recipe, files, created_at FROM sd_cpp_image_models WHERE id = %s",
            (model_id,),
        )
        row = await cur.fetchone()
    return _row_to_model(row) if row else None


async def get_image_model_by_ref(ref: str) -> Optional[StoredImageModel]:
    ref = ref.strip()
    if ref.isdigit():
        model = await get_image_model(int(ref))
        if model is not None:
            return model
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, name, recipe, files, created_at FROM sd_cpp_image_models WHERE name = %s",
            (ref,),
        )
        row = await cur.fetchone()
    return _row_to_model(row) if row else None


async def get_image_model_by_name(name: str) -> Optional[StoredImageModel]:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id, name, recipe, files, created_at FROM sd_cpp_image_models WHERE name = %s",
            (name,),
        )
        row = await cur.fetchone()
    return _row_to_model(row) if row else None


async def create_image_model(name: str, recipe: str, files: dict[str, str]) -> StoredImageModel:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO sd_cpp_image_models (name, recipe, files)
            VALUES (%s, %s, %s)
            RETURNING id, name, recipe, files, created_at
            """,
            (name, recipe, Jsonb(files)),
        )
        row = await cur.fetchone()
        await conn.commit()
    if row is None:
        raise RuntimeError("failed to store image model")
    return _row_to_model(row)


async def delete_image_model(model_id: int) -> bool:
    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute("DELETE FROM sd_cpp_image_models WHERE id = %s", (model_id,))
        deleted = cur.rowcount > 0
        await conn.commit()
    return deleted
