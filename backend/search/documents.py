from typing import Optional, Tuple, List, Any
from itertools import batched

from backend.database import AsyncConnection


async def get_document_by_path(conn: AsyncConnection, path: str) -> Optional[Tuple[int, str]]:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, file_hash FROM documents WHERE file_path = %s",
            (path,),
        )
        row = await cur.fetchone()
        return row if row else None


async def delete_documents_not_in(conn: AsyncConnection, repository_id: int, keep_paths: List[str]) -> int:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, file_path FROM documents WHERE repository_id = %s",
            (repository_id,),
        )
        rows = await cur.fetchall()

    keep = set(keep_paths)
    stale_ids = [row[0] for row in rows if row[1] not in keep]
    deleted = 0
    async with conn.cursor() as cur:
        for batch in batched(stale_ids, 500):
            placeholders = ", ".join("?" for _ in batch)
            await cur.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})",
                batch,
            )
            deleted += cur.rowcount
    return deleted


async def batch_upsert_documents(
    conn: AsyncConnection,
    docs: list[tuple[int, str, str, str]],  # (repository_id, file_path, file_hash, content)
) -> list[int]:
    doc_ids: list[int] = []
    async with conn.cursor() as cur:
        for repository_id, file_path, file_hash, content in docs:
            await cur.execute(
                """
                INSERT INTO documents (repository_id, file_path, file_hash, content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE
                SET repository_id = excluded.repository_id,
                    file_hash = excluded.file_hash,
                    content = excluded.content,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (repository_id, file_path, file_hash, content),
            )
            row = await cur.fetchone()
            assert row is not None
            doc_ids.append(row[0])
    return doc_ids


async def batch_replace_document_chunks(
    conn: AsyncConnection,
    document_ids: list[int],
    chunks: list[tuple[int, int, str, Any]],  # (doc_id, chunk_index, chunk_text, embedding)
) -> None:
    async with conn.cursor() as cur:
        for batch in batched(document_ids, 500):
            placeholders = ", ".join("?" for _ in batch)
            await cur.execute(
                f"DELETE FROM document_chunks WHERE document_id IN ({placeholders})",
                batch,
            )
        if chunks:
            await cur.executemany(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) "
                "VALUES (%s, %s, %s, %s)",
                chunks,
            )
