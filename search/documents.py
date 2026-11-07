from typing import Optional, Tuple, List, Any
from psycopg import AsyncConnection

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
            "DELETE FROM documents "
            "WHERE repository_id = %s AND file_path <> ALL(%s::text[])",
            (repository_id, keep_paths),
        )
        return cur.rowcount


async def batch_upsert_documents(
    conn: AsyncConnection,
    docs: list[tuple[int, str, str, str]],  # (repository_id, file_path, file_hash, content)
) -> list[int]:
    async with conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO documents (repository_id, file_path, file_hash, content)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE
            SET repository_id = EXCLUDED.repository_id,
                file_hash = EXCLUDED.file_hash,
                content = EXCLUDED.content
            RETURNING id
            """,
            docs,
            returning=True,
        )
        async def next_id():
            return (await cur.fetchone())[0]
        return [await next_id() async for _ in cur.results()]


async def batch_replace_document_chunks(
    conn: AsyncConnection,
    chunks: list[tuple[int, int, str, Any]],  # (doc_id, chunk_index, chunk_text, embedding)
) -> None:
    async with conn.cursor() as cur:
        doc_ids = list({doc_id for doc_id, _, _, _ in chunks})
        await cur.execute(
            "DELETE FROM document_chunks WHERE document_id = ANY(%s)",
            (doc_ids,),
        )
        await cur.executemany(
            "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s::vector)",
            chunks,
        )