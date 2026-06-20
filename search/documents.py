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


async def upsert_document(conn: AsyncConnection, path: str, file_hash: str, content: str) -> int:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id FROM documents WHERE file_path = %s",
            (path,),
        )
        row = await cur.fetchone()
        if row:
            doc_id = row[0]
            await cur.execute(
                "UPDATE documents SET file_hash = %s, content = %s WHERE id = %s",
                (file_hash, content, doc_id),
            )
        else:
            await cur.execute(
                "INSERT INTO documents (file_path, file_hash, content) VALUES (%s, %s, %s) RETURNING id",
                (path, file_hash, content),
            )
            doc_row = await cur.fetchone()
            assert doc_row is not None
            doc_id = doc_row[0]
    return doc_id


async def replace_document_chunks(conn: AsyncConnection, doc_id: int, chunk_texts: List[str], chunk_embeddings: List[Any]):
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        for index, (embedding, chunk_text) in enumerate(zip(chunk_embeddings, chunk_texts)):
            await cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s::vector)",
                (doc_id, index, chunk_text, embedding),
            )


async def batch_upsert_documents(
    conn: AsyncConnection,
    docs: list[tuple[str, str, str]],  # (file_path, file_hash, content)
) -> list[int]:
    async with conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO documents (file_path, file_hash, content)
            VALUES (%s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE
            SET file_hash = EXCLUDED.file_hash, content = EXCLUDED.content
            RETURNING id
            """,
            docs,
            returning=True,
        )
        return [row[0] for row in await cur.fetchall()]


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