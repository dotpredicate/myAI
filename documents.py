from typing import Optional, Tuple, List, Any
from psycopg2.extensions import connection

def get_document_by_path(conn: connection, path: str) -> Optional[Tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_hash FROM documents WHERE file_path = %s",
            (path,),
        )
        row = cur.fetchone()
        return row if row else None


def upsert_document(conn: connection, path: str, file_hash: str, content: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE file_path = %s",
            (path,),
        )
        row = cur.fetchone()
        if row:
            doc_id = row[0]
            cur.execute(
                "UPDATE documents SET file_hash = %s, content = %s WHERE id = %s",
                (file_hash, content, doc_id),
            )
        else:
            cur.execute(
                "INSERT INTO documents (file_path, file_hash, content) VALUES (%s, %s, %s) RETURNING id",
                (path, file_hash, content),
            )
            (doc_id,) = cur.fetchone()
    return doc_id


def replace_document_chunks(conn: connection, doc_id: int, chunk_texts: List[str], chunk_embeddings: List[Any]):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        for index, (embedding, chunk_text) in enumerate(zip(chunk_embeddings, chunk_texts)):
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                (doc_id, index, chunk_text, embedding),
            )