import asyncio
import hashlib
from pathlib import Path
from typing import NamedTuple, List, Optional

import database
import documents
from system import get_repositories, get_repo_documents, RepositoryConfig
from inference.llama_cpp_server import LlamaCppEmbeddingServer
from log_config import get_logger

_sync_lock = asyncio.Lock()

EMBEDDING_MODEL = "unsloth/embeddinggemma-300m-GGUF"
embedding_server = LlamaCppEmbeddingServer(EMBEDDING_MODEL)
logger = get_logger(__name__)
# logger.setLevel("DEBUG")

INDEXED_EXTENSIONS = {
    ".properties", ".md", ".txt", ".json", ".xml", ".csv", ".yml", ".yaml",
    ".mill", ".java", ".c", ".cpp", ".html", ".hbs", ".js", ".ts",
    ".py", ".sh", ".sql",
}

class SearchHit(NamedTuple):
    document_id: int
    file_path: str
    chunk_index: int
    score: float
    text: str

async def get_token_chunks(text: str, max_tokens: int = 400, overlap: int = 50, char_chunk_size: int = 100_000) -> tuple[list[list[int]], list[str]]:
    # Split the text into character-sized chunks before tokenizing to avoid
    # sending enormous documents to the tokenizer in one request.
    all_tokens: list[dict] = []
    for i in range(0, len(text), char_chunk_size):
        sub_text = text[i : i + char_chunk_size]
        sub_tokens = await embedding_server.tokenize(sub_text)
        all_tokens.extend(sub_tokens)

    token_ids: list[list[int]] = []
    text_chunks: list[str] = []

    for i in range(0, len(all_tokens), max_tokens - overlap):
        chunk = all_tokens[i : i + max_tokens]
        chunk_tokens: list[int] = []
        chunk_text = ''
        for piece in chunk:
            chunk_tokens.append(int(piece["id"]))  # type: ignore[call-overload]
            piece_content = piece["piece"]
            if isinstance(piece_content, str):
                chunk_text += piece_content
            elif isinstance(piece_content, list):
                chunk_text += bytes(piece_content).decode('utf-8', errors='replace')

        token_ids.append(chunk_tokens)
        text_chunks.append(chunk_text)
    assert len(token_ids) == len(text_chunks)
    return token_ids, text_chunks

async def semantic_search(query: str, top_k: int, scopes: Optional[List[str]] = None) -> list[SearchHit]:
    embed_results = await embedding_server.embed(EMBEDDING_MODEL, query)
    query_vec = embed_results[0]

    base_query = """
        SELECT document_id, file_path, chunk_index, chunk_text, embedding <=> %s::vector AS score 
        FROM document_chunks
        JOIN documents ON documents.id = document_chunks.document_id
    """

    new_params: list[object] = [query_vec]
    if scopes:
        where_clauses = []
        for scope in scopes:
            where_clauses.append("file_path LIKE %s")
            new_params.append(f"{scope}/%")

        base_query += " WHERE " + " OR ".join(where_clauses)

    base_query += " ORDER BY score ASC LIMIT %s"
    new_params.append(top_k)

    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(base_query, tuple(new_params))
        rows = cur.fetchall()
        results = []
        for row in rows:
            doc_id, file_path, chunk_index, content, score = row
            results.append(SearchHit(doc_id, file_path, chunk_index, score, content))
        return results

async def synchronize():
    """Incrementally sync repositories folder using system helpers."""
    if _sync_lock.locked():
        logger.info("Indexing is already running, skipping")
        return
    async with _sync_lock:
        logger.info("Indexing started")
        for repo in get_repositories():
            for relative_path, full_path in get_repo_documents(repo.internal_name):
                logger.debug("Indexing %s", full_path)
                if Path(relative_path).suffix.lower() not in INDEXED_EXTENSIONS:
                    logger.debug("%s - skipping unhandled extension", relative_path)
                    continue
                
                try:
                    file_bytes = full_path.read_bytes()
                    file_text = file_bytes.decode()
                except Exception as exc:
                    logger.warning("%s - unreadable file: %s", relative_path, exc)
                    continue

                file_hash = hashlib.sha256(file_bytes).hexdigest()
                with database.mk_conn() as conn:
                    existing = documents.get_document_by_path(conn, relative_path)
                if existing and existing[1] == file_hash:
                    logger.debug("%s - unchanged checksum %s", relative_path, file_hash)
                    continue

                token_ids, chunk_texts = await get_token_chunks(file_text)
                embeddings = await embedding_server.embed(EMBEDDING_MODEL, token_ids)
                assert len(embeddings) == len(chunk_texts)

                with database.mk_conn() as conn:
                    doc_id = documents.upsert_document(conn, relative_path, file_hash, file_text)
                    documents.replace_document_chunks(conn, doc_id, chunk_texts, embeddings)
                    conn.commit()
                logger.info("%s - generated %s embeddings", relative_path, len(embeddings))
    print("Indexing finished")