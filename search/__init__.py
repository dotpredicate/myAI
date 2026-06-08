import asyncio
import hashlib
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import NamedTuple, List, Optional

import database
from . import documents
from repositories import get_repositories, get_repo_documents, RepositoryConfig
from inference.llama_cpp_server import LlamaCppEmbeddingServer
from log_config import get_logger

_sync_lock = asyncio.Lock()

EMBEDDING_MODEL = "unsloth/embeddinggemma-300m-GGUF"
embedding_server = LlamaCppEmbeddingServer(EMBEDDING_MODEL)
logger = get_logger(__name__)
# logger.setLevel("DEBUG")

FILE_BATCH_SIZE = 100
EMBED_BATCH_SIZE = 100

INDEXED_EXTENSIONS = {
    ".properties", ".md", ".txt", ".json", ".xml", ".csv", ".yml", ".yaml",
    ".mill", ".java", ".c", ".cpp", ".html", ".hbs", ".js", ".ts",
    ".py", ".sh", ".sql",
}


@dataclass
class IndexedFile:
    relative_path: str
    file_hash: str
    file_text: str
    token_ids: list[list[int]] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)


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
            chunk_tokens.append(int(piece["id"]))
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


def _batched(iterable, n: int):
    """Yield successive n-sized chunks from an iterable."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


async def _index_file(repo: RepositoryConfig, relative_path: str, full_path: Path) -> Optional[IndexedFile]:
    """Read, hash and tokenize a single file. Returns None if unchanged or unreadable."""
    if Path(relative_path).suffix.lower() not in INDEXED_EXTENSIONS:
        logger.debug("%s - skipping unhandled extension", relative_path)
        return None

    try:
        file_bytes = full_path.read_bytes()
        file_text = file_bytes.decode()
    except Exception:
        logger.exception("%s - unreadable file", relative_path)
        return None

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    with database.mk_conn() as conn:
        existing = documents.get_document_by_path(conn, relative_path)
    if existing and existing[1] == file_hash:
        logger.debug("%s - unchanged checksum %s", relative_path, file_hash)
        return None

    logger.info("%s - to be indexed", relative_path)
    token_ids, chunk_texts = await get_token_chunks(file_text)
    logger.debug("%s - tokenized into %s chunks", relative_path, len(chunk_texts))
    return IndexedFile(
        relative_path=relative_path,
        file_hash=file_hash,
        file_text=file_text,
        token_ids=token_ids,
        chunk_texts=chunk_texts,
    )


async def _process_file_batch(repo: RepositoryConfig, files: list[tuple[str, Path]]) -> None:
    """Tokenize a batch of files in parallel, batch-embed all chunks, then bulk-upsert to DB."""
    # Step 1: parallel tokenization
    tasks = [_index_file(repo, rel, full) for rel, full in files]
    indexed_files: list[IndexedFile] = [r for r in await asyncio.gather(*tasks) if r is not None]

    if not indexed_files:
        logger.debug("No changed files in this batch")
        return

    repo_name = repo.internal_name
    total_plans = sum(len(f.chunk_texts) for f in indexed_files)
    logger.info("%s - %s changed files, %s chunks total", repo_name, len(indexed_files), total_plans)

    # Step 2: gather all token_ids and chunk_texts, keeping track of boundaries for splitting back
    all_token_ids: list[list[int]] = []
    all_chunk_texts: list[str] = []
    file_chunk_counts: list[int] = []
    for f in indexed_files:
        file_chunk_counts.append(len(f.token_ids))
        all_token_ids.extend(f.token_ids)
        all_chunk_texts.extend(f.chunk_texts)

    # Step 3: embed in sub-batches of EMBED_BATCH_SIZE
    all_embeddings: list[list[float]] = []
    for sub_batch in _batched(all_token_ids, EMBED_BATCH_SIZE):
        sub_embeddings = await embedding_server.embed(EMBEDDING_MODEL, sub_batch)
        all_embeddings.extend(sub_embeddings)

    assert len(all_embeddings) == len(all_chunk_texts)

    # Step 4: batch upsert documents, split embeddings back per file, bulk-insert chunks
    with database.mk_conn() as conn:
        doc_tuples = [(f.relative_path, f.file_hash, f.file_text) for f in indexed_files]
        doc_ids = documents.batch_upsert_documents(conn, doc_tuples)

        embed_offset = 0
        all_db_chunks: list[tuple[int, int, str, list[float]]] = []
        for idx, f in enumerate(indexed_files):
            doc_id = doc_ids[idx]
            n_chunks = file_chunk_counts[idx]
            for ci in range(n_chunks):
                all_db_chunks.append((
                    doc_id,
                    ci,
                    f.chunk_texts[ci],
                    all_embeddings[embed_offset + ci],
                ))
            embed_offset += n_chunks

        documents.batch_replace_document_chunks(conn, all_db_chunks)
        conn.commit()

    logger.info("%s - %s files indexed (%s chunks, %s embeddings)", repo_name, len(indexed_files), total_plans, len(all_embeddings))


async def synchronize():
    """Incrementally sync repositories using parallel batch processing."""
    if _sync_lock.locked():
        logger.info("Indexing is already running, skipping")
        return
    async with _sync_lock:
        logger.info("Indexing started")
        for repo in get_repositories():
            repo_docs = get_repo_documents(repo.internal_name)
            total_files = len(repo_docs)
            logger.info("Processing repository %s (%s files)", repo.internal_name, total_files)

            for batch_idx, file_batch in enumerate(_batched(repo_docs, FILE_BATCH_SIZE)):
                logger.info("Repository %s - batch %s (%s files)", repo.internal_name, batch_idx, len(file_batch))
                try:
                    await _process_file_batch(repo, file_batch)
                except Exception:
                    logger.exception("Repository %s - batch %s failed", repo.internal_name, batch_idx)

        logger.info("Indexing finished")