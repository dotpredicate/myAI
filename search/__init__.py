import asyncio
import hashlib
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import NamedTuple, Optional

import database
from . import documents
from domain import normalize_scope_name
from repositories import get_repo_by_id, get_repositories, scan_repo_documents, RepositoryConfig
from inference import EmbeddingProvider, TokenPiece
from inference.llama_cpp.server import LlamaCppEmbeddingServer
from log_config import get_logger

_sync_lock = asyncio.Lock()
_sync_pending = False
_sync_task: Optional[asyncio.Task[None]] = None

EMBEDDING_MODEL = "unsloth/embeddinggemma-300m-GGUF"
embedding_provider: EmbeddingProvider = LlamaCppEmbeddingServer(EMBEDDING_MODEL)
logger = get_logger(__name__)
# logger.setLevel("DEBUG")

FILE_BATCH_SIZE = 100
EMBED_BATCH_SIZE = 100

INDEXED_EXTENSIONS = {
    ".properties", ".md", ".txt", ".json", ".xml", ".csv", ".yml", ".yaml",
    ".mill", ".java", ".c", ".cpp", ".html", ".hbs", ".js", ".ts",
    ".py", ".sh", ".sql",
}


def _is_indexable(relative_path: str) -> bool:
    return Path(relative_path).suffix.lower() in INDEXED_EXTENSIONS


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
    # Cosine distance: lower is closer, and results are ordered closest first.
    cosine_distance: float
    text: str


def _resolve_embedding_provider(provider: Optional[EmbeddingProvider]) -> EmbeddingProvider:
    return embedding_provider if provider is None else provider


async def get_token_chunks(
    text: str,
    max_tokens: int = 400,
    overlap: int = 50,
    char_chunk_size: int = 100_000,
    provider: Optional[EmbeddingProvider] = None,
) -> tuple[list[list[int]], list[str]]:
    # Split the text into character-sized chunks before tokenizing to avoid
    # sending enormous documents to the tokenizer in one request.
    active_provider = _resolve_embedding_provider(provider)
    all_tokens: list[TokenPiece] = []
    for i in range(0, len(text), char_chunk_size):
        sub_text = text[i : i + char_chunk_size]
        sub_tokens = await active_provider.tokenize(sub_text)
        all_tokens.extend(sub_tokens)

    token_ids: list[list[int]] = []
    text_chunks: list[str] = []

    for i in range(0, len(all_tokens), max_tokens - overlap):
        chunk = all_tokens[i : i + max_tokens]
        chunk_tokens: list[int] = []
        chunk_text = ''
        for piece in chunk:
            chunk_tokens.append(piece.id)
            piece_content = piece.piece
            if isinstance(piece_content, str):
                chunk_text += piece_content
            elif isinstance(piece_content, list):
                chunk_text += bytes(piece_content).decode('utf-8', errors='replace')

        token_ids.append(chunk_tokens)
        text_chunks.append(chunk_text)
    assert len(token_ids) == len(text_chunks)
    return token_ids, text_chunks


async def semantic_search(
    query: str,
    top_k: int,
    scopes: list[str],
    provider: Optional[EmbeddingProvider] = None,
) -> list[SearchHit]:
    """Search indexed chunks inside resolved scopes.

    An empty scope list returns no results. Hits carry a cosine distance;
    lower is closer, and the closest chunk comes first.
    """
    if not scopes:
        return []

    names = list(dict.fromkeys(
        name for name in (normalize_scope_name(scope) for scope in scopes) if name
    ))
    if not names:
        return []

    active_provider = _resolve_embedding_provider(provider)
    embed_results = await active_provider.embed(EMBEDDING_MODEL, query)
    query_vec = embed_results[0]

    base_query = """
        SELECT dc.document_id, d.file_path, dc.chunk_index, dc.chunk_text,
               dc.embedding <=> %s::vector AS cosine_distance
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        JOIN repositories r ON r.id = d.repository_id
        WHERE r.internal_name = ANY(%s)
        ORDER BY cosine_distance ASC LIMIT %s
    """

    async with database.mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(base_query, (query_vec, names, top_k))
        rows = await cur.fetchall()
        results = []
        for row in rows:
            doc_id, file_path, chunk_index, content, cosine_distance = row
            results.append(SearchHit(doc_id, file_path, chunk_index, cosine_distance, content))
        return results


def _batched(iterable, n: int):
    """Yield successive n-sized chunks from an iterable."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


async def _index_file(
    repo: RepositoryConfig,
    relative_path: str,
    full_path: Path,
    provider: EmbeddingProvider,
) -> Optional[IndexedFile]:
    """Read, hash and tokenize a single file. Returns None if unchanged or unreadable."""
    try:
        file_bytes = full_path.read_bytes()
        file_text = file_bytes.decode()
    except Exception:
        logger.exception("%s - unreadable file", relative_path)
        return None

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    indexed_path = f"{repo.internal_name}/{relative_path}"
    async with database.mk_conn() as conn:
        existing = await documents.get_document_by_path(conn, indexed_path)
    if existing and existing[1] == file_hash:
        logger.debug("%s - unchanged checksum %s", relative_path, file_hash)
        return None

    logger.info("%s - to be indexed", relative_path)
    token_ids, chunk_texts = await get_token_chunks(file_text, provider=provider)
    logger.debug("%s - tokenized into %s chunks", relative_path, len(chunk_texts))
    return IndexedFile(
        relative_path=indexed_path,
        file_hash=file_hash,
        file_text=file_text,
        token_ids=token_ids,
        chunk_texts=chunk_texts,
    )


async def _process_file_batch(
    repo: RepositoryConfig,
    files: list[tuple[str, Path]],
    provider: EmbeddingProvider,
) -> None:
    """Tokenize a batch of files in parallel, batch-embed all chunks, then bulk-upsert to DB."""
    # Step 1: parallel tokenization
    tasks = [_index_file(repo, rel, full, provider) for rel, full in files]
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

    # Step 3: embed in sub-batches
    all_embeddings: list[list[float]] = []
    for sub_batch in _batched(all_token_ids, EMBED_BATCH_SIZE):
        sub_embeddings = await provider.embed(EMBEDDING_MODEL, sub_batch)
        all_embeddings.extend(sub_embeddings)

    assert len(all_embeddings) == len(all_chunk_texts)

    # Step 4: batch upsert documents, split embeddings back per file, bulk-insert chunks.
    async with database.mk_conn() as conn:
        doc_tuples = [(repo.id, f.relative_path, f.file_hash, f.file_text) for f in indexed_files]
        doc_ids = await documents.batch_upsert_documents(conn, doc_tuples)

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

        await documents.batch_replace_document_chunks(conn, all_db_chunks)
        await conn.commit()

    logger.info("%s - %s files indexed (%s chunks, %s embeddings)", repo_name, len(indexed_files), total_plans, len(all_embeddings))


async def clear_index() -> None:
    """Clear indexed documents and their dependent chunks."""
    global _sync_pending
    async with _sync_lock:
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute("TRUNCATE TABLE documents RESTART IDENTITY CASCADE")
            await conn.commit()
        pending = _sync_pending
        _sync_pending = False
    if pending:
        _schedule_sync()


async def _prune_repo_documents(repo: RepositoryConfig, files: list[tuple[str, Path]]) -> None:
    """Delete indexed documents that no longer exist among the indexable files."""
    current_paths = [f"{repo.internal_name}/{relative_path}" for relative_path, _ in files]
    async with database.mk_conn() as conn:
        deleted = await documents.delete_documents_not_in(conn, repo.id, current_paths)
        await conn.commit()
    if deleted:
        logger.info("%s - pruned %s stale documents", repo.internal_name, deleted)


async def _sync_repositories(provider: EmbeddingProvider) -> None:
    for repo in await get_repositories():
        if await get_repo_by_id(repo.id) is None:
            logger.info("Repository %s deleted before indexing; skipping", repo.internal_name)
            continue

        scan = await scan_repo_documents(repo)
        if scan.status != 'ok':
            logger.warning(
                "Repository %s - cannot scan (%s): %s",
                repo.internal_name,
                scan.status,
                scan.error,
            )
            continue

        repo_docs = [(rel, full) for rel, full in scan.files if _is_indexable(rel)]
        total_files = len(repo_docs)
        logger.info("Processing repository %s (%s files)", repo.internal_name, total_files)

        for batch_idx, file_batch in enumerate(_batched(repo_docs, FILE_BATCH_SIZE)):
            logger.info("Repository %s - batch %s (%s files)", repo.internal_name, batch_idx, len(file_batch))
            try:
                await _process_file_batch(repo, file_batch, provider)
            except Exception:
                logger.exception("Repository %s - batch %s failed", repo.internal_name, batch_idx)

        await _prune_repo_documents(repo, repo_docs)


def _schedule_sync() -> None:
    global _sync_task
    _sync_task = asyncio.create_task(synchronize())


async def synchronize(provider: Optional[EmbeddingProvider] = None) -> None:
    """Incrementally sync repositories using parallel batch processing.

    A request that arrives while another sync is running is remembered instead
    of being dropped; the running sync repeats once it finishes.
    """
    global _sync_pending
    active_provider = _resolve_embedding_provider(provider)
    if _sync_lock.locked():
        _sync_pending = True
        logger.info("Indexing is already running; queued a follow-up sync")
        return

    async with _sync_lock:
        while True:
            _sync_pending = False
            logger.info("Indexing started")
            try:
                await _sync_repositories(active_provider)
            except Exception:
                logger.exception("Indexing failed")
                if not _sync_pending:
                    raise
            if not _sync_pending:
                break
            logger.info("Indexing queued; running again")
        logger.info("Indexing finished")
