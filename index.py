import os
import hashlib
import subprocess
from pathlib import Path
from typing import NamedTuple, List, Optional

import database
import documents
from inference.llama_cpp_server import LlamaCppEmbeddingServer
from log_config import get_logger

EMBEDDING_MODEL = "unsloth/embeddinggemma-300m-GGUF"
embedding_server = LlamaCppEmbeddingServer(EMBEDDING_MODEL)
logger = get_logger(__name__)

class SearchHit(NamedTuple):
    document_id: int
    file_path: str
    chunk_index: int
    score: float
    text: str

async def get_token_chunks(text: str, max_tokens: int = 400, overlap: int = 50) -> tuple[list[list[int]], list[str]]:
    all_tokens = await embedding_server.tokenize(text)

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
    """Incrementally sync repositories folder."""
    from system import REPOSITORIES_DIR
    for repo_dir in os.listdir(REPOSITORIES_DIR):
        repo_full_path = str(Path(REPOSITORIES_DIR) / repo_dir)
        is_git_result = subprocess.run(
            ["git", "-C", repo_full_path, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True
        )
        is_git = is_git_result.returncode == 0

        if is_git:
            ls_git_result = subprocess.run(
                ["git", "-C", repo_full_path, "ls-files"],
                capture_output=True,
                text=True
            )
            if ls_git_result.returncode != 0:
                logger.error("Couldn't list directories of Git repo %s at %s: %s", repo_dir, repo_full_path, ls_git_result.stderr)
                continue
            files = ls_git_result.stdout.splitlines()
        else:
            files: list[str] = []
            for root, _, fnames in os.walk(repo_full_path):
                for fname in fnames:
                    f = Path(root) / fname
                    f = f.relative_to(repo_full_path)
                    files.append(f)

        for fname in files:
            full_path = Path(repo_full_path) / fname
            relative_path = str(full_path.relative_to(REPOSITORIES_DIR))

            if fext := Path(fname).suffix.lower() not in {
                ".properties", ".md", ".txt", ".json", ".xml", ".csv", ".yml", ".yaml", 
                ".mill", ".java", ".c", ".cpp", ".html", ".hbs", ".js", ".ts",
                ".py", ".sh", ".sql",
            }:
                logger.debug("%s - skipping unhandled extension %s", relative_path, fext)
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
                logger.info("%s - generated %s embeddings", relative_path, len(embeddings))