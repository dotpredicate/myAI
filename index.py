import os
import hashlib
import subprocess
import requests
import openai
from pathlib import Path
from typing import NamedTuple

import database
import documents

EMBEDDING_MODEL = 'unsloth/embeddinggemma-300m-GGUF'
LLAMA_CPP_EMBEDDINGS_URL = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:2345')
embeddings_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_EMBEDDINGS_URL + '/v1')

class SearchHit(NamedTuple):
    document_id: int
    file_path: str
    chunk_index: int
    score: float
    text: str

def get_token_chunks(text, max_tokens=400, overlap=50) -> tuple[list[list[int]], list[str]]:
    response = requests.post(
        f"{LLAMA_CPP_EMBEDDINGS_URL}/tokenize",
        json={"content": text, "with_pieces": True}
    ).json()

    all_tokens = response["tokens"]

    token_ids: list[list[int]] = []
    text_chunks: list[str] = []

    for i in range(0, len(all_tokens), max_tokens - overlap):
        chunk = all_tokens[i : i + max_tokens]
        chunk_tokens = []
        chunk_text = ''
        for piece in chunk:
            chunk_tokens.append(piece["id"])
            piece_content = piece["piece"]
            if isinstance(piece_content, str):
                chunk_text += piece_content
            elif isinstance(piece_content, list):
                chunk_text += bytes(piece_content).decode('utf-8', errors='replace')

        token_ids.append(chunk_tokens)
        text_chunks.append(chunk_text)
    assert len(token_ids) == len(text_chunks)
    return token_ids, text_chunks

def semantic_search(query: str, top_k: int) -> list[SearchHit]:
    embedding_resp = embeddings_endpoint.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vec = embedding_resp.data[0].embedding
    with database.mk_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT document_id, file_path, chunk_index, chunk_text, embedding <=> %s::vector AS score 
            FROM document_chunks
            JOIN documents ON documents.id=document_chunks.document_id
            ORDER BY score ASC LIMIT %s""",
            (query_vec, top_k)
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            doc_id, file_path, chunk_index, content, score = row
            results.append(SearchHit(doc_id, file_path, chunk_index, score, content))
        return results

def synchronize():
    """Incrementally sync repositories folder.
    For each file, compute its hash and compare it against the stored value.
    Only files that are new or changed are processed.
    """
    # Some of this should move to system module
    from system import REPOSITORIES_DIR
    for repo_dir in os.listdir(REPOSITORIES_DIR):
        repo_full_path = str(Path(REPOSITORIES_DIR) / repo_dir)
        is_git_result = subprocess.run(
            [
                "git", "-C", repo_full_path, "rev-parse", "--is-inside-work-tree"
            ],
            capture_output=True,
            text=True
        )
        is_git = is_git_result.returncode == 0

        if is_git:
            ls_git_result = subprocess.run(
                [
                    "git", "-C", repo_full_path, "ls-files" 
                ],
                capture_output=True,
                text=True
            )
            if ls_git_result.returncode != 0:
                print(f"[ERROR]: Couldn't list directories of Git repo {repo_dir} at {repo_full_path}: {ls_git_result.stderr}")
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
            if Path(fname).suffix.lower() not in {
                ".properties",
                ".md", ".txt",
                ".json", ".xml", ".csv", ".yml", ".yaml", 
                ".mill", ".java", 
                ".c", ".cpp", 
                ".html", ".hbs", ".js", ".ts",
                ".py", ".sh",
                ".sql",
            }:
                print(f"Skipping unhandled file type {fname}")
                continue

            full_path = Path(repo_full_path) / fname
            relative_path = str(full_path.relative_to(REPOSITORIES_DIR))
            try:
                file_bytes = full_path.read_bytes()
                file_text = file_bytes.decode()
            except Exception as exc:
                print(f"[WARN] Unreadable file {relative_path}: {exc}")
                continue
            
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            with database.mk_conn() as conn:
                existing = documents.get_document_by_path(conn, relative_path)
            if existing and existing[1] == file_hash:
                print(f"[SKIP] {relative_path} unchanged")
                continue

            token_ids, chunk_texts = get_token_chunks(file_text)
            embedding_resp = embeddings_endpoint.embeddings.create(
                model=EMBEDDING_MODEL,
                input=token_ids
            )
            assert len(embedding_resp.data) == len(chunk_texts)

            with database.mk_conn() as conn:
                doc_id = documents.upsert_document(conn, relative_path, file_hash, file_text)
                documents.replace_document_chunks(conn, doc_id, chunk_texts, [e.embedding for e in embedding_resp.data])
                print(f"[OK] {relative_path} - {len(embedding_resp.data)} embeddings")
