import re
from pathlib import Path
from typing import List, Tuple, Optional
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_url
import httpx


# Llama.cpp model cache and Huggingface download algorithm is based on:
# https://github.com/ggml-org/llama.cpp/blob/4eac5b45095a4e8a1ff1cce4f6d030e0872fb4ad/common/download.cpp
# https://github.com/ggml-org/llama.cpp/blob/master/common/download.cpp

DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

def get_gguf_split_info(filename: str) -> dict:
    re_split = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)
    re_tag = re.compile(r"\.([A-Z0-9_]+)\.gguf$", re.IGNORECASE)

    info = {"prefix": filename, "index": 1, "count": 1, "tag": ""}
    basename = filename

    m_split = re_split.search(filename)
    if m_split:
        basename = m_split.group(1)
        info["index"] = int(m_split.group(2))
        info["count"] = int(m_split.group(3))

    m_tag = re_tag.search(basename)
    if m_tag:
        info["tag"] = m_tag.group(1)
        info["prefix"] = basename[:m_tag.start()]

    return info

def list_cached_models(cache_dir: Path = DEFAULT_CACHE) -> List[str]:
    if not cache_dir.exists():
        return []

    cached = set()
    for repo_dir in cache_dir.glob("models--*--*"):
        if not repo_dir.is_dir(): continue

        repo_id = repo_dir.name.replace("models--", "").replace("--", "/")

        for f in repo_dir.rglob("*.gguf"):
            info = get_gguf_split_info(f.name)
            if info["index"] == 1 and "mmproj" not in info["prefix"].lower():
                tag = info["tag"] or f.name
                cached.add(f"{repo_id}:{tag}")

    return sorted(list(cached))

def resolve_hf_alias(alias: str, api: Optional[HfApi] = None) -> Tuple[str, str]:
    if ":" in alias:
        repo_id, tag = alias.rsplit(":", 1)
    else:
        repo_id, tag = alias, "Q4_K_M" # Default tag

    api = api or HfApi()
    files = api.list_repo_files(repo_id=repo_id)

    pattern = re.compile(rf"{tag}[.-]", re.IGNORECASE)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    for f in gguf_files:
        if pattern.search(f):
            info = get_gguf_split_info(f)
            if info["index"] == 1:
                return repo_id, f

    if gguf_files:
        return repo_id, gguf_files[0]

    raise FileNotFoundError(f"Couldn't find matching repo file for {alias} in {repo_id}")

async def download_file_slice(repo_id: str, filename: str, start: int, bytes_to_read: int) -> bytes:
    print(f"Downloading bytes {start}-{bytes_to_read-1} of {repo_id}/{filename} ")
    url = hf_hub_url(repo_id, filename)
    headers = {"Range": f"bytes={start}-{bytes_to_read - 1}"}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.content
