import re
from pathlib import Path
from typing import List, Tuple, Optional
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_url
import httpx
from log_config import get_logger

logger = get_logger(__name__)


# Llama.cpp model cache and Huggingface download algorithm is based on:
# https://github.com/ggml-org/llama.cpp/blob/e3ba22d6cc4dec84e59a909c7f96e1689c7384a9/common/download.cpp
# https://github.com/ggml-org/llama.cpp/blob/master/common/download.cpp

DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

def get_gguf_split_info(filename: str) -> dict:
    re_split = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)
    re_tag = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)

    info = {"prefix": filename, "index": 1, "count": 1, "tag": ""}
    prefix = filename
    if not prefix.lower().endswith(".gguf"):
        return info
    prefix = prefix[:-5]

    m_split = re_split.match(prefix)
    if m_split:
        prefix = m_split.group(1)
        info["index"] = int(m_split.group(2))
        info["count"] = int(m_split.group(3))

    # Dynamic quants use tags such as ``UD-IQ2_M`` and ``UD-Q4_K_XL``.
    # Keep the ``UD-`` prefix: dropping it makes a regular Q4_K_M and its
    # dynamic UD-Q4_K_M counterpart collide during quant deduplication.
    basename = prefix.rsplit("/", 1)[-1]
    upper_basename = basename.upper()
    ud_marker = "-UD-"
    ud_start = upper_basename.rfind(ud_marker)
    if ud_start >= 0:
        info["tag"] = basename[ud_start + 1:].upper()
    else:
        m_tag = re_tag.search(prefix)
        if m_tag:
            info["tag"] = m_tag.group(1).upper()
    info["prefix"] = prefix

    return info

def list_cached_models(cache_dir: Path = DEFAULT_CACHE) -> List[str]:
    if not cache_dir.exists():
        return []

    cached = set()
    for repo_dir in cache_dir.glob("models--*--*"):
        if not repo_dir.is_dir():
            continue

        repo_id = repo_dir.name.replace("models--", "").replace("--", "/")

        for f in repo_dir.rglob("*.gguf"):
            info = get_gguf_split_info(f.name)
            if info["index"] == 1 and "mmproj" not in info["prefix"].lower() and "imatrix" not in info["prefix"].lower() and "mtp-" not in info["prefix"].lower() and "dflash" not in info["prefix"].lower():
                tag = info["tag"] or f.name
                cached.add(f"{repo_id}:{tag}")

    return sorted(list(cached))

def resolve_hf_alias(alias: str, api: Optional[HfApi] = None) -> Tuple[str, str]:
    if ":" not in alias:
        # Difference from llama.cpp behavior - we require an explicit tag
        raise ValueError(f"Alias must include a tag (e.g. repo/model:tag), got: {alias}")
    repo_id, tag = alias.rsplit(":", 1)

    api = api or HfApi()
    files = api.list_repo_files(repo_id=repo_id)

    pattern = re.compile(rf"{tag}[.-]", re.IGNORECASE)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    for f in gguf_files:
        if not pattern.search(f):
            continue
        info = get_gguf_split_info(f)
        if info["index"] == 1:
            return repo_id, f

    raise FileNotFoundError(f"Couldn't find matching repo file for {alias} in {repo_id}")

def list_gguf_quants(repo_id: str, api: Optional[HfApi] = None) -> List[Tuple[str, str]]:
    """Return the primary GGUF file for each quantization in a repo.

    Returns a list of ``(filename, quant_tag)`` pairs, one per quant, skipping
    multimodal projectors, importance matrices and split shards.
    """
    api = api or HfApi()
    files = api.list_repo_files(repo_id=repo_id)

    quants: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for f in files:
        if not f.endswith(".gguf"):
            continue
        info = get_gguf_split_info(f)
        if info["index"] != 1:
            continue
        prefix = info["prefix"].lower()
        if "mmproj" in prefix or "imatrix" in prefix or "mtp-" in prefix or "dflash" in prefix:
            continue
        tag = info["tag"] or f
        if tag not in seen:
            seen.add(tag)
            quants.append((f, tag))
    return quants

async def download_file_slice(repo_id: str, filename: str, start: int, bytes_to_read: int) -> bytes:
    logger.info("Downloading bytes %s-%s of %s/%s", start, bytes_to_read - 1, repo_id, filename)
    url = hf_hub_url(repo_id, filename)
    headers = {"Range": f"bytes={start}-{bytes_to_read - 1}"}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.content