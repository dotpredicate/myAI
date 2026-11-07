import re
from pathlib import Path
from typing import NamedTuple, Optional

from huggingface_hub import HfApi, scan_cache_dir, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

from log_config import get_logger

logger = get_logger(__name__)

# GGUF split files follow the llama.cpp convention <name>-<index>-of-<count>.gguf.
DEFAULT_CACHE = Path(HF_HUB_CACHE)
_EXCLUDED_GGUF_MARKERS = ("mmproj", "imatrix", "mtp-", "dflash")


class SplitInfo(NamedTuple):
    prefix: str
    shard: int
    shards: int
    tag: str


def get_gguf_split_info(filename: str) -> SplitInfo:
    re_split = re.compile(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", re.IGNORECASE)
    re_tag = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)

    prefix = filename
    shard = 1
    shards = 1
    if not prefix.lower().endswith(".gguf"):
        return SplitInfo(prefix=filename, shard=shard, shards=shards, tag="")
    prefix = prefix[:-5]

    m_split = re_split.match(prefix)
    if m_split:
        prefix = m_split.group(1)
        shard = int(m_split.group(2))
        shards = int(m_split.group(3))

    # Dynamic quants use tags such as ``UD-IQ2_M`` and ``UD-Q4_K_XL``.
    # Keep the ``UD-`` prefix: dropping it makes a regular Q4_K_M and its
    # dynamic UD-Q4_K_M counterpart collide during quant deduplication.
    basename = prefix.rsplit("/", 1)[-1]
    upper_basename = basename.upper()
    ud_start = upper_basename.rfind("-UD-")
    if ud_start >= 0:
        tag = basename[ud_start + 1:].upper()
    else:
        m_tag = re_tag.search(prefix)
        tag = m_tag.group(1).upper() if m_tag else ""

    return SplitInfo(prefix=prefix, shard=shard, shards=shards, tag=tag)


def _is_excluded(prefix: str) -> bool:
    lowered = prefix.lower()
    return any(marker in lowered for marker in _EXCLUDED_GGUF_MARKERS)


def _primary_quants(names: list[str]) -> list[tuple[str, str]]:
    """Return the first file for each quant, skipping non-model GGUF files."""
    quants: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in sorted(names):
        if not name.lower().endswith(".gguf"):
            continue
        info = get_gguf_split_info(name)
        if info.shard != 1 or _is_excluded(info.prefix):
            continue
        tag = info.tag or name
        if tag not in seen:
            seen.add(tag)
            quants.append((name, tag))
    return quants


def _active_revision(repo):
    revisions = list(repo.revisions)
    if not revisions:
        return None
    main = [rev for rev in revisions if "main" in rev.refs]
    return main[0] if main else max(revisions, key=lambda rev: rev.last_modified)


def _snapshot_files(revision) -> list[tuple[str, Path]]:
    return [(entry.file_name, Path(entry.file_path)) for entry in revision.files]


def _active_files(repo) -> list[tuple[str, Path]]:
    revision = _active_revision(repo)
    return [] if revision is None else _snapshot_files(revision)


def _cached_files(repo_id: str, cache_dir: Path = DEFAULT_CACHE) -> list[tuple[str, Path]]:
    """Return (repo-relative name, path) for the active revision of a cached repo."""
    if not cache_dir.exists():
        return []
    try:
        info = scan_cache_dir(cache_dir)
    except Exception:
        logger.warning("Could not scan HuggingFace cache at %s", cache_dir)
        return []
    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            return _active_files(repo)
    return []


def list_cached_models(cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    if not cache_dir.exists():
        return []
    try:
        info = scan_cache_dir(cache_dir)
    except Exception:
        logger.warning("Could not scan HuggingFace cache at %s", cache_dir)
        return []

    cached: set[str] = set()
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        names = [name for name, path in _active_files(repo) if path.is_file()]
        for _, tag in _primary_quants(names):
            cached.add(f"{repo.repo_id}:{tag}")
    return sorted(cached)


def find_cached_gguf(repo_id: str, filename: str, cache_dir: Path = DEFAULT_CACHE) -> Optional[Path]:
    cached = try_to_load_from_cache(
        repo_id, filename, cache_dir=str(cache_dir), repo_type="model"
    )
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    # try_to_load_from_cache needs a valid ref; fall back to the snapshot index.
    for name, path in _cached_files(repo_id, cache_dir):
        if name == filename and path.is_file():
            return path
    return None


def list_cached_quants(repo_id: str, cache_dir: Path = DEFAULT_CACHE) -> list[tuple[str, str]]:
    """Return (relative filename, quant tag) for quants present in the local cache."""
    names = [name for name, path in _cached_files(repo_id, cache_dir) if path.is_file()]
    return _primary_quants(names)


def resolve_cached_alias(alias: str, cache_dir: Path = DEFAULT_CACHE) -> Optional[tuple[str, str]]:
    if ":" not in alias:
        return None
    repo_id, tag = alias.rsplit(":", 1)
    for filename, quant in list_cached_quants(repo_id, cache_dir):
        if quant.lower() == tag.lower():
            return repo_id, filename
    return None


def resolve_hf_alias(alias: str, api: Optional[HfApi] = None) -> tuple[str, str]:
    if ":" not in alias:
        # Difference from llama.cpp behavior - we require an explicit tag
        raise ValueError(f"Alias must include a tag (e.g. repo/model:tag), got: {alias}")
    repo_id, tag = alias.rsplit(":", 1)
    for filename, quant in list_gguf_quants(repo_id, api=api):
        if quant.lower() == tag.lower():
            return repo_id, filename
    raise FileNotFoundError(f"Couldn't find matching repo file for {alias} in {repo_id}")


def list_gguf_quants(repo_id: str, api: Optional[HfApi] = None) -> list[tuple[str, str]]:
    """Return the primary GGUF file for each quantization in a repo."""
    api = api or HfApi()
    return _primary_quants(api.list_repo_files(repo_id=repo_id))
