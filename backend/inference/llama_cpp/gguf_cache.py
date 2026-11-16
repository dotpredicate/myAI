import re
import shutil
from pathlib import Path
from typing import NamedTuple, Optional, TypedDict

from huggingface_hub import (
    CachedRepoInfo,
    CachedRevisionInfo,
    HFCacheInfo,
    HfApi,
    scan_cache_dir,
    try_to_load_from_cache,
)
from huggingface_hub.constants import HF_HUB_CACHE

from backend.log_config import get_logger

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

    # Keep the UD- prefix, or a dynamic UD-Q4_K_M collides with a regular Q4_K_M.
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


def primary_quants(names: list[str]) -> list[tuple[str, str]]:
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


def _active_revision(repo: CachedRepoInfo) -> Optional[CachedRevisionInfo]:
    revisions = list(repo.revisions)
    if not revisions:
        return None
    main = [rev for rev in revisions if "main" in rev.refs]
    return main[0] if main else max(revisions, key=lambda rev: rev.last_modified)


def _snapshot_files(revision: CachedRevisionInfo) -> list[tuple[str, Path]]:
    return [(entry.file_name, Path(entry.file_path)) for entry in revision.files]


def _active_files(repo: CachedRepoInfo) -> list[tuple[str, Path]]:
    revision = _active_revision(repo)
    return [] if revision is None else _snapshot_files(revision)


def _scan_cache(cache_dir: Path) -> Optional[HFCacheInfo]:
    if not cache_dir.exists():
        return None
    try:
        return scan_cache_dir(cache_dir)
    except Exception:
        logger.warning("Could not scan HuggingFace cache at %s", cache_dir)
        return None


def _cached_models_info(cache_dir: Path) -> list[CachedRepoInfo]:
    info = _scan_cache(cache_dir)
    if info is None:
        return []
    return [repo for repo in info.repos if repo.repo_type == "model"]


def _cached_files(repo_id: str, cache_dir: Path = DEFAULT_CACHE) -> list[tuple[str, Path]]:
    for repo in _cached_models_info(cache_dir):
        if repo.repo_id == repo_id:
            return _active_files(repo)
    return []


def list_cached_models(cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    cached: set[str] = set()
    for repo in _cached_models_info(cache_dir):
        names = [name for name, path in _active_files(repo) if path.is_file()]
        for _, tag in primary_quants(names):
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
    names = [name for name, path in _cached_files(repo_id, cache_dir) if path.is_file()]
    return primary_quants(names)


def resolve_cached_alias(alias: str, cache_dir: Path = DEFAULT_CACHE) -> Optional[tuple[str, str]]:
    if ":" not in alias:
        return None
    repo_id, tag = alias.rsplit(":", 1)
    for filename, quant in list_cached_quants(repo_id, cache_dir):
        if quant.lower() == tag.lower():
            return repo_id, filename
    return None


def list_gguf_quants(repo_id: str, api: Optional[HfApi] = None) -> list[tuple[str, str]]:
    api = api or HfApi()
    return primary_quants(api.list_repo_files(repo_id=repo_id))


def resolve_hf_alias(alias: str, api: Optional[HfApi] = None) -> tuple[str, str]:
    if ":" not in alias:
        # Difference from llama.cpp behavior - we require an explicit tag
        raise ValueError(f"Alias must include a tag (e.g. repo/model:tag), got: {alias}")
    repo_id, tag = alias.rsplit(":", 1)
    for filename, quant in list_gguf_quants(repo_id, api=api):
        if quant.lower() == tag.lower():
            return repo_id, filename
    raise FileNotFoundError(f"Couldn't find matching repo file for {alias} in {repo_id}")


class CachedModelEntry(TypedDict):
    id: str
    repo: str
    quant: str
    size_bytes: int


def _quant_of(filename: str) -> Optional[str]:
    if not filename.lower().endswith(".gguf"):
        return None
    info = get_gguf_split_info(filename)
    if _is_excluded(info.prefix):
        return None
    return info.tag or filename


def _active_quants(repo: CachedRepoInfo) -> dict[str, int]:
    sizes: dict[str, int] = {}
    revision = _active_revision(repo)
    for entry in revision.files if revision is not None else ():
        quant = _quant_of(entry.file_name)
        if quant is not None:
            sizes[quant] = sizes.get(quant, 0) + entry.size_on_disk
    return sizes


def _repo_blobs(repo: CachedRepoInfo, skip: set[Path]) -> set[Path]:
    return {
        Path(entry.blob_path)
        for revision in repo.revisions
        for entry in revision.files
        if Path(entry.file_path) not in skip
    }


def _cached_repo(repo_id: str, cache_dir: Path) -> Optional[CachedRepoInfo]:
    wanted = repo_id.lower()
    for repo in _cached_models_info(cache_dir):
        if repo.repo_id.lower() == wanted:
            return repo
    return None


def list_cached_model_entries(cache_dir: Path = DEFAULT_CACHE) -> list[CachedModelEntry]:
    entries = [
        CachedModelEntry(
            id=f"{repo.repo_id}:{quant}",
            repo=repo.repo_id,
            quant=quant,
            size_bytes=size,
        )
        for repo in _cached_models_info(cache_dir)
        for quant, size in _active_quants(repo).items()
    ]
    entries.sort(key=lambda entry: entry["id"])
    return entries


def delete_cached_model(model_id: str, cache_dir: Path = DEFAULT_CACHE) -> int:
    """Delete one cached quant and return the freed bytes.

    A shared blob and the rest of the repository stay. Raises ``FileNotFoundError``
    when ``model_id`` is not cached, ``ValueError`` when it lacks a ``:TAG``.
    """
    repo_id, separator, tag = model_id.rpartition(":")
    if not separator or not repo_id or not tag:
        raise ValueError(
            f"Expected a cached model alias like org/model:Q4_K_M, got: {model_id}"
        )
    repo = _cached_repo(repo_id, cache_dir)
    if repo is None:
        raise FileNotFoundError(f"{repo_id} is not in the cache")

    revision = _active_revision(repo)
    files = list(revision.files) if revision is not None else []
    victims = [
        entry
        for entry in files
        if (_quant_of(entry.file_name) or "").lower() == tag.lower()
    ]
    if not victims:
        raise FileNotFoundError(f"{model_id} is not in the cache")

    # Drop the snapshot links first, so none ever points at a missing blob.
    doomed = {Path(entry.blob_path) for entry in victims}
    for entry in victims:
        Path(entry.file_path).unlink(missing_ok=True)
    kept = _repo_blobs(repo, skip={Path(entry.file_path) for entry in victims})

    freed = 0
    for blob in doomed - kept:
        try:
            freed += blob.stat().st_size
            blob.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not delete cached blob %s", blob, exc_info=True)

    snapshots = repo.repo_path / "snapshots"
    if not any(path.is_file() for path in snapshots.rglob("*")):
        shutil.rmtree(repo.repo_path, ignore_errors=True)

    logger.info("Deleted cached model %s, freed %d bytes", model_id, freed)
    return freed
