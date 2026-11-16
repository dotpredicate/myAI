import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, NamedTuple, Optional, TypedDict, cast

import httpx
from huggingface_hub import HfApi, RepoFile, hf_hub_url
from backend.database import mk_conn
from backend.gpu_metrics import read_process_vram_bytes
from backend.log_config import get_logger
from .gguf_cache import (
    find_cached_gguf,
    get_gguf_split_info,
    list_cached_quants,
    list_gguf_quants,
    resolve_cached_alias,
    resolve_hf_alias,
)
from .gguf_header import (
    GgufHeader,
    GgufParseError,
    TruncatedGgufError,
    parse_gguf_header,
    read_gguf_header,
)

logger = get_logger(__name__)

_BENCHMARK_LOCK = asyncio.Lock()


class RemoteFileInfo(NamedTuple):
    size: int
    blob_id: str


class BenchmarkPoint(TypedDict, total=False):
    depth: int
    tg_tps: float
    pp_tps: float
    vram_peak_bytes: int
    duration_ms: int


class _BenchEntry(NamedTuple):
    depth: int
    kind: str
    tps: float


class BenchmarkResult(TypedDict, total=False):
    quant: str
    points: list[BenchmarkPoint]
    size_bytes: int
    parameters: int
    architecture: str
    context_length: int
    error: str
    done: bool


async def benchmark_model_stream(
    model_alias: str,
    n_gen: int = 128,
    n_prompt: int = 512,
    repetitions: int = 1,
    quants: Optional[list[str]] = None,
) -> AsyncIterator[BenchmarkResult]:
    """Yield one benchmark result per quantization as soon as it finishes.

    Only the GGUF header is streamed, a synthetic model with zero (sparse)
    weights is built, then llama-bench measures generation and prefill speed
    at several context depths. When ``quants`` is given, only those tags are
    benchmarked, in the order the caller listed them.
    """
    try:
        repo_id, targets = _resolve_targets(model_alias)
    except Exception as e:
        yield {"error": f"Failed to resolve model: {str(e)}"}
        return

    if quants:
        wanted = set(quants)
        order = {quant: index for index, quant in enumerate(quants)}
        targets = sorted(
            (target for target in targets if target[1] in wanted),
            key=lambda target: order[target[1]],
        )
        if not targets:
            yield {"error": "No matching quants selected"}
            return

    if _BENCHMARK_LOCK.locked():
        yield {"error": "A benchmark is already running"}
        return

    async with _BENCHMARK_LOCK:
        infos = await _batch_file_infos(
            repo_id, [filename for filename, _ in targets]
        )
        for filename, quant in targets:
            logger.info("Benchmarking %s (%s)", quant, filename)
            async for result in _benchmark_one_stream(
                repo_id,
                filename,
                quant,
                n_gen,
                n_prompt,
                repetitions,
                info=infos.get(filename),
            ):
                points = result.get("points") or []
                if points:
                    try:
                        await save_benchmark_points(repo_id, quant, points)
                    except Exception:
                        logger.exception(
                            "Could not cache benchmark result for %s (%s)",
                            quant,
                            filename,
                        )
                yield result
            logger.info("Finished benchmarking %s (%s)", quant, filename)


class ModelQuantInfo(TypedDict, total=False):
    quant: str
    filename: str
    size_bytes: int
    bits_per_weight: Optional[float]
    parameters: int
    architecture: str
    context_length: int
    points: list[BenchmarkPoint]
    error: str


async def model_info(model_alias: str) -> dict:
    """Return lightweight metadata for a HF GGUF model and all its quants.

    Only the small GGUF header is streamed for each quant, so this does not
    download the full model or run a benchmark.
    """
    try:
        repo_id, targets = _resolve_targets(model_alias)
    except Exception as e:
        return {"error": f"Failed to resolve model: {str(e)}"}

    infos = await _batch_file_infos(
        repo_id, [filename for filename, _ in targets]
    )
    quants: list[ModelQuantInfo] = list(await asyncio.gather(
        *(
            _quant_info(repo_id, filename, quant, infos.get(filename))
            for filename, quant in targets
        )
    ))

    cached = await load_all_benchmark_points(repo_id)
    for quant in quants:
        name = quant.get("quant")
        if name:
            quant["points"] = cached.get(name, [])

    return _model_info_result(model_alias, quants)


async def _quant_info(
    repo_id: str,
    filename: str,
    quant: str,
    info: Optional[RemoteFileInfo] = None,
) -> ModelQuantInfo:
    try:
        header, _ = await _load_header(
            repo_id, filename, with_bytes=False, info=info
        )
    except (OSError, GgufParseError, HeaderUnavailable) as e:
        return {"quant": quant, "filename": filename, "error": str(e)}
    return _quant_info_from_header(filename, quant, header)


def _quant_info_from_header(filename: str, quant: str, header: GgufHeader) -> ModelQuantInfo:
    bits_per_weight = header.bits_per_weight
    if bits_per_weight is None and header.parameters:
        # Unknown tensor types: fall back to the file size estimate.
        bits_per_weight = header.size * 8 / header.parameters
    return {
        "quant": quant,
        "filename": filename,
        "size_bytes": header.size,
        "bits_per_weight": bits_per_weight,
        "parameters": header.parameters,
        "architecture": header.architecture,
        "context_length": header.context_length,
    }


def _model_info_result(model_alias: str, quants: list[ModelQuantInfo]) -> dict[str, Any]:
    # Highest quality first: bits per weight, then the raw file size as a
    # tie-breaker. Entries without a header (errors) sort last.
    def sort_key(quant: ModelQuantInfo) -> tuple[float, int]:
        bits = quant.get("bits_per_weight")
        return (bits if bits is not None else -1.0, quant.get("size_bytes", 0))

    ordered = sorted(quants, key=sort_key, reverse=True)
    return {"model_id": model_alias, "quants": ordered}


async def save_benchmark_points(
    model_id: str, quant: str, points: list[BenchmarkPoint]
) -> None:
    rows = []
    for point in points:
        depth = point.get("depth")
        if depth is None:
            continue
        rows.append((
            model_id,
            quant,
            depth,
            point.get("tg_tps"),
            point.get("pp_tps"),
            point.get("vram_peak_bytes"),
            point.get("duration_ms"),
        ))
    if not rows:
        return

    async with mk_conn() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO benchmark_results
                (model_id, quant, depth, tg_tps, pp_tps, vram_peak_bytes, duration_ms, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(model_id, quant, depth) DO UPDATE SET
                tg_tps = excluded.tg_tps,
                pp_tps = excluded.pp_tps,
                vram_peak_bytes = excluded.vram_peak_bytes,
                duration_ms = excluded.duration_ms,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )


async def delete_benchmark_points(model_id: str) -> None:
    async with mk_conn() as conn, conn.cursor() as cur:
        if ":" in model_id:
            repo_id, quant = model_id.rsplit(":", 1)
            await cur.execute(
                """
                DELETE FROM benchmark_results
                WHERE model_id = %s AND upper(quant) = upper(%s)
                """,
                (repo_id, quant),
            )
        else:
            await cur.execute(
                "DELETE FROM benchmark_results WHERE model_id = %s", (model_id,)
            )


async def load_all_benchmark_points(
    model_id: str,
) -> dict[str, list[BenchmarkPoint]]:
    try:
        async with mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT quant, depth, tg_tps, pp_tps, vram_peak_bytes, duration_ms
                FROM benchmark_results
                WHERE model_id = %s
                ORDER BY quant, depth
                """,
                (model_id,),
            )
            rows = await cur.fetchall()
    except Exception:
        logger.exception("Could not load cached benchmark results for %s", model_id)
        return {}

    by_quant: dict[str, list[BenchmarkPoint]] = {}
    for quant, depth, tg_tps, pp_tps, vram_peak_bytes, duration_ms in rows:
        point: BenchmarkPoint = {"depth": int(depth)}
        if tg_tps is not None:
            point["tg_tps"] = float(tg_tps)
        if pp_tps is not None:
            point["pp_tps"] = float(pp_tps)
        if vram_peak_bytes is not None:
            point["vram_peak_bytes"] = int(vram_peak_bytes)
        if duration_ms is not None:
            point["duration_ms"] = int(duration_ms)
        by_quant.setdefault(str(quant), []).append(point)
    return by_quant


def _resolve_targets(alias: str) -> tuple[str, list[tuple[str, str]]]:
    if ":" in alias:
        cached = resolve_cached_alias(alias)
        if cached is not None:
            repo_id, filename = cached
            return repo_id, [(filename, get_gguf_split_info(filename).tag or filename)]
        repo_id, filename = resolve_hf_alias(alias)
        return repo_id, [(filename, get_gguf_split_info(filename).tag or filename)]

    try:
        quants = list_gguf_quants(alias)
    except Exception as e:
        logger.warning("Could not list remote quants for %s: %s", alias, e)
        quants = []
    if quants:
        return alias, quants

    cached_quants = list_cached_quants(alias)
    if cached_quants:
        return alias, cached_quants

    raise ValueError(f"No GGUF quants found for {alias}")


def _depth_points(context_length: int) -> list[int]:
    """0, 1K, 4K, 16K, ... plus the model's full context length."""
    if context_length <= 0:
        return [0]
    points = [0]
    step = 1024
    while step < context_length:
        points.append(step)
        step *= 4
    if context_length not in points:
        points.append(context_length)
    return points


def _merge_depth_points(
    points: list[BenchmarkPoint], depths: list[int]
) -> list[BenchmarkPoint]:
    by_depth = {int(point.get("depth", 0)): point for point in points}
    return [by_depth.get(depth, {"depth": depth}) for depth in depths]


async def _benchmark_one_stream(
    repo_id: str,
    filename: str,
    quant: str,
    n_gen: int,
    n_prompt: int,
    repetitions: int,
    *,
    info: Optional[RemoteFileInfo] = None,
) -> AsyncIterator[BenchmarkResult]:
    if get_gguf_split_info(filename).shards > 1:
        logger.warning("Skipping split GGUF benchmark for %s (%s)", quant, filename)
        yield {"quant": quant, "error": "Benchmarking split GGUF models is not supported", "done": True}
        return

    try:
        header, header_bytes = await _load_header(repo_id, filename, info=info)
    except (OSError, GgufParseError, HeaderUnavailable) as e:
        logger.warning("Could not read GGUF header for %s (%s): %s", quant, filename, e)
        yield {"quant": quant, "error": str(e), "done": True}
        return

    depth_points = _depth_points(header.context_length)
    path: Optional[Path] = None
    points_by_depth: dict[int, BenchmarkPoint] = {}
    depth_spent_ms: dict[int, float] = {}
    final: Optional[_BenchProcessResult] = None
    last_event_at = time.monotonic()

    def collected_points() -> list[BenchmarkPoint]:
        return [points_by_depth[depth] for depth in sorted(points_by_depth)]

    try:
        path = _build_synthetic(header, header_bytes)
        cmd = _bench_cmd(str(path), n_gen, n_prompt, depth_points, repetitions)
        async for event in _run_llama_bench_stream(cmd):
            if "entry" in event:
                parsed = _point_from_entry(event["entry"])
                if parsed is None:
                    continue
                now = time.monotonic()
                depth_spent_ms[parsed.depth] = (
                    depth_spent_ms.get(parsed.depth, 0.0) + (now - last_event_at) * 1000.0
                )
                last_event_at = now
                point = points_by_depth.setdefault(parsed.depth, {"depth": parsed.depth})
                if parsed.kind == "pp":
                    point["pp_tps"] = parsed.tps
                else:
                    point["tg_tps"] = parsed.tps
                point["duration_ms"] = round(depth_spent_ms[parsed.depth])
                vram = event.get("vram")
                if vram is not None:
                    point["vram_peak_bytes"] = vram
                yield _benchmark_result(
                    quant,
                    header,
                    {
                        "points": [dict(item) for item in collected_points()],
                        "done": False,
                    },
                )
            elif "process" in event:
                final = event["process"]
            elif "error" in event:
                yield _benchmark_result(
                    quant,
                    header,
                    {
                        "points": _merge_depth_points(collected_points(), depth_points),
                        "error": event["error"],
                        "done": True,
                    },
                )
                return

        if final is None:
            yield _benchmark_result(
                quant,
                header,
                {
                    "points": _merge_depth_points(collected_points(), depth_points),
                    "error": "llama-bench produced no result",
                    "done": True,
                },
            )
            return

        for depth, vram in final.vram_peaks.items():
            point = points_by_depth.setdefault(depth, {"depth": depth})
            point["vram_peak_bytes"] = vram
        bench: dict[str, Any] = {
            "points": _merge_depth_points(collected_points(), depth_points),
            "done": True,
        }
        if final.returncode != 0:
            bench["error"] = f"llama-bench error: {_extract_bench_error(final.stderr)}"
            logger.warning(
                "Benchmark %s (%s) finished with error: %s",
                quant,
                filename,
                bench["error"],
            )
        yield _benchmark_result(quant, header, bench)
    except Exception as e:
        logger.exception("Benchmark failed for %s", quant)
        yield _benchmark_result(
            quant,
            header,
            {
                "points": _merge_depth_points(collected_points(), depth_points),
                "error": f"Benchmark failed: {str(e)}",
                "done": True,
            },
        )
    finally:
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass


def _benchmark_result(
    quant: str, header: GgufHeader, bench: dict[str, Any]
) -> BenchmarkResult:
    result: BenchmarkResult = {
        "quant": quant,
        "points": bench.get("points", []),
        "size_bytes": bench.get("size_bytes") or header.size,
        "parameters": bench.get("parameters") or header.parameters,
        "architecture": header.architecture,
        "context_length": header.context_length,
    }
    if "error" in bench:
        result["error"] = bench["error"]
    if "done" in bench:
        result["done"] = bool(bench["done"])
    return result


class HeaderUnavailable(RuntimeError):
    """The GGUF header could not be obtained locally or over the network."""


REMOTE_HEADER_INITIAL_BYTES = 4 << 20
REMOTE_HEADER_MAX_BYTES = 64 << 20
_HEADER_CACHE_DIR = Path(os.path.expanduser("~/.cache/myai/headers"))


async def _remote_file_infos(
    repo_id: str, filenames: list[str]
) -> dict[str, RemoteFileInfo]:
    try:
        entries = await asyncio.to_thread(
            HfApi().get_paths_info,
            repo_id,
            filenames,
            repo_type="model",
        )
    except Exception as e:
        raise HeaderUnavailable(f"Failed to read remote file info: {e}") from e
    infos: dict[str, RemoteFileInfo] = {}
    for entry in entries:
        if isinstance(entry, RepoFile) and entry.size:
            infos[entry.path] = RemoteFileInfo(
                size=int(entry.size), blob_id=entry.blob_id
            )
    return infos


async def _batch_file_infos(
    repo_id: str, filenames: list[str]
) -> dict[str, RemoteFileInfo]:
    try:
        return await _remote_file_infos(repo_id, filenames)
    except Exception as e:
        logger.warning("Could not batch file info for %s: %s", repo_id, e)
        return {}


async def download_file_slice(repo_id: str, filename: str, start: int, length: int) -> bytes:
    """Fetch a byte range, refusing to buffer the whole file if Range is ignored."""
    end = start + length - 1
    logger.info("Downloading bytes %s-%s of %s/%s", start, end, repo_id, filename)
    url = hf_hub_url(repo_id, filename)
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
        async with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
            if response.status_code != 206:
                raise RuntimeError(
                    f"Range request for {repo_id}/{filename} returned HTTP {response.status_code}"
                )
            data = b"".join([chunk async for chunk in response.aiter_bytes()])
    if len(data) != length:
        raise RuntimeError(f"Expected {length} bytes of {repo_id}/{filename}, got {len(data)}")
    return data


async def _fetch_header(
    repo_id: str, filename: str, info: RemoteFileInfo
) -> tuple[GgufHeader, bytes]:
    """Read a GGUF header over the network, requesting only the missing tail."""
    chunk = min(REMOTE_HEADER_INITIAL_BYTES, info.size)
    data = await download_file_slice(repo_id, filename, 0, chunk)
    while True:
        try:
            header = parse_gguf_header(data, file_size=info.size)
        except TruncatedGgufError:
            if chunk >= min(REMOTE_HEADER_MAX_BYTES, info.size):
                raise HeaderUnavailable("GGUF header is too large") from None
            next_chunk = min(chunk * 2, REMOTE_HEADER_MAX_BYTES, info.size)
            data += await download_file_slice(repo_id, filename, chunk, next_chunk - chunk)
            chunk = next_chunk
            continue
        if len(data) < header.tensor_data_offset:
            data += await download_file_slice(
                repo_id, filename, len(data), header.tensor_data_offset - len(data)
            )
        return header, data[: header.tensor_data_offset]


def _header_cache_path(repo_id: str, filename: str, info: RemoteFileInfo) -> Path:
    key = "\0".join((repo_id, filename, str(info.size), info.blob_id)).encode()
    return _HEADER_CACHE_DIR / f"{hashlib.sha256(key).hexdigest()}.header"


def _read_header_cache(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Could not read GGUF header cache at %s", path, exc_info=True)
        return None


def _write_header_cache(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except OSError:
        logger.warning("Could not write GGUF header cache at %s", path, exc_info=True)


def _read_header_meta(path: Path) -> Optional[GgufHeader]:
    try:
        raw = json.loads(path.with_suffix(".json").read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Could not read GGUF metadata cache at %s", path.with_suffix(".json"), exc_info=True)
        return None
    if not isinstance(raw, dict):
        return None
    try:
        bits = raw.get("bits_per_weight")
        return GgufHeader(
            size=int(raw["size"]),
            parameters=int(raw["parameters"]),
            architecture=str(raw["architecture"]),
            context_length=int(raw["context_length"]),
            tensor_data_offset=int(raw["tensor_data_offset"]),
            bits_per_weight=float(bits) if bits is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_header_meta(path: Path, header: GgufHeader) -> None:
    meta_path = path.with_suffix(".json")
    payload = {
        "size": header.size,
        "parameters": header.parameters,
        "architecture": header.architecture,
        "context_length": header.context_length,
        "tensor_data_offset": header.tensor_data_offset,
        "bits_per_weight": header.bits_per_weight,
    }
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = meta_path.with_name(f"{meta_path.name}.tmp")
        temporary.write_text(json.dumps(payload))
        os.replace(temporary, meta_path)
    except OSError:
        logger.warning("Could not write GGUF metadata cache at %s", meta_path, exc_info=True)


def _remove_header_cache(path: Path) -> None:
    for candidate in (path, path.with_suffix(".json")):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


async def _load_header(
    repo_id: str,
    filename: str,
    *,
    with_bytes: bool = True,
    info: Optional[RemoteFileInfo] = None,
) -> tuple[GgufHeader, bytes]:
    """Return a parsed GGUF header and optionally its raw bytes.

    Headers are cached in ``~/.cache/myai/headers`` and keyed by the remote
    blob content hash. The parsed metadata lives in a small JSON sidecar, so
    metadata-only reads do not have to parse the ~10 MB header again.
    """
    cached = find_cached_gguf(repo_id, filename)
    if cached is not None:
        header = read_gguf_header(cached)
        if not with_bytes:
            return header, b""
        with cached.open("rb") as source:
            return header, source.read(header.tensor_data_offset)

    if info is None:
        infos = await _remote_file_infos(repo_id, [filename])
        info = infos.get(filename)
        if info is None:
            raise HeaderUnavailable("Could not determine remote file size")
    cache_path = _header_cache_path(repo_id, filename, info)

    meta = _read_header_meta(cache_path)
    if meta is not None:
        if not with_bytes:
            return meta, b""
        data = _read_header_cache(cache_path)
        if data is not None and len(data) == meta.tensor_data_offset:
            return meta, data
        if meta.tensor_data_offset > 0:
            data = await download_file_slice(
                repo_id, filename, 0, meta.tensor_data_offset
            )
            _write_header_cache(cache_path, data)
            return meta, data

    data = _read_header_cache(cache_path)
    if data is not None:
        parsed: Optional[GgufHeader]
        try:
            parsed = parse_gguf_header(data, file_size=info.size)
        except GgufParseError:
            parsed = None
        if parsed is not None and parsed.tensor_data_offset == len(data):
            _write_header_meta(cache_path, parsed)
            return (parsed, data) if with_bytes else (parsed, b"")
        _remove_header_cache(cache_path)

    header, data = await _fetch_header(repo_id, filename, info)
    _write_header_cache(cache_path, data)
    _write_header_meta(cache_path, header)
    return (header, data) if with_bytes else (header, b"")


_BENCH_DIR = Path(os.path.expanduser("~/.cache/myai/benchmark"))


def _build_synthetic(header: GgufHeader, header_bytes: bytes) -> Path:
    if header.tensor_data_offset <= 0 or header.size <= 0:
        raise ValueError("Could not read the GGUF header")

    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=_BENCH_DIR, prefix="synthetic-", suffix=".gguf")
    os.close(fd)
    path = Path(name)
    try:
        with open(path, "wb") as f:
            f.write(header_bytes)
            f.truncate(header.size)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _bench_cmd(
    path: str, n_gen: int, n_prompt: int, depths: list[int], repetitions: int
) -> list[str]:
    return [
        "llama-bench", "-m", path,
        "-ngl", "99",
        "-fa", "on",
        "-n", str(n_gen),
        "-p", str(n_prompt),
        "-d", ",".join(str(depth) for depth in depths),
        "-r", str(repetitions),
        "-o", "jsonl",
    ]


def _extract_bench_error(stderr: str) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if "error loading model:" in line:
            return line.split("error loading model:", 1)[1].strip()[:300]
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in ("out of memory", "failed to allocate", "alloc_buffer", "buffer size")):
            return line[:300]
    for line in reversed(lines):
        if "error:" in line:
            detail = line.split("error:", 1)[1].strip()
            if "failed to load model" not in detail:
                return detail[:300]
    return (lines[-1][:300] if lines else "unknown error")


VRAM_SAMPLE_INTERVAL_SECONDS = 0.05


@dataclass
class _PeakVram:
    value: Optional[int] = None


class _BenchProcessResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str
    vram_peaks: dict[int, int]


class _BenchEvent(TypedDict, total=False):
    entry: dict[str, Any]
    vram: int
    process: _BenchProcessResult
    error: str


async def _sample_peak_vram(pid: int, peak: _PeakVram) -> None:
    while True:
        vram = read_process_vram_bytes(pid)
        if vram is not None and (peak.value is None or vram > peak.value):
            peak.value = vram
        await asyncio.sleep(VRAM_SAMPLE_INTERVAL_SECONDS)


def _point_from_entry(entry: dict[str, Any]) -> Optional[_BenchEntry]:
    n_prompt = int(entry.get("n_prompt") or 0)
    n_gen = int(entry.get("n_gen") or 0)
    if n_prompt <= 0 and n_gen <= 0:
        return None
    depth = int(entry.get("n_depth") or 0)
    tps = float(entry.get("avg_ts") or 0.0)
    return _BenchEntry(depth, "pp" if n_prompt > 0 else "tg", tps)


async def _run_bench_process(
    cmd: list[str], queue: asyncio.Queue[_BenchEvent]
) -> _BenchProcessResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise RuntimeError(f"llama-bench failed: {e}") from e

    peak = _PeakVram()
    sampler = asyncio.create_task(_sample_peak_vram(proc.pid, peak))
    stderr_chunks: list[bytes] = []

    async def read_stderr() -> None:
        if proc.stderr is not None:
            stderr_chunks.append(await proc.stderr.read())

    stderr_task = asyncio.create_task(read_stderr())
    lines: list[str] = []
    vram_peaks: dict[int, int] = {}
    try:
        try:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace")
                lines.append(line)
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_dict = cast(dict[str, Any], entry)
                parsed = _point_from_entry(entry_dict)
                if parsed is None:
                    continue
                depth = parsed.depth
                sampled = read_process_vram_bytes(proc.pid)
                background = peak.value
                values = [value for value in (sampled, background) if value is not None]
                if values:
                    vram_peaks[depth] = max(vram_peaks.get(depth, 0), max(values))
                peak.value = None
                event: _BenchEvent = {"entry": entry_dict}
                vram = vram_peaks.get(depth)
                if vram is not None:
                    event["vram"] = vram
                queue.put_nowait(event)
            await proc.wait()
            await stderr_task
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
    finally:
        sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    return _BenchProcessResult(
        returncode=proc.returncode or 0,
        stdout="".join(lines),
        stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        vram_peaks=vram_peaks,
    )


async def _run_llama_bench_stream(
    cmd: list[str],
) -> AsyncIterator[_BenchEvent]:
    queue: asyncio.Queue[_BenchEvent] = asyncio.Queue()

    async def produce() -> None:
        try:
            result = await _run_bench_process(cmd, queue)
            queue.put_nowait({"process": result})
        except Exception as e:
            queue.put_nowait({"error": str(e)})

    task = asyncio.create_task(produce())
    try:
        while True:
            event = await queue.get()
            yield event
            if "process" in event or "error" in event:
                break
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
