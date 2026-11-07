import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict

import httpx
from huggingface_hub import HfApi, RepoFile, hf_hub_url
from log_config import get_logger
from .gguf import (
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

class BenchmarkResult(TypedDict, total=False):
    quant: str
    tps: float
    pp_tps: float
    size_bytes: int
    parameters: int
    architecture: str
    context_length: int
    error: str


async def benchmark_model(
    model_alias: str,
    n_gen: int = 128,
    n_prompt: int = 512,
    repetitions: int = 1,
    quants: Optional[list[str]] = None,
) -> list[BenchmarkResult]:
    """Benchmark real tok/s for a HuggingFace GGUF model without downloading it.

    Streams only the GGUF header, builds a synthetic model with zero (sparse)
    weights, then runs llama-bench to measure actual generation/prefill speed.
    When ``quants`` is given, only those quantization tags are benchmarked.
    """
    try:
        targets = _resolve_targets(model_alias)
    except Exception as e:
        return [{"error": f"Failed to resolve model: {str(e)}"}]

    if quants:
        wanted = set(quants)
        targets = [t for t in targets if t[2] in wanted]
        if not targets:
            return [{"error": "No matching quants selected"}]

    results: list[BenchmarkResult] = []
    for repo_id, filename, quant in targets:
        logger.info("Benchmarking %s (%s)", quant, filename)
        results.append(await _benchmark_one(repo_id, filename, quant, n_gen, n_prompt, repetitions))
        logger.info("Finished benchmarking %s (%s)", quant, filename)
    return results


class ModelQuantInfo(TypedDict, total=False):
    quant: str
    filename: str
    size_bytes: int
    parameters: int
    architecture: str
    context_length: int
    error: str


async def model_info(model_alias: str) -> dict:
    """Return lightweight metadata for a HF GGUF model and all its quants.

    Only the small GGUF header is streamed for each quant, so this does not
    download the full model or run a benchmark.
    """
    try:
        targets = _resolve_targets(model_alias)
    except Exception as e:
        return {"error": f"Failed to resolve model: {str(e)}"}

    quants: list[ModelQuantInfo] = list(await asyncio.gather(
        *(_quant_info(repo_id, filename, quant) for repo_id, filename, quant in targets)
    ))

    return _model_info_result(model_alias, quants)


async def _quant_info(repo_id: str, filename: str, quant: str) -> ModelQuantInfo:
    try:
        header = await _gguf_header(repo_id, filename)
    except (OSError, GgufParseError, HeaderUnavailable) as e:
        return {"quant": quant, "filename": filename, "error": str(e)}
    return _quant_info_from_header(filename, quant, header)


def _quant_info_from_header(filename: str, quant: str, header: GgufHeader) -> ModelQuantInfo:
    return {
        "quant": quant,
        "filename": filename,
        "size_bytes": header.size,
        "parameters": header.parameters,
        "architecture": header.architecture,
        "context_length": header.context_length,
    }


def _model_info_result(model_alias: str, quants: list[ModelQuantInfo]) -> dict[str, Any]:
    return {"model_id": model_alias, "quants": quants}


def _resolve_targets(alias: str) -> list[tuple[str, str, str]]:
    if ":" in alias:
        cached = resolve_cached_alias(alias)
        if cached is not None:
            repo_id, filename = cached
            return [(repo_id, filename, get_gguf_split_info(filename).tag or filename)]
        repo_id, filename = resolve_hf_alias(alias)
        return [(repo_id, filename, get_gguf_split_info(filename).tag or filename)]

    cached_quants = list_cached_quants(alias)
    if cached_quants:
        return [(alias, filename, tag) for filename, tag in cached_quants]

    quants = list_gguf_quants(alias)
    if not quants:
        raise ValueError(f"No GGUF quants found for {alias}")
    return [(alias, filename, tag) for filename, tag in quants]


async def _benchmark_one(repo_id: str, filename: str, quant: str, n_gen: int, n_prompt: int, repetitions: int) -> BenchmarkResult:
    try:
        header = await _gguf_header(repo_id, filename)
    except (OSError, GgufParseError, HeaderUnavailable) as e:
        return {"quant": quant, "error": str(e)}

    path: Optional[Path] = None
    try:
        path = await _build_synthetic(repo_id, filename, header.tensor_data_offset, header.size)
        bench = await _run_llama_bench(str(path), n_gen, n_prompt, repetitions)
    except Exception as e:
        logger.exception("Benchmark failed for %s", quant)
        return {"quant": quant, "error": f"Benchmark failed: {str(e)}"}
    finally:
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass

    if "error" in bench:
        return {"quant": quant, "error": bench["error"]}

    return {
        "quant": quant,
        "tps": bench["tps"],
        "pp_tps": bench["pp_tps"],
        "size_bytes": bench["size_bytes"],
        "parameters": bench["parameters"],
        "architecture": header.architecture,
        "context_length": header.context_length,
    }


class HeaderUnavailable(RuntimeError):
    """The GGUF header could not be obtained locally or over the network."""


REMOTE_HEADER_INITIAL_BYTES = 1 << 20
REMOTE_HEADER_MAX_BYTES = 64 << 20


async def _remote_file_size(repo_id: str, filename: str) -> int:
    try:
        entries = await asyncio.to_thread(
            HfApi().get_paths_info,
            repo_id,
            [filename],
            expand=True,
            repo_type="model",
        )
    except Exception as e:
        raise HeaderUnavailable(f"Failed to read remote file info: {e}") from e
    if not entries:
        raise HeaderUnavailable("File not found in repository")
    entry = entries[0]
    if not isinstance(entry, RepoFile) or not entry.size:
        raise HeaderUnavailable("Could not determine remote file size")
    return int(entry.size)


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


async def _remote_header(repo_id: str, filename: str) -> GgufHeader:
    """Read a GGUF header over the network, fetching only as many bytes as needed."""
    size = await _remote_file_size(repo_id, filename)
    chunk = min(REMOTE_HEADER_INITIAL_BYTES, size)
    while True:
        data = await download_file_slice(repo_id, filename, 0, chunk)
        try:
            return parse_gguf_header(data, file_size=size)
        except TruncatedGgufError:
            if chunk >= min(REMOTE_HEADER_MAX_BYTES, size):
                raise HeaderUnavailable("GGUF header is too large") from None
            chunk = min(chunk * 2, REMOTE_HEADER_MAX_BYTES, size)


async def _gguf_header(repo_id: str, filename: str) -> GgufHeader:
    cached = find_cached_gguf(repo_id, filename)
    if cached is not None:
        return read_gguf_header(cached)
    return await _remote_header(repo_id, filename)


_BENCH_DIR = Path(os.path.expanduser("~/.cache/myai/benchmark"))


async def _build_synthetic(repo_id: str, filename: str, offset: int, size: int) -> Path:
    if get_gguf_split_info(filename).shards > 1:
        raise ValueError("Benchmarking split GGUF models is not supported")
    if offset <= 0 or size <= 0:
        raise ValueError("Could not read the GGUF header")

    cached = find_cached_gguf(repo_id, filename)
    if cached is not None:
        with cached.open("rb") as source:
            header = source.read(offset)
    else:
        header = await download_file_slice(repo_id, filename, 0, offset)

    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = _BENCH_DIR / "synthetic.gguf"
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(size)
    return path


def _bench_cmd(path: str, n_gen: int, n_prompt: int, repetitions: int) -> list[str]:
    return [
        "llama-bench", "-m", path,
        "-ngl", "99",
        "-fa", "on",
        "-n", str(n_gen),
        "-p", str(n_prompt),
        "-r", str(repetitions),
        "-o", "json",
    ]


def _extract_bench_error(stderr: str) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if "error loading model:" in line:
            return line.split("error loading model:", 1)[1].strip()[:300]
    for line in reversed(lines):
        if "error:" in line:
            detail = line.split("error:", 1)[1].strip()
            if "failed to load model" not in detail:
                return detail[:300]
    return (lines[-1][:300] if lines else "unknown error")


async def _run_llama_bench(path: str, n_gen: int, n_prompt: int, repetitions: int) -> dict:
    cmd = _bench_cmd(path, n_gen, n_prompt, repetitions)
    try:
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # A plain llama-bench run only reports "failed to load model".
            # Re-run verbosely so the actual reason (unsupported architecture,
            # out of memory, corrupt header, ...) reaches the user.
            verbose = await asyncio.to_thread(
                subprocess.run, [*cmd, "-v"], capture_output=True, text=True
            )
            detail = _extract_bench_error(verbose.stderr) or _extract_bench_error(proc.stderr)
            return {"error": f"llama-bench error: {detail}"}
    except Exception as e:
        return {"error": f"llama-bench failed: {str(e)}"}
    try:
        return parse_llama_bench_output(json.loads(proc.stdout))
    except ValueError:
        return {"error": "Failed to parse llama-bench output"}


def parse_llama_bench_output(data: list) -> dict:
    tps = 0.0
    pp_tps = 0.0
    size_bytes = 0
    parameters = 0
    for entry in data:
        size_bytes = entry.get("model_size", size_bytes)
        parameters = entry.get("model_n_params", parameters)
        if entry.get("n_gen", 0) > 0:
            tps = float(entry.get("avg_ts", 0.0))
        elif entry.get("n_prompt", 0) > 0:
            pp_tps = float(entry.get("avg_ts", 0.0))
    return {"tps": tps, "pp_tps": pp_tps, "size_bytes": size_bytes, "parameters": parameters}


class GpuStats(NamedTuple):
    free_bytes: int
    total_bytes: int


def get_gpu_stats() -> GpuStats:
    result = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True)
    data = json.loads(result.stdout)
    used = int(data.get("card0", {}).get("VRAM Total Used Memory (B)", 0))
    total = int(data.get("card0", {}).get("VRAM Total Memory (B)", 0))
    return GpuStats(free_bytes=total - used, total_bytes=total)
