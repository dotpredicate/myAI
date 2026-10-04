import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict

from log_config import get_logger
from .hf_gguf import download_file_slice, get_gguf_split_info, list_gguf_quants, resolve_hf_alias

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
    header = await _gguf_header(repo_id, filename)
    return _quant_info_from_header(filename, quant, header)


def _quant_info_from_header(filename: str, quant: str, header: dict[str, Any]) -> ModelQuantInfo:
    entry: ModelQuantInfo = {"quant": quant, "filename": filename}
    if "error" in header:
        entry["error"] = header["error"]
        return entry
    entry["size_bytes"] = header["size"]
    entry["parameters"] = header["parameters"]
    entry["architecture"] = header["architecture"]
    entry["context_length"] = header["context_length"]
    return entry


def _model_info_result(model_alias: str, quants: list[ModelQuantInfo]) -> dict[str, Any]:
    return {"model_id": model_alias, "quants": quants}


def _resolve_targets(alias: str) -> list[tuple[str, str, str]]:
    if ":" in alias:
        repo_id, filename = resolve_hf_alias(alias)
        return [(repo_id, filename, get_gguf_split_info(filename)["tag"] or filename)]
    quants = list_gguf_quants(alias)
    if not quants:
        raise ValueError(f"No GGUF quants found for {alias}")
    return [(alias, filename, tag) for filename, tag in quants]


async def _benchmark_one(repo_id: str, filename: str, quant: str, n_gen: int, n_prompt: int, repetitions: int) -> BenchmarkResult:
    header = await _gguf_header(repo_id, filename)
    if "error" in header:
        return {"quant": quant, "error": header["error"]}

    try:
        path = await _build_synthetic(repo_id, filename, header["tensor_data_offset"], header["size"])
        bench = await _run_llama_bench(str(path), n_gen, n_prompt, repetitions)
        try:
            os.remove(path)
        except OSError:
            pass
    except Exception as e:
        logger.exception("Benchmark failed for %s", quant)
        return {"quant": quant, "error": f"Benchmark failed: {str(e)}"}

    if "error" in bench:
        return {"quant": quant, "error": bench["error"]}

    return {
        "quant": quant,
        "tps": bench["tps"],
        "pp_tps": bench["pp_tps"],
        "size_bytes": bench["size_bytes"],
        "parameters": bench["parameters"],
        "architecture": header["architecture"],
        "context_length": header["context_length"],
    }


async def _gguf_header(repo_id: str, filename: str) -> dict:
    cmd = ["./gguf-parser-linux-amd64", "--hf-repo", repo_id, "--hf-file", filename, "--raw", "--json", "--json-pretty=false"]
    try:
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
    except Exception as e:
        return {"error": f"Failed to read header: {str(e)}"}
    if proc.returncode != 0:
        return {"error": f"gguf-parser failed: {proc.stderr.strip()[:200]}"}
    try:
        data = json.loads(proc.stdout)
    except ValueError as e:
        return {"error": f"Failed to parse header: {str(e)}"}

    kv_map: dict[str, object] = {}
    for kv in data.get("header", {}).get("metadataKV", []):
        value = kv.get("value")
        if not isinstance(value, dict):
            kv_map[kv.get("key", "")] = value

    context_length = 0
    for key, value in kv_map.items():
        if key.endswith(".context_length") and isinstance(value, (int, str)):
            context_length = int(value)
            break

    architecture = kv_map.get("general.architecture")
    return {
        "tensor_data_offset": data.get("tensorDataStartOffset", 0),
        "size": data.get("size", 0),
        "parameters": data.get("modelParameters", 0),
        "architecture": architecture if isinstance(architecture, str) else "",
        "context_length": context_length,
    }


_BENCH_DIR = Path(os.path.expanduser("~/.cache/myai/benchmark"))


async def _build_synthetic(repo_id: str, filename: str, offset: int, size: int) -> Path:
    header = await download_file_slice(repo_id, filename, 0, offset)
    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = _BENCH_DIR / "synthetic.gguf"
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(size)
    return path


async def _run_llama_bench(path: str, n_gen: int, n_prompt: int, repetitions: int) -> dict:
    cmd = [
        "llama-bench", "-m", path,
        "-ngl", "99",
        "-fa", "on",
        "-n", str(n_gen),
        "-p", str(n_prompt),
        "-r", str(repetitions),
        "-o", "json",
    ]
    try:
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
    except Exception as e:
        return {"error": f"llama-bench failed: {str(e)}"}
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().split("error:")[-1].strip()[:200]
        return {"error": f"llama-bench error: {detail}"}
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
