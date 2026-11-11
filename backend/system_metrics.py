import os
import threading
from dataclasses import dataclass
from pathlib import Path

from backend.gpu_metrics import read_gpu_data
from backend.log_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CpuStats:
    used_cores: float | None
    max_cores: int


@dataclass(frozen=True)
class MemoryStats:
    used_bytes: int | None
    total_bytes: int | None


@dataclass(frozen=True)
class GpuStats:
    vram_used_bytes: int | None
    vram_total_bytes: int | None
    utilization_percent: float | None


@dataclass(frozen=True)
class SystemStats:
    cpu: CpuStats
    ram: MemoryStats
    gpu: GpuStats


_previous_cpu: tuple[int, int] | None = None
_cpu_state_lock = threading.Lock()


def _read_cpu_times() -> tuple[int, int]:
    line = Path("/proc/stat").read_text().splitlines()[0]
    fields = [int(value) for value in line.split()[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = fields[:8]
    total = user + nice + system + idle + iowait + irq + softirq + steal
    return total - idle - iowait, total


def _read_cpu_stats() -> CpuStats:
    global _previous_cpu
    max_cores = os.cpu_count() or 1
    with _cpu_state_lock:
        try:
            busy, total = _read_cpu_times()
        except (OSError, ValueError) as error:
            logger.debug("Could not read CPU times: %s", error)
            return CpuStats(used_cores=None, max_cores=max_cores)

        used_cores: float | None = None
        if _previous_cpu is not None:
            previous_busy, previous_total = _previous_cpu
            total_delta = total - previous_total
            if total_delta > 0:
                busy_ratio = (busy - previous_busy) / total_delta
                used_cores = min(float(max_cores), max(0.0, busy_ratio * max_cores))
        _previous_cpu = (busy, total)
        return CpuStats(used_cores=used_cores, max_cores=max_cores)


def _read_memory_stats() -> MemoryStats:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, separator, rest = line.partition(":")
            parts = rest.split()
            if separator and parts:
                values[key] = int(parts[0]) * 1024
    except (OSError, ValueError) as error:
        logger.debug("Could not read memory information: %s", error)
        return MemoryStats(used_bytes=None, total_bytes=None)

    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    if total <= 0:
        return MemoryStats(used_bytes=None, total_bytes=None)
    return MemoryStats(used_bytes=max(0, total - available), total_bytes=total)


def _unavailable_gpu_stats() -> GpuStats:
    return GpuStats(vram_used_bytes=None, vram_total_bytes=None, utilization_percent=None)


def _read_gpu_stats() -> GpuStats:
    data = read_gpu_data()
    card = data.get("card0") if data is not None else None
    if not isinstance(card, dict):
        return _unavailable_gpu_stats()

    try:
        used = int(card.get("VRAM Total Used Memory (B)", 0))
        total = int(card.get("VRAM Total Memory (B)", 0))
    except (TypeError, ValueError):
        used = total = 0
    try:
        utilization: float | None = float(card["GPU use (%)"])
    except (KeyError, TypeError, ValueError):
        utilization = None

    return GpuStats(
        vram_used_bytes=used if total > 0 else None,
        vram_total_bytes=total if total > 0 else None,
        utilization_percent=utilization,
    )


def sample_system_stats() -> SystemStats:
    return SystemStats(cpu=_read_cpu_stats(), ram=_read_memory_stats(), gpu=_read_gpu_stats())
