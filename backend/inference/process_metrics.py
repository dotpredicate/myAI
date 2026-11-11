import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.gpu_metrics import read_gpu_data
from backend.log_config import get_logger

logger = get_logger(__name__)


CLOCK_TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")


@dataclass(frozen=True)
class ProcessUsage:
    cpu_cores: float | None = None
    rss_bytes: int | None = None
    vram_bytes: int | None = None


_cpu_samples: dict[int, tuple[float, float]] = {}
_sample_lock = threading.Lock()


def _read_cpu_seconds(pid: int) -> float | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return None
    try:
        tail = data[data.rindex(b")") + 1 :].split()
        utime = int(tail[11])
        stime = int(tail[12])
    except (ValueError, IndexError):
        return None
    return (utime + stime) / CLOCK_TICKS_PER_SECOND


def _read_rss_bytes(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except ValueError:
                    return None
    return None


def _parse_vram_by_pid(data: dict[str, Any]) -> dict[int, int]:
    processes = data.get("system")
    if not isinstance(processes, dict):
        return {}
    usage: dict[int, int] = {}
    for key, value in processes.items():
        if not key.startswith("PID"):
            continue
        try:
            pid = int(key[3:])
            usage[pid] = int(str(value).split(",")[-3])
        except (ValueError, IndexError):
            continue
    return usage


def _read_vram_by_pid() -> dict[int, int]:
    data = read_gpu_data()
    return _parse_vram_by_pid(data) if data is not None else {}


def sample_usage(pids: list[int]) -> dict[int, ProcessUsage]:
    global _cpu_samples
    with _sample_lock:
        now = time.monotonic()
        vram = _read_vram_by_pid() if pids else {}

        samples: dict[int, tuple[float, float]] = {}
        usage: dict[int, ProcessUsage] = {}
        for pid in pids:
            cpu_seconds = _read_cpu_seconds(pid)
            previous = _cpu_samples.get(pid)
            cpu_cores: float | None = None
            if cpu_seconds is not None:
                samples[pid] = (cpu_seconds, now)
                if previous is not None:
                    elapsed = now - previous[1]
                    if elapsed > 0:
                        cpu_cores = max(0.0, (cpu_seconds - previous[0]) / elapsed)
            usage[pid] = ProcessUsage(
                cpu_cores=cpu_cores,
                rss_bytes=_read_rss_bytes(pid),
                vram_bytes=vram.get(pid),
            )

        _cpu_samples = samples
        return usage
