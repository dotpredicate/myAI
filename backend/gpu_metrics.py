import os
from pathlib import Path
from typing import Optional

_DRM_ROOT = Path("/sys/class/drm")
_PROC_ROOT = Path("/proc")
_VRAM_UNITS = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3}


def _find_gpu_device() -> Optional[Path]:
    for card in sorted(_DRM_ROOT.glob("card[0-9]*")):
        device = card / "device"
        if (device / "mem_info_vram_total").is_file():
            return device
    return None


_GPU_DEVICE = _find_gpu_device()


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_vram_total_bytes() -> Optional[int]:
    if _GPU_DEVICE is None:
        return None
    return _read_int(_GPU_DEVICE / "mem_info_vram_total")


def read_vram_used_bytes() -> Optional[int]:
    if _GPU_DEVICE is None:
        return None
    return _read_int(_GPU_DEVICE / "mem_info_vram_used")


def read_gpu_utilization_percent() -> Optional[float]:
    if _GPU_DEVICE is None:
        return None
    value = _read_int(_GPU_DEVICE / "gpu_busy_percent")
    return float(value) if value is not None else None


def _fdinfo_dir(pid: int) -> Path:
    return _PROC_ROOT / str(pid) / "fdinfo"


def read_process_vram_bytes(pid: int) -> Optional[int]:
    directory = _fdinfo_dir(pid)
    try:
        entries = os.listdir(directory)
    except OSError:
        return None

    per_client: dict[tuple[str, str], int] = {}
    for entry in entries:
        try:
            text = (directory / entry).read_text()
        except OSError:
            continue
        pdev = ""
        client = ""
        vram: Optional[int] = None
        for line in text.splitlines():
            if line.startswith("drm-pdev:"):
                pdev = line.split(":", 1)[1].strip()
            elif line.startswith("drm-client-id:"):
                client = line.split(":", 1)[1].strip()
            elif line.startswith("drm-memory-vram:"):
                parts = line.split(":", 1)[1].strip().split()
                if len(parts) != 2:
                    continue
                try:
                    amount = int(parts[0])
                except ValueError:
                    continue
                multiplier = _VRAM_UNITS.get(parts[1].lower())
                if multiplier is not None:
                    vram = amount * multiplier
        if vram is None:
            continue
        key = (pdev, client) if pdev and client else ("", entry)
        per_client[key] = max(per_client.get(key, 0), vram)
    return sum(per_client.values()) if per_client else None
