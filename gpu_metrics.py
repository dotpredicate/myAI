import json
import subprocess
from typing import Any

from log_config import get_logger

logger = get_logger(__name__)


ROCM_SMI_TIMEOUT_SECONDS = 5
ROCM_SMI_ARGS = ["rocm-smi", "--showpids", "--showmeminfo", "vram", "--showuse", "--json"]


def read_gpu_data() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ROCM_SMI_ARGS,
            capture_output=True,
            text=True,
            timeout=ROCM_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("Could not run rocm-smi: %s", error)
        return None
    if result.returncode != 0:
        logger.debug("rocm-smi failed with exit code %s", result.returncode)
        return None
    try:
        data = json.loads(result.stdout)
    except ValueError:
        logger.debug("rocm-smi returned invalid JSON")
        return None
    return data if isinstance(data, dict) else None
