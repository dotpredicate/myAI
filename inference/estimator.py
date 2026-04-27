import subprocess
import json
import os
from typing import TypedDict, Optional
from .hf_gguf import resolve_hf_alias

class EstimateResult(TypedDict, total=False):
    vram: int
    tps: float
    error: str

async def estimate_vram_remote(
    model_alias: str,
    n_ctx: int = 2048,
    device_metric: Optional[str] = None
) -> EstimateResult:
    """
    Estimate VRAM and TPS using gguf-parser binary.
    """
    try:
        repo_id, filename = resolve_hf_alias(model_alias)
    except Exception as e:
        return {"error": f"Failed to resolve model: {str(e)}"}

    # Base command
    cmd = [
        "./gguf-parser-linux-amd64",
        "--hf-repo", repo_id,
        "--hf-file", filename,
        "--ctx-size", str(n_ctx),
        "--estimate",
        "--in-short",
        "--json"
    ]
    
    if device_metric:
        cmd.extend(["--device-metric", device_metric])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = json.loads(result.stdout)
        
        # Log for debugging
        print(f"GGUF Parser output: {output}")

        estimate = output.get("estimate", {})
        items = estimate.get("items", [])
        
        total_vram = 0
        total_ram = 0
        tps = 0.0

        for item in items:
            tps = item.get("maximumTokensPerSecond", tps)
            vrams = item.get("vrams", [])
            for vram_item in vrams:
                total_vram += vram_item.get("uma", 0) + vram_item.get("nonuma", 0)
            
            rams = item.get("ram", {})
            total_ram += rams.get("uma", 0) + rams.get("nonuma", 0)
            
        return {
            "vram": total_vram,
            "ram": total_ram,
            "tps": tps,
            "total_bytes": total_vram + total_ram,
            "weight_bytes": total_vram,
            "kv_cache_bytes": total_ram
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Estimation failed: {str(e)}"}

def get_total_gpu_memory(gpu_devices: list = (0,)) -> int:
    return 34208743424 # Hardcoded for AMD Radeon AI PRO R9700 detected during research

def get_free_gpu_memory(gpu_devices: list = (0,)) -> int:
    try:
        # Re-fetch from rocm-smi
        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True)
        data = json.loads(result.stdout)
        # Extract from card0
        return int(data.get("card0", {}).get("VRAM Total Used Memory (B)", 0))
    except:
        return 0
