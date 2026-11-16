import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

import backend.inference.managed_processes as managed_processes
from backend.inference.llama_cpp import benchmark
from backend.inference.llama_cpp.gguf_cache import (
    delete_cached_model,
    list_cached_model_entries,
)
from backend.log_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/llama-cpp", tags=["llama-cpp"])


def _server_using(model_id: str) -> managed_processes.ServerInfo | None:
    wanted = model_id.lower()
    for server in managed_processes.get_server_status():
        if server.model.lower() == wanted:
            return server
    return None


@router.get("/cached-models")
async def get_cached_models():
    try:
        models = await asyncio.to_thread(list_cached_model_entries)
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.error("Error fetching cached models: %s", e)
        return JSONResponse(content={"models": []})


@router.delete("/cached-models/{model:path}")
async def delete_cached_model_endpoint(model: str):
    server = _server_using(model)
    if server is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": (
                    f"\u201c{model}\u201d is loaded by the model server on port "
                    f"{server.port}. Stop it before deleting the files."
                )
            },
        )
    try:
        freed = await asyncio.to_thread(delete_cached_model, model)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except OSError as e:
        logger.exception("Could not delete cached model %s", model)
        return JSONResponse(status_code=500, content={"error": f"Could not delete the files: {e}"})
    try:
        await benchmark.delete_benchmark_points(model)
    except Exception:
        logger.exception("Could not drop benchmark results of %s", model)
    return JSONResponse(content={"status": "deleted", "model": model, "freed_bytes": freed})


@router.get('/models/{model:path}/info')
async def model_info_endpoint(model: str):
    return JSONResponse(content=await benchmark.model_info(model))


@router.get('/models/{model:path}/benchmark/stream')
async def benchmark_model_stream_endpoint(
    model: str,
    quant: Optional[list[str]] = Query(default=None),
    n_gen: int = 128,
    n_prompt: int = 512,
    repetitions: int = 1,
):
    async def events():
        async for result in benchmark.benchmark_model_stream(
            model,
            n_gen=n_gen,
            n_prompt=n_prompt,
            repetitions=repetitions,
            quants=quant,
        ):
            yield json.dumps(result) + '\n'

    return StreamingResponse(events(), media_type='application/x-ndjson')
