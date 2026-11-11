import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from database import init_database
from typing import Optional
from conversation import router as conversations_router
from scopes import resolve_workspace_scopes
from repositories import router as repositories_router, get_repo_by_id
from agents import router as agents_router
from workspaces import router as workspaces_router
from images import router as images_router
from inference.sd_cpp.router import router as sd_cpp_router
from inference.providers import reload_providers, router as providers_router
from log_config import get_logger, setup_logging
from search import schedule_sync, synchronize, semantic_search, clear_index
from inference import managed_processes
from inference.gpu_benchmark import benchmark_tflops, benchmark_bandwidth
from inference.llama_cpp import benchmark
from inference.llama_cpp.gguf import list_cached_models
from system_metrics import sample_system_stats

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_database()
    await reload_providers()
    schedule_sync()
    yield
    await managed_processes.stop_servers()

app = FastAPI(title="MyAI FastAPI", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "web"


@app.get('/api/info')
async def server_info():
    return JSONResponse(content={'pid': os.getpid()})


@app.get('/api/cached-models')
async def get_cached_models():
    try:
        models = list_cached_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.error("Error fetching cached models: %s", e)
        return JSONResponse(content={"models": []})


app.include_router(providers_router)
app.include_router(repositories_router)
app.include_router(agents_router)
app.include_router(workspaces_router)
app.include_router(conversations_router)
app.include_router(images_router)
app.include_router(sd_cpp_router)


class SearchRequest(BaseModel):
    query: str = ''
    top_k: int = 3
    workspace_id: int | None = None
    repository_id: int | None = None


@app.post('/api/search')
async def search(payload: SearchRequest):
    if not payload.query:
        return JSONResponse(status_code=400, content={'error': 'query required'})

    if payload.repository_id is not None:
        repo = await get_repo_by_id(payload.repository_id)
        if repo is None:
            return JSONResponse(status_code=404, content={'error': 'repository not found'})
        scope_names = [repo.internal_name]
    else:
        resolved_scopes = await resolve_workspace_scopes(payload.workspace_id)
        scope_names = [scope.internal_name for scope in resolved_scopes]

    results = []
    for hit in await semantic_search(payload.query, payload.top_k, scopes=scope_names):
        snippet = hit.text[:200] + ('...' if len(hit.text) > 200 else '')
        results.append({
            'id': hit.document_id,
            'title': f"{hit.file_path} ({hit.chunk_index})",
            'snippet': snippet,
            'cosine_distance': hit.cosine_distance,
        })
    return JSONResponse(content={'results': results})


@app.delete('/api/search/index')
async def clear_search_index():
    await clear_index()
    return JSONResponse(content={"status": "cleared"})


@app.get('/models/{model:path}/info')
async def model_info_endpoint(model: str):
    return JSONResponse(content=await benchmark.model_info(model))


@app.get('/models/{model:path}/benchmark')
async def benchmark_model_endpoint(
    model: str,
    quant: Optional[list[str]] = Query(default=None),
    n_gen: int = 128,
    n_prompt: int = 512,
    repetitions: int = 1,
):
    return JSONResponse(content=await benchmark.benchmark_model(
        model, n_gen=n_gen, n_prompt=n_prompt, repetitions=repetitions, quants=quant
    ))


@app.get('/api/system-stats')
async def system_stats():
    def snapshot() -> dict[str, object]:
        return {
            "resources": sample_system_stats(),
            "servers": managed_processes.get_server_status(),
        }

    return await asyncio.to_thread(snapshot)


@app.post('/api/managed-processes/{port}/stop')
async def stop_managed_process_endpoint(port: str):
    if not port.isdigit() or not await managed_processes.stop_server(port):
        return JSONResponse(status_code=404, content={"error": f"process on port {port} not found"})
    return JSONResponse(content={"status": "stopped", "port": port})


@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(synchronize)
    return JSONResponse(status_code=202, content={"status": "accepted"})


@app.get('/api/gpu-benchmark')
async def run_gpu_benchmark(
    tflops_size: int = 4096, 
    bw_size: int = 8192
):
    try:
        tflops_results = await asyncio.to_thread(
            benchmark_tflops,
            M=tflops_size,
            K=tflops_size,
            N=tflops_size,
        )
        bandwidth_results = await asyncio.to_thread(
            benchmark_bandwidth,
            R=bw_size,
            C=bw_size,
        )
        
        return JSONResponse(content={
            "status": "success",
            "tflops": tflops_results,
            "bandwidth": bandwidth_results
        })
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )

# Mount static files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
