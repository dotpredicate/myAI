import asyncio
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from database import init_database
from typing import Optional
from conversation import router as conversations_router
from repositories import router as repositories_router
from agents import router as agents_router
from log_config import get_logger, setup_logging
from search import synchronize, semantic_search
from inference import registry, estimator, llama_cpp_server
from inference.gpu_benchmark import benchmark_tflops, benchmark_bandwidth
from inference.hf_gguf import list_cached_models

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_database()
    asyncio.create_task(synchronize())
    yield
    await llama_cpp_server.stop_llama_servers()

app = FastAPI(title="MyAI FastAPI", lifespan=lifespan)


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.get('/api/providers')
async def get_providers():
    registrations = registry.list_registrations()
    return JSONResponse(content={
        "providers": [
            {"key": r.key, "display_name": r.display_name, "description": r.description}
            for r in registrations
        ]
    })


@app.get('/api/providers/{provider_key}/models')
async def get_models(provider_key: str):
    try:
        provider = registry.get(provider_key)
        models = await provider.list_models()
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"Provider '{provider_key}' not found"})
    except Exception as e:
        logger.error("Error fetching models for provider '%s': %s", provider_key, e)
        models = []
    return JSONResponse(content={"models": [{'id': m.id, 'name': m.id} for m in models]})


@app.get('/api/cached-models')
async def get_cached_models():
    try:
        models = list_cached_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.error("Error fetching cached models: %s", e)
        return JSONResponse(content={"models": []})


app.include_router(repositories_router)
app.include_router(agents_router)
app.include_router(conversations_router)


@app.post('/api/search')
async def search(payload: dict = Body(...)):
    query_text: str = payload.get('query', '')
    top_k: int = int(payload.get('top_k', 3))
    scopes: Optional[list[str]] = payload.get('scopes')
    if not query_text:
        return JSONResponse(status_code=400, content={'error': 'query required'})
    results = []
    for hit in await semantic_search(query_text, top_k, scopes=scopes):
        snippet = hit.text[:200] + ('...' if len(hit.text) > 200 else '')
        results.append({'id': hit.document_id, 'title': f"{hit.file_path} ({hit.chunk_index})", 'snippet': snippet, 'score': hit.score})
    return JSONResponse(content={'results': results})


@app.get('/api/estimate')
async def estimate_model(model_id: str, n_ctx: int = 2048, device_metric: Optional[str] = None):
    data = await estimator.estimate_vram_remote(model_id, n_ctx=n_ctx, device_metric=device_metric)
    return JSONResponse(content=data)


@app.get('/api/gpu-stats')
async def gpu_stats():
    stats = estimator.get_gpu_stats()
    return JSONResponse(content={"free": stats.free_bytes, "total": stats.total_bytes})


@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(synchronize)
    return JSONResponse({"status": "sync started"})


@app.get('/api/gpu-benchmark')
async def run_gpu_benchmark(
    tflops_size: int = 4096, 
    bw_size: int = 8192
):
    try:
        tflops_results = benchmark_tflops(M=tflops_size, K=tflops_size, N=tflops_size)
        bandwidth_results = benchmark_bandwidth(R=bw_size, C=bw_size)
        
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
app.mount("/", StaticFiles(directory="static"), name="static")