import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from database import init_database
from typing import Optional
from conversation import router as conversations_router
from domain import (
    LLAMA_CPP_PROVIDER_KEY,
    InferenceProviderConfig,
    InferenceProviderInput,
    InferenceProviderUpdate,
)
from scopes import UserScopeChoice, resolve_scopes
from repositories import router as repositories_router
from agents import provider_in_use, router as agents_router
from projects import router as projects_router
from images import router as images_router
from inference.sd_cpp.router import router as sd_cpp_router
from inference.providers import (
    create_provider_config,
    delete_provider_config,
    get_provider_config,
    list_provider_configs,
    reload_providers,
    update_provider_config,
)
from log_config import get_logger, setup_logging
from search import synchronize, semantic_search, clear_index
from inference import managed_processes, registry
from inference.gpu_benchmark import benchmark_tflops, benchmark_bandwidth
from inference.llama_cpp import benchmark
from inference.llama_cpp.gguf import list_cached_models

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_database()
    await reload_providers()
    asyncio.create_task(synchronize())
    yield
    await managed_processes.stop_servers()

app = FastAPI(title="MyAI FastAPI", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "public"


class CreateProviderRequest(InferenceProviderInput):
    key: str


def _provider_response(config: InferenceProviderConfig) -> dict[str, object]:
    return {
        'key': config.key,
        'display_name': config.display_name,
        'description': config.description,
        'base_url': config.base_url,
        'has_api_key': config.has_api_key,
        'models': config.models,
        'timeout_seconds': config.timeout_seconds,
        'extra_body': config.extra_body,
        'include_reasoning_content': config.include_reasoning_content,
    }


@app.get('/api/providers')
async def get_providers():
    configs = {config.key: config for config in await list_provider_configs()}
    providers = []
    for registration in registry.list_registrations():
        config = configs.get(registration.key)
        providers.append({
            "key": registration.key,
            "display_name": config.display_name if config else registration.display_name,
            "description": config.description if config else registration.description,
            "kind": "openai_compatible" if config else "builtin",
            "has_api_key": config.has_api_key if config else False,
        })
    return JSONResponse(content={"providers": providers})


@app.get('/api/providers/{provider_key}')
async def get_provider(provider_key: str):
    config = await get_provider_config(provider_key)
    if config is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    return JSONResponse(content=_provider_response(config))


@app.post('/api/providers')
async def create_provider(payload: CreateProviderRequest):
    key = payload.key.strip()
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', key):
        return JSONResponse(status_code=400, content={'error': 'key must contain only letters, numbers, dots, hyphens and underscores'})
    if key == LLAMA_CPP_PROVIDER_KEY or any(r.key == key for r in registry.list_registrations()):
        return JSONResponse(status_code=409, content={'error': f"provider '{key}' already exists"})

    config = InferenceProviderConfig(key=key, **payload.model_dump(exclude={'key'}))
    try:
        await create_provider_config(config)
    except psycopg.errors.UniqueViolation:
        return JSONResponse(status_code=409, content={'error': f"provider '{key}' already exists"})
    await reload_providers()
    return JSONResponse(content=_provider_response(config), status_code=201)


@app.patch('/api/providers/{provider_key}')
async def update_provider(provider_key: str, payload: InferenceProviderUpdate):
    existing = await get_provider_config(provider_key)
    if existing is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    if not payload.model_fields_set:
        return JSONResponse(content=_provider_response(existing))

    await update_provider_config(provider_key, payload)
    await reload_providers()
    result = await get_provider_config(provider_key)
    if result is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    return JSONResponse(content=_provider_response(result))


@app.delete('/api/providers/{provider_key}')
async def delete_provider(provider_key: str):
    existing = await get_provider_config(provider_key)
    if existing is None:
        return JSONResponse(status_code=404, content={'error': 'provider not found'})
    if await provider_in_use(provider_key):
        return JSONResponse(status_code=409, content={'error': 'provider is used by one or more agents or projects'})
    await delete_provider_config(provider_key)
    await reload_providers()
    return JSONResponse(content={'status': 'deleted', 'key': provider_key})


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


@app.get('/api/providers/{provider_key}/inference-params')
async def get_inference_params(provider_key: str):
    try:
        provider = registry.get(provider_key)
        return provider.get_inference_params()
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"Provider '{provider_key}' not found"})


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
app.include_router(projects_router)
app.include_router(conversations_router)
app.include_router(images_router)
app.include_router(sd_cpp_router)


class SearchRequest(BaseModel):
    query: str = ''
    top_k: int = 3
    scopes: list[str] = Field(default_factory=list)
    agent_id: str | None = None


@app.post('/api/search')
async def search(payload: SearchRequest):
    if not payload.query:
        return JSONResponse(status_code=400, content={'error': 'query required'})

    choices = [UserScopeChoice(internal_name=name) for name in payload.scopes]
    try:
        resolved_scopes = await resolve_scopes(payload.agent_id, choices)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={'error': str(exc)})
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


@app.get('/api/gpu-stats')
async def gpu_stats():
    stats = await asyncio.to_thread(benchmark.get_gpu_stats)
    return JSONResponse(content={"free": stats.free_bytes, "total": stats.total_bytes})


@app.get('/api/managed-processes/list')
async def managed_processes_list():
    return {"servers": managed_processes.get_server_status()}


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
