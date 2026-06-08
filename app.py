import asyncio
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from database import mk_conn, init_database
from typing import cast, Optional
from conversation import (
    prepare_conversation_with_prompt,
    ConversationBlockedError,
    get_conversations as fetch_conversations,
    get_conversation_details,
    decide_tool_call,
    continue_conversation,
    delete_conversation
)
from repositories import router as repositories_router
from log_config import get_logger, setup_logging
from search import synchronize, semantic_search
from inference import registry, estimator, llama_cpp_server
from inference.gpu_benchmark import benchmark_tflops, benchmark_bandwidth
from inference.hf_gguf import list_cached_models
from tools import TOOL_REGISTRY

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


@app.post('/api/conversations/prompt')
async def prompt_model(request: Request):
    payload: dict[str, object] = await request.json()
    prompt = cast(str, payload.get('prompt', ''))
    provider_key = cast(Optional[str], payload.get('provider_key'))
    model_id = cast(Optional[str], payload.get('model_id'))
    scopes = cast(Optional[list[str]], payload.get('scopes'))
    if provider_key is None:
        return JSONResponse(status_code=400, content={'error': 'provider_key required'})
    if model_id is None:
        return JSONResponse(status_code=400, content={'error': 'model_id required'})
    conversation_id = cast(Optional[int], payload.get('conversation_id'))

    try:
        conn = mk_conn()
        conversation_id = prepare_conversation_with_prompt(conn, prompt, conversation_id, scopes=scopes)
    except ConversationBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": "Action required", "blocking_message_id": e.blocking_message_id}
        )

    available_tools = TOOL_REGISTRY
    return StreamingResponse(
        continue_conversation(conn, conversation_id, model_id, available_tools, provider_key=provider_key),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )


@app.get('/api/conversations')
async def get_conversations():
    conn = mk_conn()
    conversations = fetch_conversations(conn)
    return JSONResponse(content=conversations)


@app.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: int):
    conn = mk_conn()
    details = get_conversation_details(conn, conv_id)
    if not details:
        return JSONResponse(status_code=404, content={'error': 'Not found'})
    return JSONResponse(content=details)


@app.delete('/api/conversations/{conv_id}')
async def delete_conversation_endpoint(conv_id: int):
    try:
        conn = mk_conn()
        success = delete_conversation(conn, conv_id)
        if not success:
            return JSONResponse(status_code=404, content={'error': 'Conversation not found'})
        return JSONResponse(content={'status': 'deleted', 'id': conv_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


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


@app.post('/api/conversations/{conv_id}/tool_calls/{msg_id}/decide')
async def decide_tool_call_endpoint(conv_id: int, msg_id: int, request: Request):
    payload = await request.json()
    decision = payload.get('decision')
    comment = payload.get('comment')
    if decision not in {'approve', 'reject'}:
        return JSONResponse(status_code=400, content={'error': 'invalid decision'})

    try:
        conn = mk_conn()
        executed = await decide_tool_call(conn, conv_id, msg_id, decision, comment=comment)
        return JSONResponse(content={'status': 'success', 'executed': executed})
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/api/conversations/{conversation_id}/continue')
async def continue_conversation_endpoint(conversation_id: int, request: Request):
    provider_key = request.query_params.get('provider_key')
    model_id = request.query_params.get('model_id')
    if not provider_key:
        return JSONResponse(status_code=400, content={'error': 'provider_key query parameter required'})
    if not model_id:
        return JSONResponse(status_code=400, content={'error': 'model_id query parameter required'})
    conn = mk_conn()
    details = get_conversation_details(conn, conversation_id)
    if not details:
        return JSONResponse(status_code=404, content={'error': 'conversation not found'})
    available_tools = TOOL_REGISTRY
    return StreamingResponse(
        continue_conversation(conn, conversation_id, model_id, available_tools, provider_key=provider_key),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )


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