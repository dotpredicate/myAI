from fastapi import BackgroundTasks, FastAPI, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from database import mk_conn, init_database
from typing import Optional, Any
from conversation import (
    prepare_conversation_with_prompt,
    ConversationBlockedError,
    get_conversations as fetch_conversations,
    get_conversation_details,
    decide_tool_call,
    continue_conversation
)
from inference import list_models
from tools import TOOL_REGISTRY
FUNCTIONS = [t['schema'] for t in TOOL_REGISTRY]


app = FastAPI(title="MyAI FastAPI")


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.get('/api/models')
async def get_models():
    try:
        res = list_models()
        models = list(res)
    except Exception as e:
        print(f"Error fetching models: {e}")
        models = []
    return JSONResponse(content={"models": [{'id': m.id, 'name': m.id} for m in models]})

@app.post('/api/conversations/prompt')
async def prompt_model(request: Request):
    payload: dict[str, Any] = await request.json()
    prompt: str = payload.get('prompt', '')
    model_id: Optional[str] = payload.get('model_id')
    if model_id is None:
        return JSONResponse(status_code=400, content={'error': 'model_id required'})
    conversation_id: Optional[int] = payload.get('conversation_id')

    try:
        with mk_conn() as conn:
            conversation_id = prepare_conversation_with_prompt(conn, prompt, conversation_id)
    except ConversationBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": "Action required", "blocking_message_id": e.blocking_message_id}
        )

    return StreamingResponse(
        continue_conversation(conn, conversation_id, model_id, FUNCTIONS),
        media_type='application/x-ndjson', headers={'X-Conversation-ID': str(conversation_id)}
    )

@app.get('/api/conversations')
async def get_conversations():
    with mk_conn() as conn:
        conversations = fetch_conversations(conn)
    return JSONResponse(content=conversations)

@app.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: int):
    with mk_conn() as conn:
        details = get_conversation_details(conn, conv_id)
        if not details:
            return JSONResponse(status_code=404, content={'error': 'Not found'})
        return JSONResponse(content=details)


import index

@app.post('/api/search')
async def search(payload: dict = Body(...)):
    query_text: str = payload.get('query', '')
    top_k: int = int(payload.get('top_k', 3))
    if not query_text:
        return JSONResponse(status_code=400, content={'error': 'query required'})
    results = []
    for hit in index.semantic_search(query_text, top_k):
        snippet = hit.text[:200] + ('...' if len(hit.text) > 200 else '')
        results.append({'id': hit.document_id, 'title': f"{hit.file_path} ({hit.chunk_index})", 'snippet': snippet, 'score': hit.score})
    return JSONResponse(content={'results': results})

@app.post("/api/sync")
async def sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(index.synchronize)
    return JSONResponse({"status": "sync started"})

@app.on_event("startup")
async def startup_event():
    with mk_conn() as conn:
        init_database(conn)
    index.synchronize()

@app.post('/api/conversations/{conv_id}/tool_calls/{msg_id}/decide')
async def decide_tool_call_endpoint(conv_id: int, msg_id: int, request: Request):
    """Handle the user's approval or rejection of a tool call proposal."""
    payload = await request.json()
    decision = payload.get('decision')
    if decision not in {'approve', 'reject'}:
        return JSONResponse(status_code=400, content={'error': 'invalid decision'})

    try:
        with mk_conn() as conn:
            executed = decide_tool_call(conn, conv_id, msg_id, decision)
        return JSONResponse(content={'status': 'success', 'executed': executed})
    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.get('/api/conversations/{conv_id}/continue')
async def continue_conversation_endpoint(conv_id: int, request: Request):
    """Resume a conversation after a proposal decision.

    The client must supply the ``model_id`` as a query parameter. The endpoint
    streams the assistant output until a new proposal is generated or the
    conversation ends.
    """
    model_id = request.query_params.get('model_id')
    if not model_id:
        return JSONResponse(status_code=400, content={'error': 'model_id query parameter required'})
    with mk_conn() as conn:
        details = get_conversation_details(conn, conv_id)
        if not details:
            return JSONResponse(status_code=404, content={'error': 'conversation not found'})
        return StreamingResponse(continue_conversation(conn, conv_id, model_id, FUNCTIONS), media_type='application/x-ndjson')

# Mount static files
app.mount("/", StaticFiles(directory="static"), name="static")
