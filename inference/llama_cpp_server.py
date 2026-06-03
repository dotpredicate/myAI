import os
import subprocess
import time
import httpx
import openai
from typing import List, Any, Dict

from domain import ConversationElement
from .openai import DeltaProcessor

# LLM Configuration
LLAMA_CPP_ENDPOINT = os.getenv('LLAMA_CPP_ENDPOINT', 'http://localhost:1234')
completions_endpoint = openai.Client(api_key='dummy', base_url=LLAMA_CPP_ENDPOINT)

_server_processes: Dict[str, subprocess.Popen] = {}

def _start_server(port: str, args: List[str]):
    global _server_processes
    if port not in _server_processes:
        try:
            cmd = ["llama-server", "--port", port] + args
            print(f"Starting server on port {port}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid
            )
            _server_processes[port] = proc
        except FileNotFoundError:
            print("Error: llama-server not found in PATH.")
            raise

def _wait_for_server(port: str, timeout: int = 15):
    """Wait for the server to respond to HTTP requests."""
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()
    print(f"Waiting for server on port {port} to be ready...")
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code in (200, 401):  # 401 is also valid for dummy API key
                print(f"Server on port {port} is ready.")
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server on port {port} did not start within {timeout} seconds.")

def _ensure_server_started():
    _start_server("1234", ["--offline", "--jinja", "--chat-template-kwargs", '{"preserve_thinking":true}'])

def ensure_embedding_server_started():
    _start_server("2345", ["--embedding", "-hf", "unsloth/embeddinggemma-300m-GGUF"])
    _wait_for_server("2345")

def stop():
    global _server_processes
    for port, process in list(_server_processes.items()):
        print(f"Terminating llama-server process {process.pid} on port {port}")
        process.terminate()
        process.wait()
        del _server_processes[port]

def run_chat_completion_stream(model_id: str, context: list[tuple[int, ConversationElement]], functions: List[Any]):
    from .openai import _to_oai_messages
    _ensure_server_started()
    return completions_endpoint.chat.completions.create(
        model=model_id,
        messages=_to_oai_messages(context),
        reasoning_effort='high',
        stream=True,
        tools=functions,
    )

def list_models():
    _ensure_server_started()
    return completions_endpoint.models.list()
