import os
import subprocess
import time
import openai
import atexit
from typing import List, Generator, Any, Dict
from .openai import (
    Message,
    Thinking,
    ToolCall,
    ToolCallResult,
    StreamingElement,
    FinishedElement,
    DeltaProcessor,
    ChatContext,
    _to_oai_elements
)

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            _server_processes[port] = proc
        except FileNotFoundError:
            print("Error: llama-server not found in PATH.")
            raise

def _ensure_server_started():
    _start_server("1234", ["--offline", "--jinja"])

def ensure_embedding_server_started():
    _start_server("2345", ["--embedding", "-hf", "unsloth/embeddinggemma-300m-GGUF"])

def stop():
    global _server_processes
    for port, process in list(_server_processes.items()):
        print(f"Terminating llama-server process {process.pid} on port {port}")
        process.terminate()
        process.wait()
        del _server_processes[port]

def run_chat_completion_stream(model_id: str, context: ChatContext, functions: List[Any]):
    _ensure_server_started()
    return completions_endpoint.chat.completions.create(
        model=model_id,
        messages=context.to_list(),
        reasoning_effort='high',
        stream=True,
        tools=functions,
    )

def list_models():
    _ensure_server_started()
    return completions_endpoint.models.list()
