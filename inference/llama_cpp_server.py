import os
import subprocess
import time
import openai
import atexit
from typing import List, Generator, Any
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

_server_process = None

def _ensure_server_started():
    global _server_process
    if _server_process is None:
        try:
            _server_process = subprocess.Popen(
                ["llama-server", "--offline", "--port", "1234"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        except FileNotFoundError:
            print("Error: llama-server not found in PATH.")
            raise

def stop():
    global _server_process
    if _server_process:
        print(f"Terminating llama-server process {_server_process.pid}")
        _server_process.terminate()
        _server_process.wait()
        _server_process = None

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
