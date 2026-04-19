from typing import Dict, Optional, Any, Union, List, NamedTuple
from llama_cpp import Llama
from .hf_gguf import list_cached_models, resolve_hf_alias

# Simple model cache
_loaded_model: Optional[Llama] = None
_loaded_model_id: Optional[str] = None

class ModelInfo:
    def __init__(self, id: str):
        self.id = id

def get_llm(model_id: str) -> Llama:
    global _loaded_model, _loaded_model_id
    if _loaded_model is not None and _loaded_model_id == model_id:
        return _loaded_model

    print(f"Loading model: {model_id}")
    repo_id, filename = resolve_hf_alias(model_id)
    
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    _loaded_model = Llama(
        model_path=model_path,
        model_id=model_id,
        seed=-1,
        n_ctx=0,
        flash_attn=True,
        verbose=False,
    )
    _loaded_model_id = model_id
    return _loaded_model

class Message(NamedTuple):
    content: str
class Thinking(NamedTuple):
    content: str
class ToolCall(NamedTuple):
    name: str
    parameters: str
StreamingElement = Union[Thinking, Message, ToolCall]

class ToolCallResult(NamedTuple):
    name: str
    parameters: str
    result: str
    is_blocking: bool = False
FinishedElement = Union[Thinking, Message, ToolCallResult]

# Mappings and Helpers remain similarly structured as before
def _to_oai_elements(message_id: int, element: Union[FinishedElement, dict]) -> list[dict[str, Any]]:
    # Internal helper for OAI format mapping
    if isinstance(element, dict):
        # Compatibility for historical messages from DB
        elem_type = element['type']
        if elem_type == 'thinking':
            return []
        if elem_type == 'message':
            return [{'role': element.get('role', 'user'), 'content': element['content']}]
        if elem_type == 'tool_call':
            tool_call = {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': str(message_id),
                    'type': 'function',
                    'function': {
                        'name': element['name'],
                        'arguments': element['parameters'],
                    },
                }]
            }
            res = [tool_call]
            if element.get('status') != 'pending':
                res.append({
                    'role': 'tool',
                    'tool_call_id': str(message_id),
                    'content': str(element.get('result', ''))
                })
            return res
        if elem_type == 'tool_result':
            return [{
                'role': 'tool',
                'tool_call_id': str(element['original_message_id']),
                'content': str(element.get('result', ''))
            }]
        if elem_type == 'tool_decision' and element.get('decision') == 'reject':
            return [{
                'role': 'tool',
                'tool_call_id': str(element['original_message_id']),
                'content': "User rejected this tool call proposal."
            }]
        return []

    match element:
        case Message(content=content):
            return [{'role': 'assistant', 'content': content}]
        case Thinking():
            return []
        case ToolCall(name=name, parameters=params):
            return [{
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': str(message_id),
                    'type': 'function',
                    'function': {'name': name, 'arguments': params},
                }]
            }]
        case ToolCallResult(name=name, parameters=params, result=result):
            # For context, we need both the call and the result
            return [
                {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{
                        'id': str(message_id),
                        'type': 'function',
                        'function': {'name': name, 'arguments': params},
                    }]
                },
                {
                    'role': 'tool',
                    'tool_call_id': str(message_id),
                    'content': result
                }
            ]
    return []

class ChatContext:
    def __init__(self, messages: List[Dict[str, Any]] = None):
        self._messages = messages or []

    def append_from_db(self, message_id: int, element: dict):
        self._messages.extend(_to_oai_elements(message_id, element))

    def append_finalized(self, message_id: int, element: FinishedElement):
        self._messages.extend(_to_oai_elements(message_id, element))

    def to_list(self) -> List[Dict[str, Any]]:
        return self._messages

class DeltaProcessor:
    def __init__(self):
        self.buffered_element: Optional[StreamingElement] = None

    def process(self, chunk: Any) -> tuple[Optional[StreamingElement], Optional[StreamingElement]]:
        # llama-cpp-python returns chunks differently than OpenAI
        choice = chunk['choices'][0]
        delta = choice['delta']
        
        new_elem: Optional[StreamingElement] = None
        
        if 'tool_calls' in delta:
            func = delta['tool_calls'][0]['function']
            new_elem = ToolCall(func.get('name') or "", func.get('arguments') or "")
        elif 'content' in delta and delta['content'] is not None:
            new_elem = Message(delta['content'])
        # Add logic for reasoning if supported by the model/library
        
        finalized: Optional[StreamingElement] = None
        match (self.buffered_element, new_elem):
            case (None, elem):
                self.buffered_element = elem
            case (Message(c1), Message(c2)):
                self.buffered_element = Message(c1 + c2)
            # Add other cases for aggregation
            case _:
                finalized = self.buffered_element
                self.buffered_element = new_elem
        return new_elem, finalized

def run_chat_completion_stream(model_id: str, context: ChatContext, functions: List[Any]):
    llm = get_llm(model_id)
    return llm.create_chat_completion(
        messages=context.to_list(),
        tools=functions,
        stream=True
    )

def list_models():
    return [ModelInfo(id=m) for m in list_cached_models()]
