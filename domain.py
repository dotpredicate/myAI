from pydantic import BaseModel, TypeAdapter
from typing import Literal, Optional, Union

class Message(BaseModel):
    type: Literal['message'] = 'message'
    author: Literal['user'] | Literal['assistant']
    content: str
    scopes: list[str]


class Thinking(BaseModel):
    type: Literal['thinking'] = 'thinking'
    content: str


class ToolCallFinishedOrBlocked(BaseModel):
    type: Literal['tool_call'] = 'tool_call'
    name: str
    parameters: str
    result: str
    is_blocking: bool
    status: Literal['pending'] | Literal['completed']


class ToolCallResult(BaseModel):
    type: Literal['tool_call_result'] = 'tool_call_result'
    original_message_id: int
    result: str


class ToolCallDecision(BaseModel):
    type: Literal['tool_call_decision'] = 'tool_call_decision'
    original_message_id: int
    decision: Literal['approve'] | Literal['reject']
    comment: Optional[str] = None


ConversationElement = Union[Message, Thinking, ToolCallFinishedOrBlocked, ToolCallResult, ToolCallDecision]
stored_element_adapter = TypeAdapter(ConversationElement)