from enum import StrEnum
from typing import Any, Literal, Optional, Union, List
from pydantic import BaseModel, TypeAdapter

class SecurityPolicy(StrEnum):
    READ_ONLY = 'read-only'
    PRIVILEGED_WRITE = 'privileged-write'
    WRITE = 'write'

def scope_policy_is_escalation(override: Optional[SecurityPolicy], base_policy: SecurityPolicy) -> bool:
    if override is None:
        return False
    order: list[SecurityPolicy] = [SecurityPolicy.READ_ONLY, SecurityPolicy.PRIVILEGED_WRITE, SecurityPolicy.WRITE]
    return order.index(override) > order.index(base_policy)

class ScopeSpec(BaseModel):
    internal_name: str
    security_policy: SecurityPolicy

class RepositoryConfig(BaseModel):
    id: int
    display_name: str
    internal_name: str
    repo_type: Literal['plain', 'git']
    path: str
    security_policy: SecurityPolicy

class AgentRepositoryAccess(BaseModel):
    id: int = 0
    agent_id: int = 0
    repository_id: int
    repository_internal_name: str = ''
    security_policy_override: Optional[SecurityPolicy] = None

class AgentConfig(BaseModel):
    id: int
    display_name: str
    internal_name: str
    description: str
    provider_key: str
    model_id: str
    inference_config: dict[str, Any]
    repository_access: List[AgentRepositoryAccess]

class Message(BaseModel):
    type: Literal['message'] = 'message'
    author: Literal['user'] | Literal['assistant']
    content: str
    scopes: List[ScopeSpec]

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

ConversationElement = Union[
    Message, 
    Thinking, 
    ToolCallFinishedOrBlocked, 
    ToolCallResult, 
    ToolCallDecision
]

stored_element_adapter: TypeAdapter[ConversationElement] = TypeAdapter(ConversationElement)
