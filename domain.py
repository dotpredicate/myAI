from enum import StrEnum
from typing import Any, Literal, Optional, Union, List
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

class SecurityPolicy(StrEnum):
    READ_ONLY = 'read-only'
    PRIVILEGED_WRITE = 'privileged-write'
    WRITE = 'write'

def scope_policy_is_escalation(override: Optional[SecurityPolicy], base_policy: SecurityPolicy) -> bool:
    if override is None:
        return False
    order: list[SecurityPolicy] = [SecurityPolicy.READ_ONLY, SecurityPolicy.PRIVILEGED_WRITE, SecurityPolicy.WRITE]
    return order.index(override) > order.index(base_policy)


def normalize_scope_name(name: str) -> str:
    """Strip accidental whitespace and surrounding slashes from a repository name."""
    return name.strip().strip('/')

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

class RepositoryAccessEntry(BaseModel):
    repository_id: int
    security_policy_override: Optional[SecurityPolicy] = None

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
    instructions: Optional[str] = None
    provider_key: str
    model_id: str
    inference_config: dict[str, Any]
    repository_access: List[AgentRepositoryAccess]
    spawnable_agents: List[str] = []

class ProjectRepositoryAccess(BaseModel):
    id: int = 0
    project_id: int = 0
    repository_id: int
    repository_internal_name: str = ''
    security_policy_override: Optional[SecurityPolicy] = None

class ProjectConfig(BaseModel):
    id: int
    display_name: str
    internal_name: str
    description: str
    instructions: Optional[str] = None
    default_agent: Optional[int] = None
    provider_key: Optional[str] = None
    model_id: Optional[str] = None
    inference_config: dict[str, Any] = {}
    repository_access: List[ProjectRepositoryAccess] = []


class _InferenceProviderRules:
    """Validation shared by provider create and update payloads.

    Validators run on explicitly provided fields only. On update, ``None``
    clears clearable fields and is rejected for required ones.
    """

    @field_validator('display_name')
    @classmethod
    def _display_name(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError('display_name cannot be null')
        value = value.strip()
        if not value:
            raise ValueError('display_name is required')
        return value

    @field_validator('description')
    @classmethod
    def _description(cls, value: Optional[str]) -> str:
        return (value or '').strip()

    @field_validator('base_url')
    @classmethod
    def _base_url(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError('base_url cannot be null')
        value = value.strip().rstrip('/')
        if not value.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return value

    @field_validator('timeout_seconds')
    @classmethod
    def _timeout_seconds(cls, value: Optional[int]) -> int:
        if value is None or value <= 0:
            raise ValueError('timeout_seconds must be positive')
        return value

    @field_validator('include_reasoning_content')
    @classmethod
    def _include_reasoning_content(cls, value: Optional[bool]) -> bool:
        if value is None:
            raise ValueError('include_reasoning_content cannot be null')
        return value

    @field_validator('models')
    @classmethod
    def _models(cls, value: Optional[List[str]]) -> List[str]:
        return value or []

    @field_validator('headers')
    @classmethod
    def _headers(cls, value: Optional[dict[str, str]]) -> dict[str, str]:
        result = value or {}
        for key, item in result.items():
            if any(char in key or char in item for char in ('\r', '\n')):
                raise ValueError('headers must not contain newlines')
        return result

    @field_validator('extra_body')
    @classmethod
    def _extra_body(cls, value: Optional[dict[str, object]]) -> dict[str, object]:
        return value or {}

    @field_validator('api_key')
    @classmethod
    def _api_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError('api_key cannot be empty; send null to clear it')
        if '\r' in value or '\n' in value:
            raise ValueError('api_key must not contain newlines')
        return value


class InferenceProviderInput(_InferenceProviderRules, BaseModel):
    model_config = ConfigDict(extra='forbid')

    display_name: str
    description: str = ''
    base_url: str
    api_key: Optional[str] = None
    models: List[str] = []
    timeout_seconds: int = 120
    headers: dict[str, str] = {}
    extra_body: dict[str, object] = {}
    include_reasoning_content: bool = False


class InferenceProviderConfig(InferenceProviderInput):
    key: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


class InferenceProviderUpdate(_InferenceProviderRules, BaseModel):
    """Partial provider update.

    Omitted fields stay unchanged. Required fields reject an explicit null;
    clearable fields accept null to clear them.
    """

    model_config = ConfigDict(extra='forbid')

    display_name: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    models: Optional[List[str]] = None
    timeout_seconds: Optional[int] = None
    headers: Optional[dict[str, str]] = None
    extra_body: Optional[dict[str, object]] = None
    include_reasoning_content: Optional[bool] = None


LLAMA_CPP_PROVIDER_KEY = 'llama.cpp'

class Message(BaseModel):
    type: Literal['message'] = 'message'
    author: Literal['user'] | Literal['assistant']
    content: str

class Thinking(BaseModel):
    type: Literal['thinking'] = 'thinking'
    content: str

class ToolCall(BaseModel):
    type: Literal['tool_call'] = 'tool_call'
    name: str
    parameters: str
    result: str
    scopes: List[ScopeSpec] = []

class ApprovalRequest(BaseModel):
    type: Literal['approval_request'] = 'approval_request'
    original_message_id: int

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
    ToolCall,
    ApprovalRequest,
    ToolCallResult, 
    ToolCallDecision
]

stored_element_adapter: TypeAdapter[ConversationElement] = TypeAdapter(ConversationElement)
