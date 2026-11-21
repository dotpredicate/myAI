from dataclasses import dataclass, field
import asyncio
import json
import subprocess
import difflib
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Literal, Optional, TypedDict, TypeVar, cast

from pydantic import BaseModel, ValidationError

from backend.system import REPOSITORIES_VROOT, get_repo_from_vpath, is_safe_realpath, is_safe_vpath, resolve_repo_vpath, resolve_space_vpath, run_sandboxed_command
from backend.domain import (
    AgentConfig,
    ApplyPatchInput,
    ApplyPatchSuccess,
    BashInput,
    BashSuccess,
    Message,
    Prompt,
    ReplaceFileInput,
    ReplaceFileSuccess,
    ScopeSpec,
    SearchHit as SearchHitOutput,
    SearchInput,
    SearchSuccess,
    SpawnAgentInput,
    SpawnAgentSuccess,
    ToolArguments,
    ToolError,
    ToolOutcome,
    ToolProposal,
)
from backend.database import mk_conn


class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolExecution:
    arguments: Optional[ToolArguments]
    outcome: ToolOutcome


@dataclass(frozen=True)
class ToolContext:
    conversation_id: int
    workspace_id: int
    privileged: bool = False
    scopes: list[ScopeSpec] = field(default_factory=list)
    agent: AgentConfig | None = None
    spawn_depth: int = 0


InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT')


class Tool(TypedDict, Generic[InputT, OutputT]):
    name: str
    schema: FunctionDefinition
    input: type[InputT]
    executor: Callable[[InputT, ToolContext], Awaitable[OutputT]]


# The registry holds tools that take different inputs.
RegisteredTool = Tool[Any, ToolOutcome]


MAX_SPAWN_DEPTH = 3


def proposal(
    change_format: Literal['replace', 'patch'],
    target: Path,
    source: Path,
    diff: str,
) -> ToolProposal:
    return ToolProposal(
        format=change_format,
        target=str(target),
        source=str(source),
        diff=diff,
    )


async def run_shell_command(input: BashInput, context: ToolContext) -> ToolOutcome:
    shell = await run_sandboxed_command(
        input.command,
        scopes=context.scopes,
        workspace_id=context.workspace_id,
        conversation_id=context.conversation_id,
    )
    return BashSuccess(
        returncode=shell.returncode,
        stdout=shell.stdout,
        stderr=shell.stderr,
    )


async def run_semantic_search(input: SearchInput, context: ToolContext) -> ToolOutcome:
    from backend.search import semantic_search

    scope_names: list[str] = [scope.internal_name for scope in context.scopes]
    try:
        results = await semantic_search(input.query, input.top_k, scopes=scope_names)
    except Exception as exc:
        return ToolError(message=f'search failed: {exc}')
    return SearchSuccess(results=[SearchHitOutput(**result._asdict()) for result in results])


@dataclass(frozen=True)
class ChangePaths:
    """Where a repository change reads from and where it writes."""

    target: Path
    source: Path
    target_realpath: Path
    source_realpath: Path


async def resolve_change(
    target: Path,
    source: Path,
    context: ToolContext,
    source_label: str,
    *,
    target_must_exist: bool,
) -> ChangePaths | ToolError:
    """The paths a change touches, or the first reason it cannot run.

    Both change tools check the same things; only the word for the source file
    and whether the target must already exist differ between them.
    """
    target_safe, target_err = is_safe_vpath(target, Path(REPOSITORIES_VROOT), allowed_scopes=context.scopes)
    if not target_safe:
        return ToolError(message=target_err)

    target_repo = await get_repo_from_vpath(target)
    target_realpath = await resolve_repo_vpath(target)
    if not target_realpath:
        return ToolError(message='Could not resolve target repository path')

    source_realpath = resolve_space_vpath(source, context.workspace_id)
    if source_realpath is None:
        return ToolError(message=f'{source_label} path escapes the workspace: {source}')

    if target_repo and not is_safe_realpath(target_realpath, Path(target_repo.path)):
        return ToolError(message=f'Target path escapes the repository: {target}')

    if target_must_exist and not target_realpath.exists():
        return ToolError(message=f'Target file not found: {target}')
    if not source_realpath.exists():
        return ToolError(message=f'{source_label} file not found: {source}')

    return ChangePaths(target, source, target_realpath, source_realpath)


def unified_diff(before: str, after: str, before_path: Path, after_path: Path) -> str:
    """A unified diff of two texts, with every line newline-terminated."""
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=str(before_path),
        tofile=str(after_path),
    )
    return ''.join(line if line.endswith('\n') else line + '\n' for line in lines)


async def run_propose_replace(input: ReplaceFileInput, context: ToolContext) -> ToolOutcome:
    change = await resolve_change(
        Path(input.target),
        Path(input.source),
        context,
        'Source',
        target_must_exist=False,
    )
    if isinstance(change, ToolError):
        return change

    source_content = change.source_realpath.read_text(encoding='utf-8')

    if not context.privileged:
        # A target that does not exist yet is replaced by a file with content.
        target_content = change.target_realpath.read_text(encoding='utf-8') if change.target_realpath.exists() else ''
        diff_text = unified_diff(target_content, source_content, change.source, change.target)
        return proposal('replace', change.target, change.source, diff_text)

    try:
        change.target_realpath.parent.mkdir(parents=True, exist_ok=True)
        change.target_realpath.write_text(source_content, encoding='utf-8')
    except Exception as exc:
        return ToolError(message=f'replace_file failed: {exc}')
    return ReplaceFileSuccess()


async def run_propose_diff(input: ApplyPatchInput, context: ToolContext) -> ToolOutcome:
    change = await resolve_change(
        Path(input.target),
        Path(input.diff_path),
        context,
        'Diff',
        target_must_exist=True,
    )
    if isinstance(change, ToolError):
        return change

    diff_content = change.source_realpath.read_text(encoding='utf-8')

    if not context.privileged:
        return proposal('patch', change.target, change.source, diff_content)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            # Without this, `patch` drops `<file>.orig` into the repository
            ['patch', '-p0', '--no-backup-if-mismatch', str(change.target_realpath)],
            input=diff_content,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f'Patch failed: {result.stderr}')
    except Exception as exc:
        return ToolError(message=f'apply_patch failed: {exc}')
    return ApplyPatchSuccess()


async def run_spawn_agent(input: SpawnAgentInput, context: ToolContext) -> ToolOutcome:
    from backend.conversation import (
        create_conversation,
        get_blocking_message_id,
        get_conversation_config,
        get_messages_for_continuation,
        insert_message,
        prompt_conversation,
    )

    symbol = input.agent_symbol
    if context.agent is None:
        return ToolError(message='spawn_agent is only available to configured agents')
    if symbol not in context.agent.spawnable_agents:
        return ToolError(message=f"agent '{symbol}' is not in the allowed list")
    if context.spawn_depth >= MAX_SPAWN_DEPTH:
        return ToolError(message=f'spawn depth limit reached ({MAX_SPAWN_DEPTH})')

    try:
        async with mk_conn() as conn:
            parent = await get_conversation_config(conn, context.conversation_id)
            if parent is None:
                return ToolError(message='parent conversation not found')
            if not parent['provider_key'] or not parent['model_id']:
                return ToolError(message='parent conversation has no model')
            child_id = await create_conversation(
                conn,
                parent['workspace_id'],
                agent_id=symbol,
                provider_key=parent['provider_key'],
                model_id=parent['model_id'],
                inference_config=parent['inference_config'],
            )
            await insert_message(conn, child_id, Prompt(content=input.prompt))
            await conn.commit()

        async for _ in prompt_conversation(child_id, spawn_depth=context.spawn_depth + 1):
            pass

        messages = await get_messages_for_continuation(child_id)
        final = ''
        has_message = False
        # The child started with the prompt above, so any message is its answer.
        for _, element in messages:
            if isinstance(element, Message):
                has_message = True
                final = element.content
        async with mk_conn() as conn:
            blocked = await get_blocking_message_id(conn, child_id)
        if blocked is None and not has_message:
            return ToolError(
                message=(
                    'spawn_agent failed: agent produced no message '
                    f'(conversation {child_id}, agent {symbol})'
                ),
            )
        return SpawnAgentSuccess(
            conversation_id=child_id,
            agent=symbol,
            response=final,
            blocked=blocked is not None,
            blocking_message_id=blocked,
        )
    except Exception as exc:
        return ToolError(message=f'spawn_agent failed: {exc}')


def _tool(
    name: str,
    description: str,
    input_model: type[InputT],
    executor: Callable[[InputT, ToolContext], Awaitable[OutputT]],
) -> Tool[InputT, OutputT]:
    """A registry entry: the input model *is* the schema the model receives."""
    schema = input_model.model_json_schema()
    # The tool description is the one the registry passes; the model's own
    # title and docstring would only repeat it in the prompt.
    schema.pop('title', None)
    schema.pop('description', None)
    return {
        'name': name,
        'schema': {
            'name': name,
            'description': description,
            'parameters': schema,
        },
        'input': input_model,
        'executor': executor,
    }


TOOL_REGISTRY: list[RegisteredTool] = [
    _tool(
        'bash',
        'Execute a shell command in the sandbox and return stdout, stderr and the exit code.',
        BashInput,
        run_shell_command,
    ),
    _tool(
        'search',
        (
            'Search indexed repositories for relevant documents. Use it for '
            'semantic or conceptual questions. Results are limited to the '
            "conversation's resolved repository scopes; an empty scope list "
            'returns no results. Results are ordered by cosine distance, '
            'lowest first.'
        ),
        SearchInput,
        run_semantic_search,
    ),
    _tool(
        'spawn_agent',
        (
            'Start a new conversation with another agent and return its final '
            'response. Use it for a self-contained subtask that should be '
            'handled by a different agent.'
        ),
        SpawnAgentInput,
        run_spawn_agent,
    ),
    _tool(
        'replace_file',
        'Replace a file in the repositories folder with a file from the workspace. Requires user approval.',
        ReplaceFileInput,
        run_propose_replace,
    ),
    _tool(
        'apply_patch',
        'Apply a unified diff from the workspace to a file in the repositories. Requires user approval.',
        ApplyPatchInput,
        run_propose_diff,
    ),
]


def describe_errors(exc: ValidationError) -> str:
    return '; '.join(
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors(include_url=False)
    )


def parse_arguments(
    tool: RegisteredTool,
    arguments: Optional[str | dict[str, Any] | BaseModel],
) -> ToolArguments | ToolError:
    """The tool's validated input, or the error the model should read instead."""
    raw: Any = arguments
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolError(message=f'Invalid JSON arguments: {exc}')
    if not isinstance(raw, dict):
        return ToolError(message='Tool arguments must be a JSON object')
    try:
        validated = tool['input'].model_validate(raw)
    except ValidationError as exc:
        return ToolError(message=f"Invalid arguments for {tool['name']}: {describe_errors(exc)}")
    return cast(ToolArguments, validated)


async def run_tool_call(
    name: str,
    arguments: Optional[str | dict[str, Any] | BaseModel],
    context: ToolContext,
) -> ToolExecution:
    """Run one call. Every failure, including bad arguments and unknown tools,
    comes back as a `ToolError` outcome so the model can see it and retry."""
    tool = next((entry for entry in TOOL_REGISTRY if entry['name'] == name), None)
    if tool is None:
        return ToolExecution(None, ToolError(message=f'Unknown tool: {name}'))

    parsed = parse_arguments(tool, arguments)
    if isinstance(parsed, ToolError):
        return ToolExecution(None, parsed)

    try:
        outcome = await tool['executor'](parsed, context)
    except Exception as exc:
        return ToolExecution(parsed, ToolError(message=f'{name} failed: {exc}'))
    return ToolExecution(parsed, outcome)
