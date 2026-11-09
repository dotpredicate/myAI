from dataclasses import dataclass, field
import asyncio
import json
import subprocess
import difflib
from pathlib import Path
from typing import Awaitable, Callable, TypedDict

from system import REPOSITORIES_VROOT, get_repo_from_vpath, is_safe_realpath, is_safe_vpath, resolve_repo_vpath, resolve_workspace_vpath, run_sandboxed_command
from domain import AgentConfig, Message, SecurityPolicy, ScopeSpec
from database import mk_conn

class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]

@dataclass(frozen=True)
class ToolCallResult:
    result: str
    is_blocking: bool = False

@dataclass(frozen=True)
class ToolContext:
    conversation_id: int
    privileged: bool = False
    scopes: list[ScopeSpec] = field(default_factory=list)
    agent: AgentConfig | None = None
    spawn_depth: int = 0


class Tool(TypedDict):
    name: str
    schema: FunctionDefinition
    executor: Callable[[str, str, ToolContext], Awaitable[ToolCallResult]]


MAX_SPAWN_DEPTH = 3


async def run_shell_command(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    params = json.loads(parameters)
    command: str = params['command']
    if not isinstance(command, str):
        raise ValueError('Cannot parse the command field. Double-check if input is valid JSON.')

    shell = await run_sandboxed_command(command, scopes=context.scopes, conversation_id=context.conversation_id)

    output_str = json.dumps({'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return ToolCallResult(output_str)

async def run_semantic_search(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    from search import semantic_search
    params = json.loads(parameters)
    query: str = params["query"]
    top_k: int = int(params.get("top_k", 5))

    scope_names: list[str] = [s.internal_name for s in context.scopes]
    try:
        results = await semantic_search(
            query,
            top_k,
            scopes=scope_names,
        )
    except Exception as exc:
        return ToolCallResult(json.dumps({"error": f"search failed: {exc}"}))
    return ToolCallResult(result=json.dumps({"results": [result._asdict() for result in results]}))

async def run_propose_replace(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    try:
        params = json.loads(parameters)
        target_vpath_str = params.get("target")
        source_vpath_str = params.get("source")
        
        if not target_vpath_str or not source_vpath_str:
            raise ValueError("Missing target or source path")
        
        target_vpath = Path(target_vpath_str)
        source_vpath = Path(source_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=context.scopes)
        if not target_safe:
            return ToolCallResult(json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = await get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == SecurityPolicy.READ_ONLY:
            return ToolCallResult(json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        target_realpath = await resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return ToolCallResult(json.dumps({"error": "Could not resolve target repository path"}))
        source_realpath = resolve_workspace_vpath(source_vpath, context.conversation_id)
        if source_realpath is None:
            return ToolCallResult(json.dumps({"error": f"Source path escapes the workspace: {source_vpath}"}))

        if target_repo and not is_safe_realpath(target_realpath, Path(target_repo.path)):
            return ToolCallResult(json.dumps({"error": f"Target path escapes the repository: {target_vpath}"}))
        
        if not source_realpath.exists():
            return ToolCallResult(json.dumps({"error": f"Source file not found: {source_vpath}"}))
        
        if target_realpath.exists():
            target_content = target_realpath.read_text(encoding='utf-8')
        else:
            target_content = ""
        
        source_content = source_realpath.read_text(encoding='utf-8')
        
        if not context.privileged:
            target_lines = target_content.splitlines()
            source_lines = source_content.splitlines()
            diff_lines = difflib.unified_diff(
                target_lines,
                source_lines,
                fromfile=str(source_vpath),
                tofile=str(target_vpath),
            )
            diff_text = ''.join(line if line.endswith('\n') else line + '\n' for line in diff_lines)
            
            proposal = {
                "type": "replace_proposal",
                "target": str(target_vpath),
                "source": str(source_vpath),
                "diff": diff_text,
                "status": "pending"
            }
            return ToolCallResult(json.dumps(proposal), is_blocking=True)
        
        # privileged: perform actual replace
        try:
            target_realpath.parent.mkdir(parents=True, exist_ok=True)
            target_realpath.write_text(source_content, encoding='utf-8')
            return ToolCallResult(json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return ToolCallResult(json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return ToolCallResult(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(json.dumps({"error": f"Unexpected error: {str(e)}"}))

async def run_propose_diff(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    try:
        params = json.loads(parameters)
        target_vpath_str = params.get("target")
        diff_vpath_str = params.get("diff_path")
        
        if not target_vpath_str or not diff_vpath_str:
            raise ValueError("Missing target or diff_path")
        
        target_vpath = Path(target_vpath_str)
        diff_vpath = Path(diff_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=context.scopes)
        if not target_safe:
            return ToolCallResult(json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = await get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == SecurityPolicy.READ_ONLY:
            return ToolCallResult(json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        target_realpath = await resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return ToolCallResult(json.dumps({"error": "Could not resolve target repository path"}))
        diff_realpath = resolve_workspace_vpath(diff_vpath, context.conversation_id)
        if diff_realpath is None:
            return ToolCallResult(json.dumps({"error": f"Diff path escapes the workspace: {diff_vpath}"}))

        if target_repo and not is_safe_realpath(target_realpath, Path(target_repo.path)):
            return ToolCallResult(json.dumps({"error": f"Target path escapes the repository: {target_vpath}"}))
        
        if not target_realpath.exists():
            return ToolCallResult(json.dumps({"error": f"Target file not found: {target_vpath}"}))
        
        if not diff_realpath.exists():
            return ToolCallResult(json.dumps({"error": f"Diff file not found: {diff_vpath}"}))
        
        diff_content = diff_realpath.read_text(encoding='utf-8')
        
        if not context.privileged:
            proposal = {
                "type": "diff_proposal",
                "diff_path": str(diff_vpath),
                "target": str(target_vpath),
                "diff": diff_content,
                "status": "pending"
            }
            return ToolCallResult(json.dumps(proposal), is_blocking=True)
        
        # privileged: apply patch
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ['patch', '-p0', str(target_realpath)],
                input=diff_content,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Patch failed: {result.stderr}")
            return ToolCallResult(json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return ToolCallResult(json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return ToolCallResult(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(json.dumps({"error": str(e)}))



async def run_spawn_agent(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    try:
        params = json.loads(parameters)
        symbol = params.get("agent_symbol")
        prompt = params.get("prompt")
        if not isinstance(symbol, str) or not symbol.strip():
            return ToolCallResult(json.dumps({"error": "agent_symbol is required"}))
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolCallResult(json.dumps({"error": "prompt is required"}))
        if context.agent is None:
            return ToolCallResult(json.dumps({"error": "spawn_agent is only available to configured agents"}))
        symbol = symbol.strip()
        if symbol not in context.agent.spawnable_agents:
            return ToolCallResult(json.dumps({"error": f"agent '{symbol}' is not in the allowed list"}))
        if context.spawn_depth >= MAX_SPAWN_DEPTH:
            return ToolCallResult(json.dumps({"error": f"spawn depth limit reached ({MAX_SPAWN_DEPTH})"}))

        from conversation import (
            create_conversation,
            get_blocking_message_id,
            get_conversation_project,
            get_messages_for_continuation,
            insert_message,
            prompt_conversation,
        )

        async with mk_conn() as conn:
            project_id, _ = await get_conversation_project(conn, context.conversation_id)
            child_id = await create_conversation(conn, project_id)
            await insert_message(conn, child_id, 'user', Message(author='user', content=prompt))
            await conn.commit()

        async for _ in prompt_conversation(
            child_id,
            agent_id=symbol,
            resolved_scopes=context.scopes,
            spawn_depth=context.spawn_depth + 1,
        ):
            pass

        messages = await get_messages_for_continuation(child_id)
        final = ''
        has_message = False
        for _, element in messages:
            if isinstance(element, Message) and element.author == 'assistant':
                has_message = True
                final = element.content
        async with mk_conn() as conn:
            blocked = await get_blocking_message_id(conn, child_id)
        result: dict[str, object] = {
            "conversation_id": child_id,
            "agent": symbol,
            "response": final,
        }
        if blocked is not None:
            result["blocked"] = True
            result["blocking_message_id"] = blocked
        elif not has_message:
            return ToolCallResult(json.dumps({
                "error": "spawn_agent failed: agent produced no message",
                "conversation_id": child_id,
                "agent": symbol,
            }))
        return ToolCallResult(json.dumps(result))
    except json.JSONDecodeError as exc:
        return ToolCallResult(json.dumps({"error": f"Invalid JSON: {exc}"}))
    except Exception as exc:
        return ToolCallResult(json.dumps({"error": f"spawn_agent failed: {exc}"}))


TOOL_REGISTRY: list[Tool] = [
    {
        "name": "bash",
        "schema": {
            "name": "bash",
            "description": "Execute a shell command in the sandbox and return stdout, stderr and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                },
                "required": ["command"],
            },
        },
        "executor": run_shell_command
    },
    {
        "name": "search",
        "schema": {
            "name": "search",
            "description": (
                "Search indexed repositories for relevant documents. Use it for "
                "semantic or conceptual questions. Results are limited to the "
                "conversation's resolved repository scopes; an empty scope list "
                "returns no results. Results are ordered by cosine distance, "
                "lowest first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Query or keywords to search for. Exact identifiers are "
                            "matched semantically, not literally."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of documents to return",
                    },
                },
                "required": ["query"],
            },
        },
        "executor": run_semantic_search
    },
    {
        "name": "spawn_agent",
        "schema": {
            "name": "spawn_agent",
            "description": (
                "Start a new conversation with another agent and return its final "
                "response. Use it for a self-contained subtask that should be "
                "handled by a different agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_symbol": {
                        "type": "string",
                        "description": "Internal name of the agent to spawn.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Self-contained task for the spawned agent.",
                    },
                },
                "required": ["agent_symbol", "prompt"],
            },
        },
        "executor": run_spawn_agent,
    },
    {
        "name": "replace_file",
        "schema": {
            "name": "replace_file",
            "description": "Replace a file in the repositories folder with a file from the workspace. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                    "source": {"type": "string", "description": "Absolute path to source file in workspace folder (e.g., '/workspace/baz/qux.txt')"},
                },
                "required": ["target", "source"],
            },
        },
        "executor": run_propose_replace
    },
    {
        "name": "apply_patch",
        "schema": {
            "name": "apply_patch",
            "description": "Apply a unified diff from the workspace to a file in the repositories. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                    "diff_path": {"type": "string", "description": "Absolute path to diff file in workspace folder (e.g., '/workspace/patch.diff')"},
                },
                "required": ["target", "diff_path"],
            },
        },
        "executor": run_propose_diff
    }
]

async def run_tool_call(name: str, parameters: str, context: ToolContext) -> ToolCallResult:
    for entry in TOOL_REGISTRY:
        if entry["name"] == name:
            try:
                return await entry["executor"](name, parameters, context)
            except Exception as exc:
                return ToolCallResult(json.dumps({"error": f"Uncaught error: {exc}"}))

    raise ValueError(f'Unsupported tool call {name}')
