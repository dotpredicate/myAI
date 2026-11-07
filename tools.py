from dataclasses import dataclass
import asyncio
import json
import subprocess
import difflib
from pathlib import Path
from typing import Awaitable, Callable, TypedDict

from system import REPOSITORIES_VROOT, get_repo_from_vpath, is_safe_realpath, is_safe_vpath, resolve_repo_vpath, resolve_workspace_vpath, run_sandboxed_command
from domain import SecurityPolicy, ScopeSpec

class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]

@dataclass(frozen=True)
class ToolCallResult:
    result: str
    is_blocking: bool = False

class Tool(TypedDict):
    name: str
    schema: FunctionDefinition
    executor: Callable[[str, str, int, bool, list[ScopeSpec]], Awaitable[ToolCallResult]]


async def run_shell_command(name: str, parameters: str, conversation_id: int, privileged: bool = False, scopes: list[ScopeSpec] = []) -> ToolCallResult:
    params = json.loads(parameters)
    command: str = params['command']
    if not isinstance(command, str):
        raise ValueError('Cannot parse the command field. Double-check if input is valid JSON.')

    shell = await run_sandboxed_command(command, scopes=scopes, conversation_id=conversation_id)

    output_str = json.dumps({'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return ToolCallResult(output_str)

async def run_semantic_search(
    name: str,
    parameters: str,
    conversation_id: int,
    privileged: bool = False,
    scopes: list[ScopeSpec] = [],
) -> ToolCallResult:
    from search import semantic_search
    params = json.loads(parameters)
    query: str = params["query"]
    top_k: int = int(params.get("top_k", 5))

    scope_names: list[str] = [s.internal_name for s in scopes]
    try:
        results = await semantic_search(
            query,
            top_k,
            scopes=scope_names,
        )
    except Exception as exc:
        return ToolCallResult(json.dumps({"error": f"search failed: {exc}"}))
    return ToolCallResult(result=json.dumps({"results": [result._asdict() for result in results]}))

async def run_propose_replace(name: str, parameters: str, conversation_id: int, privileged: bool = False, scopes: list[ScopeSpec] = []) -> ToolCallResult:
    try:
        params = json.loads(parameters)
        target_vpath_str = params.get("target")
        source_vpath_str = params.get("source")
        
        if not target_vpath_str or not source_vpath_str:
            raise ValueError("Missing target or source path")
        
        target_vpath = Path(target_vpath_str)
        source_vpath = Path(source_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=scopes)
        if not target_safe:
            return ToolCallResult(json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = await get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == SecurityPolicy.READ_ONLY:
            return ToolCallResult(json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        target_realpath = await resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return ToolCallResult(json.dumps({"error": "Could not resolve target repository path"}))
        source_realpath = resolve_workspace_vpath(source_vpath, conversation_id)
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
        
        if not privileged:
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

async def run_propose_diff(name: str, parameters: str, conversation_id: int, privileged: bool = False, scopes: list[ScopeSpec] = []) -> ToolCallResult:
    try:
        params = json.loads(parameters)
        target_vpath_str = params.get("target")
        diff_vpath_str = params.get("diff_path")
        
        if not target_vpath_str or not diff_vpath_str:
            raise ValueError("Missing target or diff_path")
        
        target_vpath = Path(target_vpath_str)
        diff_vpath = Path(diff_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=scopes)
        if not target_safe:
            return ToolCallResult(json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = await get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == SecurityPolicy.READ_ONLY:
            return ToolCallResult(json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        target_realpath = await resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return ToolCallResult(json.dumps({"error": "Could not resolve target repository path"}))
        diff_realpath = resolve_workspace_vpath(diff_vpath, conversation_id)
        if diff_realpath is None:
            return ToolCallResult(json.dumps({"error": f"Diff path escapes the workspace: {diff_vpath}"}))

        if target_repo and not is_safe_realpath(target_realpath, Path(target_repo.path)):
            return ToolCallResult(json.dumps({"error": f"Target path escapes the repository: {target_vpath}"}))
        
        if not target_realpath.exists():
            return ToolCallResult(json.dumps({"error": f"Target file not found: {target_vpath}"}))
        
        if not diff_realpath.exists():
            return ToolCallResult(json.dumps({"error": f"Diff file not found: {diff_vpath}"}))
        
        diff_content = diff_realpath.read_text(encoding='utf-8')
        
        if not privileged:
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
                "semantic or conceptual questions; for exact or exhaustive literal "
                "matches use bash grep instead. Results are limited to the "
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

async def run_tool_call(name: str, parameters: str, privileged: bool, scopes: list[ScopeSpec], conversation_id: int) -> ToolCallResult:
    for entry in TOOL_REGISTRY:
        if entry["name"] == name:
            try:
                return await entry["executor"](name, parameters, conversation_id, privileged, scopes)
            except Exception as exc:
                return ToolCallResult(json.dumps({"error": f"Uncaught error: {exc}"}))
    
    raise ValueError(f'Unsupported tool call {name}')
