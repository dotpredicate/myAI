import json
import subprocess
import difflib
from pathlib import Path
from typing import Optional
from inference.engine import FinishedToolCall, FinishedToolCallResult, Tool
import system
import index
from repositories import get_repo_from_vpath, resolve_repo_vpath

async def run_shell_command(tool_call: FinishedToolCall, privileged: bool = False, scopes: Optional[list[str]] = None) -> FinishedToolCallResult:
    params = json.loads(tool_call.parameters)
    command: str = params['command']
    if not isinstance(command, str):
        raise ValueError('Cannot parse the command field. Double-check if input is valid JSON.')

    shell = system.run_sandboxed_command(command, scopes=scopes)

    output_str = json.dumps({'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return FinishedToolCallResult(tool_call.name, tool_call.parameters, output_str)

async def run_semantic_search(tool_call: FinishedToolCall, privileged: bool = False, scopes: Optional[list[str]] = None) -> FinishedToolCallResult:
    params = json.loads(tool_call.parameters)
    prompt: str = params["prompt"]
    top_k: int = int(params.get("top_k", 5))

    try:
        results = await index.semantic_search(prompt, top_k, scopes=scopes)
    except Exception as exc:
        return FinishedToolCallResult(
            name=tool_call.name,
            parameters=tool_call.parameters,
            result=json.dumps({"error": f"search failed: {exc}"})
        )
    return FinishedToolCallResult(
        name=tool_call.name,
        parameters=tool_call.parameters,
        result=json.dumps({"results": json.dumps(results)})
    )

async def run_propose_replace(tool_call: FinishedToolCall, privileged: bool = False, scopes: Optional[list[str]] = None) -> FinishedToolCallResult:
    from system import is_safe_vpath, vpath_to_realpath, REPOSITORIES_VROOT, WORKSPACE_VROOT, WORKSPACE_DIR
    try:
        params = json.loads(tool_call.parameters)
        target_vpath_str = params.get("target")
        source_vpath_str = params.get("source")
        
        if not target_vpath_str or not source_vpath_str:
            raise ValueError("Missing target or source path")
        
        target_vpath = Path(target_vpath_str)
        source_vpath = Path(source_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=scopes)
        if not target_safe:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == 'read-only':
            return FinishedToolCallResult(tool_call.name, tool_call.parameters,
                                 json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        source_safe, source_err = is_safe_vpath(source_vpath, Path(WORKSPACE_VROOT))
        if not source_safe:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": source_err}))
        
        target_realpath = resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters,
                                 json.dumps({"error": "Could not resolve target repository path"}))
        source_realpath = vpath_to_realpath(source_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not source_realpath.exists():
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Source file not found: {source_vpath}"}))
        
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
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal), is_blocking=True)
        
        # privileged: perform actual replace
        try:
            target_realpath.parent.mkdir(parents=True, exist_ok=True)
            target_realpath.write_text(source_content, encoding='utf-8')
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Unexpected error: {str(e)}"}))

async def run_propose_diff(tool_call: FinishedToolCall, privileged: bool = False, scopes: Optional[list[str]] = None) -> FinishedToolCallResult:
    from system import is_safe_vpath, vpath_to_realpath, REPOSITORIES_VROOT, WORKSPACE_VROOT, WORKSPACE_DIR
    try:
        params = json.loads(tool_call.parameters)
        target_vpath_str = params.get("target")
        diff_vpath_str = params.get("diff_path")
        
        if not target_vpath_str or not diff_vpath_str:
            raise ValueError("Missing target or diff_path")
        
        target_vpath = Path(target_vpath_str)
        diff_vpath = Path(diff_vpath_str)
        
        target_safe, target_err = is_safe_vpath(target_vpath, Path(REPOSITORIES_VROOT), allowed_scopes=scopes)
        if not target_safe:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        # Security policy check: read-only repos cannot be written to at all
        target_repo = get_repo_from_vpath(target_vpath)
        if target_repo and target_repo.security_policy == 'read-only':
            return FinishedToolCallResult(tool_call.name, tool_call.parameters,
                                 json.dumps({"error": f"Repository '{target_repo.internal_name}' is read-only, writes are not permitted"}))

        diff_safe, diff_err = is_safe_vpath(diff_vpath, Path(WORKSPACE_VROOT))
        if not diff_safe:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": diff_err}))
        
        target_realpath = resolve_repo_vpath(target_vpath)
        if not target_realpath:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters,
                                 json.dumps({"error": "Could not resolve target repository path"}))
        diff_realpath = vpath_to_realpath(diff_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not target_realpath.exists():
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Target file not found: {target_vpath}"}))
        
        if not diff_realpath.exists():
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Diff file not found: {diff_vpath}"}))
        
        diff_content = diff_realpath.read_text(encoding='utf-8')
        
        if not privileged:
            proposal = {
                "type": "diff_proposal",
                "diff_path": str(diff_vpath),
                "target": str(target_vpath),
                "diff": diff_content,
                "status": "pending"
            }
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal), is_blocking=True)
        
        # privileged: apply patch
        try:
            result = subprocess.run(['patch', '-p0', str(target_realpath)], input=diff_content, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Patch failed: {result.stderr}")
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return FinishedToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return FinishedToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": str(e)}))



TOOL_REGISTRY: list[Tool] = [
    {
        "name": "run_shell_command",
        "schema": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return the output.",
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
        "name": "run_semantic_search",
        "schema": {
            "name": "run_semantic_search",
            "description": (
                "Search the repositories for the most relevant documents "
                "based on a natural-language prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Natural-language query to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of documents to return",
                    },
                },
                "required": ["prompt"],
            },
        },
        "executor": run_semantic_search
    },
    {
        "name": "propose_replace",
        "schema": {
            "name": "propose_replace",
            "description": "Propose replacing a target file in the repositories folder with a source file from the workspace.",
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
        "name": "propose_diff",
        "schema": {
            "name": "propose_diff",
            "description": "Propose applying a diff file from the workspace to a target file in the repositories.",
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

async def run_tool_call(call: FinishedToolCall, privileged: bool = False, scopes: list[str] = []) -> FinishedToolCallResult:
    for entry in TOOL_REGISTRY:
        if entry["name"] == call.name:
            try:
                return await entry["executor"](call, privileged, scopes)
            except Exception as exc:
                return FinishedToolCallResult(
                    name=call.name,
                    parameters=call.parameters,
                    result=json.dumps({"error": f"Uncaught error: {exc}"})
                )
    
    raise ValueError(f'Unsupported tool call {call.name}')