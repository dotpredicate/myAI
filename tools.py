import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, NamedTuple, Optional, List
from conversation import ToolCall, ToolCallResult
import system
import index

def run_shell_command(tool_call: ToolCall, privileged: bool = False, scopes: Optional[List[str]] = None) -> ToolCallResult:
    params = json.loads(tool_call.parameters)
    command: str = params['command']
    if not isinstance(command, str):
        raise ValueError('Cannot parse the command field. Double-check if input is valid JSON.')

    shell = system.run_sandboxed_command(command, scopes=scopes)

    output_str = json.dumps({'returncode': shell.returncode, 'stdout': shell.stdout, 'stderr': shell.stderr})
    return ToolCallResult(tool_call.name, tool_call.parameters, output_str)

def run_semantic_search(tool_call: ToolCall, privileged: bool = False, scopes: Optional[List[str]] = None) -> ToolCallResult:
    try:
        params = json.loads(tool_call.parameters)
        prompt: str = params["prompt"]
        top_k: int = int(params.get("top_k", 5))
    except Exception as exc:
        return ToolCallResult(
            name=tool_call.name,
            parameters=tool_call.parameters,
            result=json.dumps({"error": f"bad parameters: {exc}"})
        )

    try:
        results = index.semantic_search(prompt, top_k, scopes=scopes)
    except Exception as exc:
        return ToolCallResult(
            name=tool_call.name,
            parameters=tool_call.parameters,
            result=json.dumps({"error": f"search failed: {exc}"})
        )
    return ToolCallResult(
        name=tool_call.name,
        parameters=tool_call.parameters,
        result=json.dumps({"results": json.dumps(results)})
    )

def run_propose_replace(tool_call: ToolCall, privileged: bool = False, scopes: Optional[List[str]] = None) -> ToolCallResult:
    from system import is_safe_vpath, vpath_to_realpath, REPOSITORIES_VROOT, REPOSITORIES_DIR, WORKSPACE_VROOT, WORKSPACE_DIR
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
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        source_safe, source_err = is_safe_vpath(source_vpath, Path(WORKSPACE_VROOT))
        if not source_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": source_err}))
        
        target_realpath = vpath_to_realpath(target_vpath, REPOSITORIES_VROOT, REPOSITORIES_DIR)
        source_realpath = vpath_to_realpath(source_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not source_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Source file not found: {source_vpath}"}))
        
        if target_realpath.exists():
            target_content = target_realpath.read_text(encoding='utf-8')
        else:
            target_content = ""
        
        source_content = source_realpath.read_text(encoding='utf-8')
        
        if not privileged:
            proposal = {
                "type": "replace_proposal",
                "target": str(target_vpath),
                "source": str(source_vpath),
                "preview": f"Replace {target_vpath} with content from {source_vpath}",
                "before": target_content,
                "after": source_content,
                "status": "pending"
            }
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal), is_blocking=True)
        # privileged: perform actual replace
        try:
            target_realpath.parent.mkdir(parents=True, exist_ok=True)
            target_realpath.write_text(source_content, encoding='utf-8')
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Unexpected error: {str(e)}"}))

def run_propose_diff(tool_call: ToolCall, privileged: bool = False, scopes: Optional[List[str]] = None) -> ToolCallResult:
    from system import is_safe_vpath, vpath_to_realpath, REPOSITORIES_VROOT, REPOSITORIES_DIR, WORKSPACE_VROOT, WORKSPACE_DIR
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
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": target_err}))
        
        diff_safe, diff_err = is_safe_vpath(diff_vpath, Path(WORKSPACE_VROOT))
        if not diff_safe:
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": diff_err}))
        
        target_realpath = vpath_to_realpath(target_vpath, REPOSITORIES_VROOT, REPOSITORIES_DIR)
        diff_realpath = vpath_to_realpath(diff_vpath, WORKSPACE_VROOT, WORKSPACE_DIR)
        
        if not target_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
                                 json.dumps({"error": f"Target file not found: {target_vpath}"}))
        
        if not diff_realpath.exists():
            return ToolCallResult(tool_call.name, tool_call.parameters, 
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
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps(proposal), is_blocking=True)
        # privileged: apply patch
        try:
            result = subprocess.run(['patch', '-p0', str(target_realpath)], input=diff_content, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Patch failed: {result.stderr}")
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "applied", "error": None}))
        except Exception as e:
            return ToolCallResult(tool_call.name, tool_call.parameters, json.dumps({"status": "failed", "error": str(e)}))
    except json.JSONDecodeError as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Invalid JSON: {str(e)}"}))
    except Exception as e:
        return ToolCallResult(tool_call.name, tool_call.parameters, 
                             json.dumps({"error": f"Unexpected error: {str(e)}"}))

TOOL_REGISTRY = [
    {
        "name": "run_shell_command",
        "schema": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return the output.",
            "type": "function",
            "function": {
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
            "type": "function",
            "function": {
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
        },
        "executor": run_semantic_search
    },
    {
        "name": "propose_replace",
        "schema": {
            "name": "propose_replace",
            "description": "Propose replacing a target file in the repositories folder with a source file from the workspace.",
            "type": "function",
            "function": {
                "name": "propose_replace",
                "description": "Replace a file in repositories with a file from workspace. Provide absolute paths: target='/repositories/foo/bar.txt', source='/workspace/baz/qux.txt'. Do not use '..'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                        "source": {"type": "string", "description": "Absolute path to source file in workspace folder (e.g., '/workspace/baz/qux.txt')"},
                    },
                    "required": ["target", "source"],
                },
            },
        },
        "executor": run_propose_replace
    },
    {
        "name": "propose_diff",
        "schema": {
            "name": "propose_diff",
            "description": "Propose applying a diff file from the workspace to a target file in the repositories.",
            "type": "function",
            "function": {
                "name": "propose_diff",
                "description": "Apply a diff from workspace to a target file in repositories. Provide absolute paths: target='/repositories/foo/bar.txt', diff_path='/workspace/patch.diff'. Do not use '..'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Absolute path to target file in repositories folder (e.g., '/repositories/foo/bar.txt')"},
                        "diff_path": {"type": "string", "description": "Absolute path to diff file in workspace folder (e.g., '/workspace/patch.diff')"},
                    },
                    "required": ["target", "diff_path"],
                },
            },
        },
        "executor": run_propose_diff
    }
]

def run_tool_call(call: ToolCall, privileged: bool = False, scopes: Optional[List[str]] = None) -> ToolCallResult:
    for entry in TOOL_REGISTRY:
        if entry["name"] == call.name:
            return entry["executor"](call, privileged, scopes)
    
    raise ValueError(f'Unsupported tool call {call.name}')
