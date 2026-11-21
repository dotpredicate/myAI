import json
from pathlib import Path
from typing import Any, cast

from backend import paths
from tests.helpers import BaseTestCase
from backend.domain import AgentConfig, ScopeSpec
from backend.tools import (
    MAX_SPAWN_DEPTH,
    TOOL_REGISTRY,
    ToolContext,
    run_tool_call,
)


class TestTools(BaseTestCase):
    conversation_id = 1
    workspace_id = 1

    @property
    def workspace_dir(self) -> Path:
        path = paths.SPACES_ROOT / str(self.workspace_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ctx(self, scopes=None, privileged=False, agent=None) -> ToolContext:
        return ToolContext(
            conversation_id=self.conversation_id,
            workspace_id=self.workspace_id,
            privileged=privileged,
            scopes=scopes or [],
            agent=agent,
        )

    async def call(self, name: str, arguments: Any, **kwargs) -> dict[str, Any]:
        execution = await run_tool_call(name, arguments, self.ctx(**kwargs))
        return execution.outcome.model_dump()

    async def test_schema_is_generated_from_the_input_model(self):
        bash = next(tool for tool in TOOL_REGISTRY if tool['name'] == 'bash')
        parameters = cast(dict[str, Any], bash['schema']['parameters'])
        self.assertEqual(parameters['type'], 'object')
        self.assertEqual(parameters['required'], ['command'])
        self.assertEqual(parameters['properties']['command']['description'], 'Command to execute')

    async def test_spawn_agent_rejects_beyond_depth_limit(self):
        agent = AgentConfig(
            id=1,
            display_name="Parent",
            internal_name="parent",
            description="",
            provider_key="mock_e2e",
            model_id="model-1",
            inference_config={},
            spawnable_agents=["child_agent"],
        )
        outcome = await run_tool_call(
            "spawn_agent",
            json.dumps({"agent_symbol": "child_agent", "prompt": "do work"}),
            ToolContext(
                conversation_id=1,
                workspace_id=1,
                agent=agent,
                spawn_depth=MAX_SPAWN_DEPTH,
            ),
        )
        self.assertEqual(outcome.outcome.type, 'error')
        self.assertIn("depth limit", outcome.outcome.model_dump()['message'])

    async def test_run_shell_command_basic(self):
        outcome = await self.call("bash", '{"command": "echo hello"}')
        self.assertEqual(outcome["type"], "success")
        self.assertEqual(outcome["tool"], "bash")
        self.assertEqual(outcome["returncode"], 0)
        self.assertEqual(outcome["stdout"].strip(), "hello")

    async def test_run_shell_command_stderr(self):
        outcome = await self.call("bash", '{"command": "echo stderr >&2"}')
        self.assertEqual(outcome["returncode"], 0)
        self.assertEqual(outcome["stderr"].strip(), "stderr")

    async def test_run_shell_command_nonzero_exit(self):
        outcome = await self.call("bash", '{"command": "exit 42"}')
        self.assertEqual(outcome["returncode"], 42)

    async def test_run_shell_command_missing_command(self):
        outcome = await self.call("bash", '{"not_command": "hello"}')
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Invalid arguments for bash", outcome["message"])
        self.assertIn("command", outcome["message"])

    async def test_run_shell_command_invalid_json(self):
        outcome = await self.call("bash", "not json")
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Invalid JSON arguments", outcome["message"])

    async def test_unknown_tool_is_an_error_outcome(self):
        outcome = await self.call("does_not_exist", '{}')
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Unknown tool", outcome["message"])

    async def test_run_propose_replace_success(self):
        # Create a repository so the vpath resolver works
        repo_name = "test_replace_success"
        create_res = await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Success",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        self.assertEqual(create_res.status_code, 201)
        scope = ScopeSpec(internal_name=repo_name)

        # Create a source file in the workspace
        source_rel = "source.txt"
        (self.workspace_dir / source_rel).write_text("new content", encoding="utf-8")

        # Target file in the repo
        target_rel = "target.txt"
        (Path(self._repo_dir.name) / target_rel).write_text("old content", encoding="utf-8")

        target_vpath = f"/repositories/{repo_name}/{target_rel}"
        source_vpath = f"/workspace/{source_rel}"

        outcome = await self.call(
            "replace_file",
            {"target": target_vpath, "source": source_vpath},
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "proposal")
        self.assertEqual(outcome["format"], "replace")
        self.assertEqual(outcome["target"], target_vpath)
        self.assertEqual(outcome["source"], source_vpath)
        self.assertIn("+new content", outcome["diff"])

    async def test_run_propose_replace_source_not_found(self):
        repo_name = "test_replace_notfound"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Not Found",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        scope = ScopeSpec(internal_name=repo_name)

        outcome = await self.call(
            "replace_file",
            {
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/nonexistent.txt",
            },
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Source file not found", outcome["message"])

    async def test_run_propose_replace_resolve_error(self):
        scope = ScopeSpec(internal_name="nonexistent_repo")
        outcome = await self.call(
            "replace_file",
            {"target": "/repositories/nonexistent_repo/file.txt", "source": "/workspace/source.txt"},
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertEqual(outcome["message"], "Could not resolve target repository path")

    async def test_run_propose_replace_missing_source(self):
        repo_name = "test_replace_missing_source"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Missing Source",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        scope = ScopeSpec(internal_name=repo_name)
        (Path(self._repo_dir.name) / "target.txt").write_text("old content", encoding="utf-8")

        outcome = await self.call(
            "replace_file",
            {
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/source.txt",
            },
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Source file not found", outcome["message"])

    async def test_run_propose_replace_requires_matching_scope(self):
        repo_name = "test_replace_scope"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Scope",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })

        outcome = await self.call(
            "replace_file",
            {
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/source.txt",
            },
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Access denied", outcome["message"])

    async def test_run_propose_replace_rejects_prefix_escape(self):
        outcome = await self.call(
            "replace_file",
            {"target": "/repositories/foobar/target.txt", "source": "/workspace/source.txt"},
            scopes=[ScopeSpec(internal_name="foo")],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Access denied", outcome["message"])

    async def test_run_propose_replace_rejects_symlink_escape(self):
        repo_name = "test_replace_symlink"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Symlink",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        scope = ScopeSpec(internal_name=repo_name)

        outside = Path(self._repo_dir.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.workspace_dir / "leak.txt").symlink_to(outside)

        outcome = await self.call(
            "replace_file",
            {
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/leak.txt",
            },
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("escapes the workspace", outcome["message"])

    async def test_privileged_replace_writes_the_file(self):
        repo_name = "test_replace_privileged"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Privileged",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        scope = ScopeSpec(internal_name=repo_name)
        (self.workspace_dir / "note.txt").write_text("new", encoding="utf-8")
        target = Path(self._repo_dir.name) / "note.txt"
        target.write_text("old", encoding="utf-8")

        outcome = await self.call(
            "replace_file",
            {"target": f"/repositories/{repo_name}/note.txt", "source": "/workspace/note.txt"},
            scopes=[scope],
            privileged=True,
        )
        self.assertEqual(outcome["type"], "success")
        self.assertEqual(outcome["tool"], "replace_file")
        self.assertEqual(target.read_text(), "new")

    async def test_run_propose_diff_success(self):
        repo_name = "test_diff_success"
        create_res = await self.client.post("/api/repositories", json={
            "display_name": "Test Diff Success",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        self.assertEqual(create_res.status_code, 201)
        scope = ScopeSpec(internal_name=repo_name)

        # Create a target file in the repo
        target_rel = "greeting.txt"
        (Path(self._repo_dir.name) / target_rel).write_text("Hello, World!\n", encoding="utf-8")

        # Create a diff file in the workspace
        diff_rel = "patch.diff"
        diff_content = (
            "--- greeting.txt\n"
            "+++ greeting.txt\n"
            "@@ -1 +1 @@\n"
            "-Hello, World!\n"
            "+Hello, Bob!\n"
        )
        (self.workspace_dir / diff_rel).write_text(diff_content, encoding="utf-8")

        target_vpath = f"/repositories/{repo_name}/{target_rel}"
        diff_vpath = f"/workspace/{diff_rel}"

        outcome = await self.call(
            "apply_patch",
            {"target": target_vpath, "diff_path": diff_vpath},
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "proposal")
        self.assertEqual(outcome["format"], "patch")
        self.assertEqual(outcome["target"], target_vpath)
        self.assertEqual(outcome["source"], diff_vpath)
        self.assertIn("+Hello, Bob!", outcome["diff"])

    async def test_run_propose_diff_target_not_found(self):
        repo_name = "test_diff_notfound"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Diff Not Found",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
        })
        scope = ScopeSpec(internal_name=repo_name)
        (self.workspace_dir / "patch.diff").write_text("dummy diff", encoding="utf-8")

        outcome = await self.call(
            "apply_patch",
            {
                "target": f"/repositories/{repo_name}/nonexistent.txt",
                "diff_path": "/workspace/patch.diff",
            },
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertIn("Target file not found", outcome["message"])

    async def test_run_propose_diff_resolve_error(self):
        scope = ScopeSpec(internal_name="test")
        outcome = await self.call(
            "apply_patch",
            {"target": "/repositories/test/file.txt", "diff_path": "/workspace/patch.diff"},
            scopes=[scope],
        )
        self.assertEqual(outcome["type"], "error")
        self.assertEqual(outcome["message"], "Could not resolve target repository path")
