import json
import unittest
from pathlib import Path
from tests.helpers import BaseTestCase
from domain import AgentConfig, ScopeSpec, SecurityPolicy
from tools import (
    MAX_SPAWN_DEPTH,
    ToolContext,
    run_propose_diff,
    run_propose_replace,
    run_shell_command,
    run_spawn_agent,
)


class TestTools(BaseTestCase):
    conversation_id = 1

    @property
    def workspace_dir(self) -> Path:
        path = Path(self._workspace_dir.name) / str(self.conversation_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ctx(self, scopes=None, privileged=False) -> ToolContext:
        return ToolContext(
            conversation_id=self.conversation_id,
            privileged=privileged,
            scopes=scopes or [],
        )

    async def test_spawn_agent_rejects_beyond_depth_limit(self):
        agent = AgentConfig(
            id=1,
            display_name="Parent",
            internal_name="parent",
            description="",
            provider_key="mock_e2e",
            model_id="model-1",
            inference_config={},
            repository_access=[],
            spawnable_agents=["child_agent"],
        )
        result = await run_spawn_agent(
            "spawn_agent",
            json.dumps({"agent_symbol": "child_agent", "prompt": "do work"}),
            ToolContext(
                conversation_id=1,
                agent=agent,
                spawn_depth=MAX_SPAWN_DEPTH,
            ),
        )
        self.assertIn("depth limit", json.loads(result.result)["error"])

    async def test_run_shell_command_basic(self):
        result = await run_shell_command("bash", '{"command": "echo hello"}', context=self.ctx(scopes=[]))
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 0)
        self.assertEqual(parsed["stdout"].strip(), "hello")

    async def test_run_shell_command_stderr(self):
        result = await run_shell_command("bash", '{"command": "echo stderr >&2"}', context=self.ctx(scopes=[]))
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 0)
        self.assertEqual(parsed["stderr"].strip(), "stderr")

    async def test_run_shell_command_nonzero_exit(self):
        result = await run_shell_command("bash", '{"command": "exit 42"}', context=self.ctx(scopes=[]))
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 42)

    async def test_run_shell_command_missing_command(self):
        with self.assertRaises((ValueError, KeyError)):
            await run_shell_command("bash", '{"not_command": "hello"}', context=self.ctx(scopes=[]))

    async def test_run_shell_command_invalid_json(self):
        with self.assertRaises(ValueError):
            await run_shell_command("bash", "not json", context=self.ctx(scopes=[]))

    async def test_run_propose_replace_success(self):
        # Create a repository so the vpath resolver works
        repo_name = "test_replace_success"
        create_res = await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Success",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        self.assertEqual(create_res.status_code, 201)
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)

        # Create a source file in the workspace
        source_rel = "source.txt"
        source_real = self.workspace_dir / source_rel
        source_real.write_text("new content", encoding="utf-8")

        # Target file in the repo
        target_rel = "target.txt"
        target_real = Path(self._repo_dir.name) / target_rel
        target_real.write_text("old content", encoding="utf-8")

        target_vpath = f"/repositories/{repo_name}/{target_rel}"
        source_vpath = f"/workspace/{source_rel}"

        result = await run_propose_replace(
            "replace_file",
            json.dumps({"target": target_vpath, "source": source_vpath}),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["type"], "replace_proposal")
        self.assertEqual(parsed["target"], target_vpath)
        self.assertEqual(parsed["source"], source_vpath)
        self.assertEqual(parsed["status"], "pending")
        self.assertIn("diff", parsed)
        self.assertTrue(result.is_blocking)

    async def test_run_propose_replace_source_not_found(self):
        repo_name = "test_replace_notfound"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Not Found",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)

        result = await run_propose_replace(
            "replace_file",
            json.dumps({
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/nonexistent.txt",
            }),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertIn("Source file not found", parsed["error"])

    async def test_run_propose_replace_resolve_error(self):
        scope = ScopeSpec(internal_name="nonexistent_repo", security_policy=SecurityPolicy.WRITE)
        result = await run_propose_replace(
            "replace_file",
            '{"target": "/repositories/nonexistent_repo/file.txt", "source": "/workspace/source.txt"}',
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["error"], "Could not resolve target repository path")

    async def test_run_propose_replace_missing_source(self):
        repo_name = "test_replace_missing_source"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Missing Source",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)
        (Path(self._repo_dir.name) / "target.txt").write_text("old content", encoding="utf-8")

        result = await run_propose_replace(
            "replace_file",
            json.dumps({
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/source.txt",
            }),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertIn("Source file not found", parsed["error"])

    async def test_run_propose_replace_requires_matching_scope(self):
        repo_name = "test_replace_scope"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Scope",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })

        result = await run_propose_replace(
            "replace_file",
            json.dumps({
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/source.txt",
            }),
            context=self.ctx(scopes=[]),
        )
        parsed = json.loads(result.result)
        self.assertIn("Access denied", parsed["error"])

    async def test_run_propose_replace_rejects_prefix_escape(self):
        result = await run_propose_replace(
            "replace_file",
            json.dumps({
                "target": "/repositories/foobar/target.txt",
                "source": "/workspace/source.txt",
            }),
            context=self.ctx(scopes=[ScopeSpec(internal_name="foo", security_policy=SecurityPolicy.WRITE)]),
        )
        parsed = json.loads(result.result)
        self.assertIn("Access denied", parsed["error"])

    async def test_run_propose_replace_rejects_symlink_escape(self):
        repo_name = "test_replace_symlink"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Replace Symlink",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)

        outside = Path(self._repo_dir.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.workspace_dir / "leak.txt").symlink_to(outside)

        result = await run_propose_replace(
            "replace_file",
            json.dumps({
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/leak.txt",
            }),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertIn("escapes the workspace", parsed["error"])

    async def test_run_propose_diff_success(self):
        repo_name = "test_diff_success"
        create_res = await self.client.post("/api/repositories", json={
            "display_name": "Test Diff Success",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        self.assertEqual(create_res.status_code, 201)
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)

        # Create a target file in the repo
        target_rel = "greeting.txt"
        target_real = Path(self._repo_dir.name) / target_rel
        target_real.write_text("Hello, World!\n", encoding="utf-8")

        # Create a diff file in the workspace
        diff_rel = "patch.diff"
        diff_real = self.workspace_dir / diff_rel
        # A unified diff that changes "Hello, World!" to "Hello, Bob!"
        diff_content = (
            "--- greeting.txt\n"
            "+++ greeting.txt\n"
            "@@ -1 +1 @@\n"
            "-Hello, World!\n"
            "+Hello, Bob!\n"
        )
        diff_real.write_text(diff_content, encoding="utf-8")

        target_vpath = f"/repositories/{repo_name}/{target_rel}"
        diff_vpath = f"/workspace/{diff_rel}"

        result = await run_propose_diff(
            "apply_patch",
            json.dumps({"target": target_vpath, "diff_path": diff_vpath}),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["type"], "diff_proposal")
        self.assertEqual(parsed["target"], target_vpath)
        self.assertEqual(parsed["diff_path"], diff_vpath)
        self.assertEqual(parsed["status"], "pending")
        self.assertIn("diff", parsed)
        self.assertTrue(result.is_blocking)

    async def test_run_propose_diff_target_not_found(self):
        repo_name = "test_diff_notfound"
        await self.client.post("/api/repositories", json={
            "display_name": "Test Diff Not Found",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        scope = ScopeSpec(internal_name=repo_name, security_policy=SecurityPolicy.WRITE)

        diff_rel = "patch.diff"
        diff_real = self.workspace_dir / diff_rel
        diff_real.write_text("dummy diff", encoding="utf-8")

        result = await run_propose_diff(
            "apply_patch",
            json.dumps({
                "target": f"/repositories/{repo_name}/nonexistent.txt",
                "diff_path": f"/workspace/{diff_rel}",
            }),
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertIn("Target file not found", parsed["error"])

    async def test_run_propose_diff_resolve_error(self):
        scope = ScopeSpec(internal_name="test", security_policy=SecurityPolicy.WRITE)
        result = await run_propose_diff(
            "apply_patch",
            '{"target": "/repositories/test/file.txt", "diff_path": "/workspace/patch.diff"}',
            context=self.ctx(scopes=[scope]),
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["error"], "Could not resolve target repository path")


if __name__ == "__main__":
    unittest.main()
