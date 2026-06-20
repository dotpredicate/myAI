import json
import unittest
from pathlib import Path
from tests.helpers import BaseTestCase
from tools import run_shell_command, run_propose_replace, run_propose_diff


class TestTools(BaseTestCase):
    async def test_run_shell_command_basic(self):
        result = await run_shell_command("run_shell_command", '{"command": "echo hello"}', scopes=[])
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 0)
        self.assertEqual(parsed["stdout"].strip(), "hello")

    async def test_run_shell_command_stderr(self):
        result = await run_shell_command("run_shell_command", '{"command": "echo stderr >&2"}', scopes=[])
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 0)
        self.assertEqual(parsed["stderr"].strip(), "stderr")

    async def test_run_shell_command_nonzero_exit(self):
        result = await run_shell_command("run_shell_command", '{"command": "exit 42"}', scopes=[])
        parsed = json.loads(result.result)
        self.assertEqual(parsed["returncode"], 42)

    async def test_run_shell_command_missing_command(self):
        with self.assertRaises((ValueError, KeyError)):
            await run_shell_command("run_shell_command", '{"not_command": "hello"}', scopes=[])

    async def test_run_shell_command_invalid_json(self):
        with self.assertRaises(ValueError):
            await run_shell_command("run_shell_command", "not json", scopes=[])

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

        # Create a source file in the workspace
        source_rel = "source.txt"
        source_real = Path(self._workspace_dir.name) / source_rel
        source_real.write_text("new content", encoding="utf-8")

        # Target file in the repo
        target_rel = "target.txt"
        target_real = Path(self._repo_dir.name) / target_rel
        target_real.write_text("old content", encoding="utf-8")

        target_vpath = f"/repositories/{repo_name}/{target_rel}"
        source_vpath = f"/workspace/{source_rel}"

        result = await run_propose_replace(
            "propose_replace",
            json.dumps({"target": target_vpath, "source": source_vpath}),
            scopes=[],
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

        result = await run_propose_replace(
            "propose_replace",
            json.dumps({
                "target": f"/repositories/{repo_name}/target.txt",
                "source": "/workspace/nonexistent.txt",
            }),
            scopes=[],
        )
        parsed = json.loads(result.result)
        self.assertIn("Source file not found", parsed["error"])

    async def test_run_propose_replace_resolve_error(self):
        # No repo exists -> target resolve fails
        result = await run_propose_replace(
            "propose_replace",
            '{"target": "/repositories/nonexistent_repo/file.txt", "source": "/workspace/source.txt"}',
            scopes=[],
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["error"], "Could not resolve target repository path")

    async def test_run_propose_replace_missing_source(self):
        result = await run_propose_replace(
            "propose_replace",
            '{"target": "/repositories", "source": "/workspace/source.txt"}',
            scopes=[],
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["error"], "Could not resolve target repository path")

    async def test_run_propose_diff_success(self):
        repo_name = "test_diff_success"
        create_res = await self.client.post("/api/repositories", json={
            "display_name": "Test Diff Success",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })
        self.assertEqual(create_res.status_code, 201)

        # Create a target file in the repo
        target_rel = "greeting.txt"
        target_real = Path(self._repo_dir.name) / target_rel
        target_real.write_text("Hello, World!\n", encoding="utf-8")

        # Create a diff file in the workspace
        diff_rel = "patch.diff"
        diff_real = Path(self._workspace_dir.name) / diff_rel
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
            "propose_diff",
            json.dumps({"target": target_vpath, "diff_path": diff_vpath}),
            scopes=[],
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

        diff_rel = "patch.diff"
        diff_real = Path(self._workspace_dir.name) / diff_rel
        diff_real.write_text("dummy diff", encoding="utf-8")

        result = await run_propose_diff(
            "propose_diff",
            json.dumps({
                "target": f"/repositories/{repo_name}/nonexistent.txt",
                "diff_path": f"/workspace/{diff_rel}",
            }),
            scopes=[],
        )
        parsed = json.loads(result.result)
        self.assertIn("Target file not found", parsed["error"])

    async def test_run_propose_diff_resolve_error(self):
        result = await run_propose_diff(
            "propose_diff",
            '{"target": "/repositories/test/file.txt", "diff_path": "/workspace/patch.diff"}',
            scopes=[],
        )
        parsed = json.loads(result.result)
        self.assertEqual(parsed["error"], "Could not resolve target repository path")


if __name__ == "__main__":
    unittest.main()