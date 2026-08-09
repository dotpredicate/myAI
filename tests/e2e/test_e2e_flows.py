import json
import unittest
from pathlib import Path

from tests.helpers import BaseTestCase, Yield
from inference.engine import FinishedMessage, FinishedToolCall


class TestE2EFlows(BaseTestCase):

    async def test_reader_flow(self):
        """Create agent, ask it about file content -> response contains file content."""
        f = Path(self._repo_dir.name, "data.txt")
        f.write_text("secret-data-42", encoding="utf-8")

        await self.client.post("/api/repositories", json={
            "display_name": "Reader Repo",
            "internal_name": "reader_repo",
            "path": self._repo_dir.name,
            "security": "write",
        })

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="run_shell_command",
                parameters=json.dumps({"command": "cat /repositories/reader_repo/data.txt"}),
            )),
            Yield(finished=FinishedMessage(content="The file contains: secret-data-42")),
        )

        resp = await self.client.post("/api/conversations/prompt", json={
            "provider_key": "mock_e2e",
            "model_id": "m",
            "prompt": "read data.txt",
            "scopes": [{"internal_name": "reader_repo"}],
        })
        self.assertEqual(resp.status_code, 200)

        conv_id = int(resp.headers["x-conversation-id"])
        conv = (await self.client.get(f"/api/conversations/{conv_id}")).json()

        tool_calls = [m for m in conv["messages"] if m["element"]["type"] == "tool_call"]
        self.assertGreaterEqual(len(tool_calls), 1)
        result = json.loads(tool_calls[-1]["element"]["result"])
        self.assertEqual(result["returncode"], 0)
        self.assertIn("secret-data-42", result["stdout"])

    async def test_editor_flow(self):
        """Create file, agent proposes replace -> proposal has correct content."""
        f = Path(self._repo_dir.name, "hello.txt")
        f.write_text("Hello, World!", encoding="utf-8")

        Path(self._workspace_dir.name, "new.txt").write_text("Hello, Agent!", encoding="utf-8")

        await self.client.post("/api/repositories", json={
            "display_name": "Editor Repo",
            "internal_name": "editor_repo",
            "path": self._repo_dir.name,
            "security": "write",
        })

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="propose_replace",
                parameters=json.dumps({
                    "target": "/repositories/editor_repo/hello.txt",
                    "source": "/workspace/new.txt",
                }),
            )),
        )

        resp = await self.client.post("/api/conversations/prompt", json={
            "provider_key": "mock_e2e",
            "model_id": "m",
            "prompt": "replace hello.txt",
        })
        self.assertEqual(resp.status_code, 200)

        conv_id = int(resp.headers["x-conversation-id"])
        conv = (await self.client.get(f"/api/conversations/{conv_id}")).json()

        tool_calls = [m for m in conv["messages"] if m["element"]["type"] == "tool_call"]
        self.assertGreaterEqual(len(tool_calls), 1)

        result = json.loads(tool_calls[-1]["element"]["result"])
        self.assertEqual(result["type"], "replace_proposal")
        self.assertEqual(result["status"], "pending")
        self.assertIn("Hello, Agent!", result["diff"])

    async def test_discovery_flow(self):
        """Create multiple repos, agent lists them -> response names all repos."""
        repo_names = ["alpha", "beta", "gamma"]
        for name in repo_names:
            p = Path(self._repo_dir.name, name)
            p.mkdir(exist_ok=True)
            await self.client.post("/api/repositories", json={
                "display_name": name.title(),
                "internal_name": name,
                "path": str(p),
                "security": "write",
            })

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="run_shell_command",
                parameters=json.dumps({"command": "ls /repositories/"}),
            )),
            Yield(finished=FinishedMessage(content="Found repos: alpha, beta, gamma")),
        )

        resp = await self.client.post("/api/conversations/prompt", json={
            "provider_key": "mock_e2e",
            "model_id": "m",
            "prompt": "list my repositories",
            "scopes": [{"internal_name": name} for name in repo_names],
        })
        self.assertEqual(resp.status_code, 200)

        conv_id = int(resp.headers["x-conversation-id"])
        conv = (await self.client.get(f"/api/conversations/{conv_id}")).json()

        tool_calls = [m for m in conv["messages"] if m["element"]["type"] == "tool_call"]
        self.assertGreaterEqual(len(tool_calls), 1)
        result = json.loads(tool_calls[-1]["element"]["result"])
        self.assertEqual(result["returncode"], 0)
        for name in repo_names:
            self.assertIn(name, result["stdout"])


if __name__ == "__main__":
    unittest.main()