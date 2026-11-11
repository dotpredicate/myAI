import json
from pathlib import Path

from tests.helpers import BaseTestCase, Yield
from inference.engine import FinishedMessage, FinishedToolCall


class TestE2EFlows(BaseTestCase):

    async def test_reader_flow(self):
        """Create agent, ask it about file content -> response contains file content."""
        f = Path(self._repo_dir.name, "data.txt")
        f.write_text("secret-data-42", encoding="utf-8")

        source = await self._helper_repo_source()
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="bash",
                parameters=json.dumps({"command": f"cat /repositories/{source}/data.txt"}),
            )),
            Yield(finished=FinishedMessage(content="The file contains: secret-data-42")),
        )

        resp = await self._helper_prompt("read data.txt", model_id="m")
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

        source = await self._helper_repo_source()
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="bash",
                parameters=json.dumps({"command": "printf 'Hello, Agent!' > /workspace/new.txt"}),
            )),
            Yield(finished=FinishedToolCall(
                name="replace_file",
                parameters=json.dumps({
                    "target": f"/repositories/{source}/hello.txt",
                    "source": "/workspace/new.txt",
                }),
            )),
        )

        resp = await self._helper_prompt("replace hello.txt", model_id="m")
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
        repo_ids = []
        for name in repo_names:
            p = Path(self._repo_dir.name, name)
            p.mkdir(exist_ok=True)
            response = await self.client.post("/api/repositories", json={
                "display_name": name.title(),
                "internal_name": name,
                "path": str(p),
            })
            repo_ids.append(response.json()["id"])

        workspace = await self._helper_create_workspace()
        for repo_id in repo_ids:
            added = await self.client.post(
                f"/api/workspaces/{workspace.id}/sources",
                json={"repository_id": repo_id},
            )
            self.assertEqual(added.status_code, 200, added.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="bash",
                parameters=json.dumps({"command": "ls /repositories/"}),
            )),
            Yield(finished=FinishedMessage(content="Found repos: alpha, beta, gamma")),
        )

        resp = await self._helper_prompt("list my repositories", model_id="m")
        self.assertEqual(resp.status_code, 200)

        conv_id = int(resp.headers["x-conversation-id"])
        conv = (await self.client.get(f"/api/conversations/{conv_id}")).json()

        tool_calls = [m for m in conv["messages"] if m["element"]["type"] == "tool_call"]
        self.assertGreaterEqual(len(tool_calls), 1)
        result = json.loads(tool_calls[-1]["element"]["result"])
        self.assertEqual(result["returncode"], 0)
        for name in repo_names:
            self.assertIn(name, result["stdout"])
