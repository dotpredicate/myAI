import json
import unittest

from inference.engine import FinishedMessage, FinishedToolCall
from tests.helpers import BaseTestCase, Yield


async def _read_ndjson_response(response) -> list[dict[str, object]]:
    return [json.loads(line) async for line in response.aiter_lines() if line]


class TestProjectConversations(BaseTestCase):

    async def _create_project(self, internal_name: str, instructions: str | None = None) -> dict:
        response = await self.client.post("/api/projects", json={
            "display_name": internal_name.replace("_", " ").title(),
            "internal_name": internal_name,
            "instructions": instructions,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def test_project_instructions_are_applied_and_conversation_is_bound(self):
        await self._create_project("rules_project", "Project rules")
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="done")),
        )

        response = await self.client.post("/api/conversations/prompt", json={
            "prompt": "hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "project_id": "rules_project",
        })
        self.assertEqual(response.status_code, 200, response.text)
        await _read_ndjson_response(response)

        conversation_id = int(response.headers["X-Conversation-ID"])
        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        self.assertEqual(details["project_id"], "rules_project")
        self.assertEqual(self.mock_provider.contexts[-1].project_instructions, "Project rules")

    async def test_project_and_agent_instructions_are_both_applied(self):
        await self._create_project("combined_project", "Project rules")
        agent_response = await self.client.post("/api/agents", json={
            "display_name": "Combined Agent",
            "internal_name": "combined_agent",
            "description": "",
            "instructions": "Agent rules",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
        })
        self.assertEqual(agent_response.status_code, 201, agent_response.text)
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="done")),
        )

        response = await self.client.post("/api/conversations/prompt", json={
            "prompt": "hello",
            "agent_id": "combined_agent",
            "project_id": "combined_project",
        })
        self.assertEqual(response.status_code, 200, response.text)
        await _read_ndjson_response(response)

        context = self.mock_provider.contexts[-1]
        self.assertEqual(context.project_instructions, "Project rules")
        self.assertEqual(context.instructions, "Agent rules")

    async def test_project_cannot_change_after_creation(self):
        await self._create_project("first_project")
        await self._create_project("second_project")
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="first")),
            Yield(finished=FinishedMessage(content="second")),
        )

        first = await self.client.post("/api/conversations/prompt", json={
            "prompt": "hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "project_id": "first_project",
        })
        self.assertEqual(first.status_code, 200, first.text)
        await _read_ndjson_response(first)
        conversation_id = int(first.headers["X-Conversation-ID"])

        second = await self.client.post("/api/conversations/prompt", json={
            "prompt": "again",
            "conversation_id": conversation_id,
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "project_id": "second_project",
        })
        self.assertEqual(second.status_code, 400)
        self.assertIn("project", second.json()["error"])

    async def test_unknown_project_is_rejected(self):
        response = await self.client.post("/api/conversations/prompt", json={
            "prompt": "hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "project_id": "missing_project",
        })
        self.assertEqual(response.status_code, 400)

    async def test_spawned_agent_inherits_project(self):
        await self._create_project("spawn_project", "Project rules")

        child_response = await self.client.post("/api/agents", json={
            "display_name": "Child",
            "internal_name": "project_child",
            "description": "",
            "instructions": "Child rules",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
        })
        self.assertEqual(child_response.status_code, 201, child_response.text)

        parent_response = await self.client.post("/api/agents", json={
            "display_name": "Parent",
            "internal_name": "project_parent",
            "description": "",
            "instructions": "Parent rules",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
            "spawnable_agents": ["project_child"],
        })
        self.assertEqual(parent_response.status_code, 201, parent_response.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "project_child", "prompt": "do work"}),
            )),
            Yield(finished=FinishedMessage(content="parent done")),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="child done")),
        )

        response = await self.client.post("/api/conversations/prompt", json={
            "prompt": "delegate",
            "agent_id": "project_parent",
            "project_id": "spawn_project",
        })
        self.assertEqual(response.status_code, 200, response.text)
        await _read_ndjson_response(response)

        parent_id = int(response.headers["X-Conversation-ID"])
        parent = (await self.client.get(f"/api/conversations/{parent_id}")).json()
        tool_calls = [
            message for message in parent["messages"]
            if message["element"]["type"] == "tool_call"
        ]
        self.assertTrue(tool_calls)
        child_result = json.loads(tool_calls[-1]["element"]["result"])
        child_id = child_result["conversation_id"]

        child = (await self.client.get(f"/api/conversations/{child_id}")).json()
        self.assertEqual(child["project_id"], "spawn_project")
        self.assertTrue(any(
            context.project_instructions == "Project rules"
            for context in self.mock_provider.contexts
        ))


if __name__ == "__main__":
    unittest.main()
