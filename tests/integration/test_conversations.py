import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

from tests.helpers import BaseTestCase, Wait, Yield
from inference.engine import FinishedMessage, FinishedToolCall, StreamingMessage


async def _read_ndjson_response(response) -> list[dict[str, Any]]:
    return [json.loads(line) async for line in response.aiter_lines() if line]


class TestConversations(BaseTestCase):

    async def test_helpful_agent_creation(self):
        payload = {
            "display_name": "Helpful Agent",
            "internal_name": "helpful_agent",
            "description": "A helpful agent",
            "instructions": "Be helpful",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)

        list_res = await self.client.get("/api/agents")
        self.assertEqual(list_res.status_code, 200)
        agents = list_res.json()["agents"]
        self.assertTrue(any(a["internal_name"] == "helpful_agent" for a in agents))

    async def test_rude_agent_creation(self):
        payload = {
            "display_name": "Rude Agent",
            "internal_name": "rude_agent",
            "description": "A rude agent",
            "instructions": "Be rude",
            "provider_key": "mock_e2e",
            "model_id": "model-2",
            "inference_config": {},
            "repository_access": []
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201)

        list_res = await self.client.get("/api/agents")
        self.assertEqual(list_res.status_code, 200)
        agents = list_res.json()["agents"]
        self.assertTrue(any(a["internal_name"] == "rude_agent" for a in agents))

    async def test_prompt_stream_finalizes_with_sequence_ids(self):
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="I am here to help!")),
            Yield(finished=FinishedMessage(content="I am here to help!")),
        )
        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(res.status_code, 200)
        events = await _read_ndjson_response(res)
        self.assertEqual(events[0]["type"], "message")
        self.assertEqual(events[0]["content"], "I am here to help!")
        self.assertEqual(events[1]["type"], "finalized")
        self.assertEqual(events[1]["sequence_id"], 2)

        conversation_id = int(res.headers["X-Conversation-ID"])
        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        self.assertEqual([m["sequence_id"] for m in details["messages"]], [1, 2])

    async def test_conversation_stream_replays_buffer_then_live_tokens(self):
        wait = Wait()
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="Hello")),
            wait,
            Yield(streaming=StreamingMessage(content=" world")),
            Yield(finished=FinishedMessage(content="Hello world")),
        )

        prompt_task = asyncio.create_task(self.client.post("/api/conversations/prompt", json={
            "prompt": "Hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        }))

        await asyncio.sleep(0.1)
        conversations = (await self.client.get("/api/conversations")).json()
        self.assertEqual(len(conversations), 1)
        conversation_id = conversations[0]["id"]

        stream_task = asyncio.create_task(self.client.get(f"/api/conversations/{conversation_id}/stream?after_sequence_id=1"))
        await asyncio.sleep(0.1)
        wait.set()

        stream_res = await stream_task
        prompt_res = await prompt_task
        self.assertEqual(stream_res.status_code, 200)
        self.assertEqual(prompt_res.status_code, 200)

        replay_events = await _read_ndjson_response(stream_res)
        self.assertEqual([e["type"] for e in replay_events], ["message", "message", "finalized"])
        self.assertEqual(replay_events[0]["content"], "Hello")
        self.assertEqual(replay_events[1]["content"], " world")
        self.assertEqual(replay_events[2]["sequence_id"], 2)

        prompt_events = await _read_ndjson_response(prompt_res)
        self.assertEqual([e["type"] for e in prompt_events], ["message", "message", "finalized"])
        self.assertEqual(prompt_events[2]["sequence_id"], 2)

    async def test_rejects_second_prompt_while_generation_is_active(self):
        wait = Wait()
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="Hello")),
            wait,
            Yield(finished=FinishedMessage(content="Hello world")),
        )

        prompt_task = asyncio.create_task(self.client.post("/api/conversations/prompt", json={
            "prompt": "Hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        }))
        await asyncio.sleep(0.1)
        conversation_id = (await self.client.get("/api/conversations")).json()[0]["id"]

        second = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Again",
            "conversation_id": conversation_id,
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(second.status_code, 409)

        wait.set()
        res = await prompt_task
        self.assertEqual(res.status_code, 200)
        await _read_ndjson_response(res)

    async def _setup_blocking_conversation(self) -> tuple[int, int]:
        """Create a conversation that ends on a blocking propose_replace tool call.

        Returns (conversation_id, blocking_message_id).
        """
        repo_name = "blocking_repo"
        await self.client.post("/api/repositories", json={
            "display_name": "Blocking Repo",
            "internal_name": repo_name,
            "path": self._repo_dir.name,
            "security": "write",
        })

        source_rel = "source.txt"
        Path(self._workspace_dir.name, source_rel).write_text("new content", encoding="utf-8")
        Path(self._repo_dir.name, "target.txt").write_text("old content", encoding="utf-8")

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="propose_replace",
                parameters=json.dumps({
                    "target": f"/repositories/{repo_name}/target.txt",
                    "source": f"/workspace/{source_rel}",
                }),
            )),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "replace the file",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(res.status_code, 200)
        await _read_ndjson_response(res)

        conversation_id = int(res.headers["X-Conversation-ID"])
        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        blocking_id = details["blocking_message_id"]
        self.assertIsNotNone(blocking_id)
        return conversation_id, blocking_id

    async def test_blocking_tool_call_blocks_conversation(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        self.assertEqual(details["blocking_message_id"], blocking_id)

    async def test_prompt_on_blocked_conversation_returns_403(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "another prompt",
            "conversation_id": conversation_id,
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["blocking_message_id"], blocking_id)

    async def test_decide_clears_block_but_does_not_resume(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        res = await self.client.post(
            f"/api/conversations/{conversation_id}/tool_calls/{blocking_id}/decide",
            json={"decision": "approve"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["executed"])

        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        self.assertIsNone(details["blocking_message_id"])

        # Decide must not auto-resume generation: no new assistant message was produced.
        self.assertEqual(self.mock_provider.calls, 1)
        messages = details["messages"]
        self.assertEqual(messages[-1]["element"]["type"], "tool_call_result")

    async def test_continue_after_decide_resumes_generation(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        await self.client.post(
            f"/api/conversations/{conversation_id}/tool_calls/{blocking_id}/decide",
            json={"decision": "approve"},
        )

        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="done after continue")),
            Yield(finished=FinishedMessage(content="done after continue")),
        )

        res = await self.client.post(f"/api/conversations/{conversation_id}/continue", json={
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(res.status_code, 200)
        events = await _read_ndjson_response(res)
        self.assertEqual(events[0]["type"], "message")
        self.assertEqual(events[0]["content"], "done after continue")
        self.assertEqual(events[1]["type"], "finalized")


if __name__ == "__main__":
    unittest.main()