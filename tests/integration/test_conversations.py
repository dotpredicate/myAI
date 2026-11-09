import asyncio
import json
import unittest
from pathlib import Path
from typing import Any

import paths
from tests.helpers import BaseTestCase, Wait, Yield
from inference.engine import FinishedMessage, FinishedThinking, FinishedToolCall, StreamingMessage


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

    async def test_spawn_agent_starts_child_conversation(self):
        child_payload = {
            "display_name": "Child Agent",
            "internal_name": "child_agent",
            "description": "Child",
            "instructions": "Answer the sub-task",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
        }
        child_res = await self.client.post("/api/agents", json=child_payload)
        self.assertEqual(child_res.status_code, 201, child_res.text)

        parent_payload = {
            **child_payload,
            "display_name": "Parent Agent",
            "internal_name": "parent_agent",
            "spawnable_agents": ["child_agent"],
        }
        parent_res = await self.client.post("/api/agents", json=parent_payload)
        self.assertEqual(parent_res.status_code, 201, parent_res.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "child_agent", "prompt": "say hi"}),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="child answer")),
            Yield(finished=FinishedMessage(content="child answer")),
        )
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="parent done")),
            Yield(finished=FinishedMessage(content="parent done")),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Delegate to the child",
            "agent_id": "parent_agent",
        })
        self.assertEqual(res.status_code, 200)
        events = await _read_ndjson_response(res)
        tool_events = [event for event in events if event["type"] == "tool_call"]
        self.assertEqual(len(tool_events), 1)
        tool_result = json.loads(tool_events[0]["result"])
        self.assertEqual(tool_result["agent"], "child_agent")
        self.assertEqual(tool_result["response"], "child answer")
        self.assertEqual(self.mock_provider.calls, 3)

        child_id = tool_result["conversation_id"]
        details = (await self.client.get(f"/api/conversations/{child_id}")).json()
        self.assertEqual(
            [message["element"]["content"] for message in details["messages"] if message["role"] == "assistant"],
            ["child answer"],
        )
        self.assertTrue(any(
            event["type"] == "message" and event.get("content") == "parent done"
            for event in events
        ))

    async def test_spawn_agent_rejects_agent_outside_allowlist(self):
        payload = {
            "display_name": "Lonely Agent",
            "internal_name": "lonely_agent",
            "description": "No children",
            "instructions": "Do not delegate",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
        }
        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201, create_res.text)
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "child_agent", "prompt": "say hi"}),
            )),
            Yield(finished=FinishedMessage(content="could not delegate")),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Delegate anyway",
            "agent_id": "lonely_agent",
        })
        events = await _read_ndjson_response(res)
        tool_events = [event for event in events if event["type"] == "tool_call"]
        self.assertEqual(len(tool_events), 1)
        self.assertIn("not in the allowed list", tool_events[0]["result"])

    async def test_spawn_agent_child_inherits_parent_scopes(self):
        repo_res = await self._helper_create_repo(
            "spawn_scope_repo", self._repo_dir.name, security="read-only",
        )
        self.assertEqual(repo_res.status_code, 201, repo_res.text)
        access = [{"repository_id": repo_res.json()["id"]}]
        child_payload = {
            "display_name": "Scoped Child",
            "internal_name": "scoped_child",
            "description": "Child",
            "instructions": "Run a command",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": access,
        }
        child_res = await self.client.post("/api/agents", json=child_payload)
        self.assertEqual(child_res.status_code, 201, child_res.text)

        parent_res = await self.client.post("/api/agents", json={
            **child_payload,
            "display_name": "Scoped Parent",
            "internal_name": "scoped_parent",
            "spawnable_agents": ["scoped_child"],
        })
        self.assertEqual(parent_res.status_code, 201, parent_res.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "scoped_child", "prompt": "run true"}),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(name="bash", parameters=json.dumps({"command": "true"}))),
            Yield(finished=FinishedMessage(content="child done")),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="parent done")),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Delegate",
            "agent_id": "scoped_parent",
        })
        events = await _read_ndjson_response(res)
        spawn_result = json.loads(
            next(event for event in events if event["type"] == "tool_call")["result"]
        )

        child_details = (await self.client.get(
            f"/api/conversations/{spawn_result['conversation_id']}"
        )).json()
        bash_call = next(
            message["element"]
            for message in child_details["messages"]
            if message["element"]["type"] == "tool_call" and message["element"]["name"] == "bash"
        )
        self.assertEqual(
            [scope["internal_name"] for scope in bash_call["scopes"]],
            ["spawn_scope_repo"],
        )

    async def test_spawn_agent_reports_blocked_child(self):
        repo_res = await self._helper_create_repo(
            "spawn_block_repo", self._repo_dir.name, security="write",
        )
        self.assertEqual(repo_res.status_code, 201, repo_res.text)
        access = [{"repository_id": repo_res.json()["id"]}]
        child_payload = {
            "display_name": "Blocked Child",
            "internal_name": "blocked_child",
            "description": "Child",
            "instructions": "Write a file",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": access,
        }
        child_res = await self.client.post("/api/agents", json=child_payload)
        self.assertEqual(child_res.status_code, 201, child_res.text)

        parent_res = await self.client.post("/api/agents", json={
            **child_payload,
            "display_name": "Blocked Parent",
            "internal_name": "blocked_parent",
            "spawnable_agents": ["blocked_child"],
        })
        self.assertEqual(parent_res.status_code, 201, parent_res.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "blocked_child", "prompt": "write it"}),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="bash",
                parameters=json.dumps({"command": "printf new > /workspace/source.txt"}),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="replace_file",
                parameters=json.dumps({
                    "target": "/repositories/spawn_block_repo/target.txt",
                    "source": "/workspace/source.txt",
                }),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="parent done")),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Delegate",
            "agent_id": "blocked_parent",
        })
        events = await _read_ndjson_response(res)
        spawn_result = json.loads(
            next(event for event in events if event["type"] == "tool_call")["result"]
        )
        self.assertTrue(spawn_result["blocked"])
        child_details = (await self.client.get(
            f"/api/conversations/{spawn_result['conversation_id']}"
        )).json()
        self.assertEqual(
            child_details["blocking_message_id"],
            spawn_result["blocking_message_id"],
        )

    async def test_spawn_agent_rejects_thinking_only_child(self):
        child_payload = {
            "display_name": "Thinking Child",
            "internal_name": "thinking_child",
            "description": "Child",
            "instructions": "Think",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
            "inference_config": {},
            "repository_access": [],
        }
        child_res = await self.client.post("/api/agents", json=child_payload)
        self.assertEqual(child_res.status_code, 201, child_res.text)

        parent_res = await self.client.post("/api/agents", json={
            **child_payload,
            "display_name": "Thinking Parent",
            "internal_name": "thinking_parent",
            "spawnable_agents": ["thinking_child"],
        })
        self.assertEqual(parent_res.status_code, 201, parent_res.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="spawn_agent",
                parameters=json.dumps({"agent_symbol": "thinking_child", "prompt": "think"}),
            )),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedThinking(content="only thinking")),
        )
        self.mock_provider.schedule_stream(
            Yield(finished=FinishedMessage(content="parent done")),
        )

        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Delegate",
            "agent_id": "thinking_parent",
        })
        events = await _read_ndjson_response(res)
        spawn_result = json.loads(
            next(event for event in events if event["type"] == "tool_call")["result"]
        )
        self.assertIn("produced no message", spawn_result["error"])

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

    async def test_cancel_stops_generation_and_keeps_transcript(self):
        wait = Wait()
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="partial")),
            wait,
            Yield(finished=FinishedMessage(content="never")),
        )

        prompt_task = asyncio.create_task(self.client.post("/api/conversations/prompt", json={
            "prompt": "Hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        }))
        await asyncio.sleep(0.1)
        conversation_id = (await self.client.get("/api/conversations")).json()[0]["id"]
        workspace = paths.WORKSPACES_ROOT / str(conversation_id)
        workspace.mkdir(parents=True, exist_ok=True)

        cancel_res = await self.client.post(f"/api/conversations/{conversation_id}/cancel")
        self.assertEqual(cancel_res.status_code, 200)
        self.assertEqual(cancel_res.json(), {"status": "cancelled", "id": conversation_id})

        prompt_res = await prompt_task
        self.assertEqual(prompt_res.status_code, 200)
        await _read_ndjson_response(prompt_res)

        details = (await self.client.get(f"/api/conversations/{conversation_id}")).json()
        self.assertEqual([m["element"]["type"] for m in details["messages"]], ["message"])
        self.assertTrue(workspace.exists())

        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="after cancel")),
            Yield(finished=FinishedMessage(content="after cancel")),
        )
        again = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Again",
            "conversation_id": conversation_id,
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        self.assertEqual(again.status_code, 200)
        events = await _read_ndjson_response(again)
        self.assertEqual(events[0]["content"], "after cancel")

    async def test_cancel_idle_conversation_returns_conflict(self):
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="done")),
            Yield(finished=FinishedMessage(content="done")),
        )
        res = await self.client.post("/api/conversations/prompt", json={
            "prompt": "Hello",
            "provider_key": "mock_e2e",
            "model_id": "model-1",
        })
        await _read_ndjson_response(res)
        conversation_id = int(res.headers["X-Conversation-ID"])

        cancel_res = await self.client.post(f"/api/conversations/{conversation_id}/cancel")
        self.assertEqual(cancel_res.status_code, 409)

    async def test_cancel_missing_conversation_returns_not_found(self):
        res = await self.client.post("/api/conversations/9999/cancel")
        self.assertEqual(res.status_code, 404)

    async def _setup_blocking_conversation(self) -> tuple[int, int]:
        """Create a conversation that ends on a blocking replace_file tool call.

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
        Path(self._repo_dir.name, "target.txt").write_text("old content", encoding="utf-8")

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name="bash",
                parameters=json.dumps({"command": f"printf 'new content' > /workspace/{source_rel}"}),
            )),
            Yield(finished=FinishedToolCall(
                name="replace_file",
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
            "scopes": [{"internal_name": repo_name}],
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

    async def test_stream_replays_approval_request_and_decision(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        res = await self.client.post(
            f"/api/conversations/{conversation_id}/tool_calls/{blocking_id}/decide",
            json={"decision": "reject", "comment": "not now"},
        )
        self.assertEqual(res.status_code, 200)

        res = await self.client.get(f"/api/conversations/{conversation_id}/stream")
        self.assertEqual(res.status_code, 200)
        events = await _read_ndjson_response(res)

        approval_events = [e for e in events if e["type"] == "approval_request"]
        self.assertEqual(len(approval_events), 1)
        self.assertEqual(approval_events[0]["original_message_id"], blocking_id)

        decision_events = [e for e in events if e["type"] == "tool_call_decision"]
        self.assertEqual(len(decision_events), 1)
        self.assertEqual(decision_events[0]["original_message_id"], blocking_id)
        self.assertEqual(decision_events[0]["decision"], "reject")
        self.assertEqual(decision_events[0]["comment"], "not now")

    async def test_approve_returns_result_and_stream_replays_it(self):
        conversation_id, blocking_id = await self._setup_blocking_conversation()

        res = await self.client.post(
            f"/api/conversations/{conversation_id}/tool_calls/{blocking_id}/decide",
            json={"decision": "approve"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["executed"])
        self.assertIn("applied", res.json()["result"])

        res = await self.client.get(f"/api/conversations/{conversation_id}/stream")
        self.assertEqual(res.status_code, 200)
        events = await _read_ndjson_response(res)

        result_events = [e for e in events if e["type"] == "tool_call_result"]
        self.assertEqual(len(result_events), 1)
        self.assertEqual(result_events[0]["original_message_id"], blocking_id)
        self.assertIn("applied", result_events[0]["result"])

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

    async def test_delete_conversation_removes_workspace(self):
        conversation_id, _ = await self._setup_blocking_conversation()
        workspace = paths.WORKSPACES_ROOT / str(conversation_id)
        self.assertTrue(workspace.exists())

        res = await self.client.delete(f"/api/conversations/{conversation_id}")

        self.assertEqual(res.status_code, 200)
        self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()