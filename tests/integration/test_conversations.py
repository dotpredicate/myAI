import json
from pathlib import Path
from typing import Any

import paths
from tests.helpers import BaseTestCase, Yield
from inference.engine import FinishedMessage, FinishedToolCall, StreamingMessage


async def _read_ndjson_response(response) -> list[dict[str, Any]]:
    return [json.loads(line) async for line in response.aiter_lines() if line]


class TestConversations(BaseTestCase):

    async def _create_conversation(self, agent_id: str | None = None, path: str | None = None) -> dict[str, Any]:
        workspace = await self._helper_create_workspace(path)
        response = await self.client.post(
            '/api/conversations',
            json={'workspace_id': workspace.id, 'agent_id': agent_id},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def _prompt(self, conversation_id: int, prompt: str = 'Hello', model_id: str = 'model-1'):
        return await self.client.post(
            f'/api/conversations/{conversation_id}/prompt',
            json={
                'prompt': prompt,
                'provider_key': 'mock_e2e',
                'model_id': model_id,
                'inference_config': {},
            },
        )

    async def test_agent_model_is_optional(self):
        response = await self.client.post('/api/agents', json={
            'display_name': 'Plain Agent',
            'internal_name': 'plain_agent',
            'instructions': 'Be plain',
        })
        self.assertEqual(response.status_code, 201, response.text)
        agent = response.json()
        self.assertIsNone(agent['provider_key'])
        self.assertIsNone(agent['model_id'])

        response = await self.client.put('/api/agents/plain_agent', json={
            'provider_key': 'mock_e2e',
            'model_id': 'model-1',
        })
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.put('/api/agents/plain_agent', json={
            'provider_key': None,
            'model_id': None,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()['provider_key'])

    async def test_prompt_stream_finalizes_with_sequence_ids(self):
        self.mock_provider.schedule_stream(
            Yield(streaming=StreamingMessage(content='I am here to help!')),
            Yield(finished=FinishedMessage(content='I am here to help!')),
        )
        created = await self._create_conversation()
        conversation_id = created['id']
        res = await self._prompt(conversation_id)
        self.assertEqual(res.status_code, 200, res.text)
        events = await _read_ndjson_response(res)
        self.assertEqual(events[0]['type'], 'message')
        self.assertEqual(events[0]['content'], 'I am here to help!')
        self.assertEqual(events[1]['type'], 'finalized')
        self.assertEqual(events[1]['sequence_id'], 2)

        details = (await self.client.get(f'/api/conversations/{conversation_id}')).json()
        self.assertEqual([m['sequence_id'] for m in details['messages']], [1, 2])
        self.assertEqual(details['model_id'], 'model-1')
        self.assertEqual(details['workspace_id'], created['workspace_id'])

    async def test_prompt_requires_model(self):
        conversation_id = (await self._create_conversation())['id']
        response = await self.client.post(
            f'/api/conversations/{conversation_id}/prompt',
            json={'prompt': 'Hello'},
        )
        self.assertEqual(response.status_code, 400)

    async def test_prompt_switches_last_used_model(self):
        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='one')))
        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='two')))
        conversation_id = (await self._create_conversation())['id']
        self.assertEqual((await self._prompt(conversation_id)).status_code, 200)
        self.assertEqual((await self._prompt(conversation_id, model_id='model-2')).status_code, 200)
        details = (await self.client.get(f'/api/conversations/{conversation_id}')).json()
        self.assertEqual(details['model_id'], 'model-2')

    async def test_blocking_replace_file_requires_approval(self):
        created = await self._create_conversation()
        conversation_id = created['id']
        source = await self._helper_repo_source()
        target = Path(self._repo_dir.name) / 'note.txt'
        target.write_text('old')
        space_file = paths.SPACES_ROOT / str(created['workspace_id']) / 'note.txt'
        space_file.parent.mkdir(parents=True, exist_ok=True)
        space_file.write_text('new')

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name='replace_file',
                parameters=json.dumps({
                    'target': f'/repositories/{source}/note.txt',
                    'source': '/workspace/note.txt',
                }),
            )),
        )
        res = await self._prompt(conversation_id)
        self.assertEqual(res.status_code, 200, res.text)
        events = await _read_ndjson_response(res)
        self.assertTrue(any(event['type'] == 'approval_request' for event in events))
        self.assertEqual(target.read_text(), 'old')

        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='done')))
        message_id = next(event['original_message_id'] for event in events if event['type'] == 'approval_request')
        decided = await self.client.post(
            f'/api/conversations/{conversation_id}/tool_calls/{message_id}/decide',
            json={'decision': 'approve'},
        )
        self.assertEqual(decided.status_code, 200, decided.text)
        self.assertTrue(decided.json()['executed'])
        self.assertEqual(json.loads(decided.json()['result'])['status'], 'applied')
        self.assertEqual(target.read_text(), 'new')

        continued = await self.client.post(f'/api/conversations/{conversation_id}/continue')
        self.assertEqual(continued.status_code, 200, continued.text)
        continued_events = await _read_ndjson_response(continued)
        self.assertTrue(any(event['type'] == 'message' for event in continued_events))

    async def test_reject_leaves_file_and_continues(self):
        created = await self._create_conversation()
        conversation_id = created['id']
        source = await self._helper_repo_source()
        target = Path(self._repo_dir.name) / 'note.txt'
        target.write_text('old')
        space_file = paths.SPACES_ROOT / str(created['workspace_id']) / 'note.txt'
        space_file.parent.mkdir(parents=True, exist_ok=True)
        space_file.write_text('new')

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name='replace_file',
                parameters=json.dumps({
                    'target': f'/repositories/{source}/note.txt',
                    'source': '/workspace/note.txt',
                }),
            )),
        )
        res = await self._prompt(conversation_id)
        events = await _read_ndjson_response(res)
        message_id = next(event['original_message_id'] for event in events if event['type'] == 'approval_request')

        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='skipped')))
        decided = await self.client.post(
            f'/api/conversations/{conversation_id}/tool_calls/{message_id}/decide',
            json={'decision': 'reject'},
        )
        self.assertEqual(decided.status_code, 200, decided.text)
        self.assertFalse(decided.json()['executed'])
        continued = await self.client.post(f'/api/conversations/{conversation_id}/continue')
        self.assertEqual(continued.status_code, 200, continued.text)
        await _read_ndjson_response(continued)
        self.assertEqual(target.read_text(), 'old')

    async def test_spawn_agent_inherits_parent_model_and_workspace(self):
        child = await self.client.post('/api/agents', json={
            'display_name': 'Child',
            'internal_name': 'child_agent',
            'instructions': 'Child instructions',
            'spawnable_agents': [],
        })
        self.assertEqual(child.status_code, 201, child.text)
        parent = await self.client.post('/api/agents', json={
            'display_name': 'Parent',
            'internal_name': 'parent_agent',
            'instructions': 'Parent instructions',
            'spawnable_agents': ['child_agent'],
        })
        self.assertEqual(parent.status_code, 201, parent.text)

        self.mock_provider.schedule_stream(
            Yield(finished=FinishedToolCall(
                name='spawn_agent',
                parameters=json.dumps({'agent_symbol': 'child_agent', 'prompt': 'say hi'}),
            )),
        )
        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='child answer')))
        self.mock_provider.schedule_stream(Yield(finished=FinishedMessage(content='parent done')))

        conversation_id = (await self._create_conversation(agent_id='parent_agent'))['id']
        res = await self._prompt(conversation_id, prompt='Delegate')
        self.assertEqual(res.status_code, 200, res.text)
        events = await _read_ndjson_response(res)
        tool_events = [event for event in events if event['type'] == 'tool_call']
        self.assertEqual(len(tool_events), 1)
        result = json.loads(tool_events[0]['result'])
        self.assertEqual(result['agent'], 'child_agent')
        self.assertEqual(result['response'], 'child answer')

        child_details = (await self.client.get(f"/api/conversations/{result['conversation_id']}")).json()
        self.assertEqual(child_details['model_id'], 'model-1')
        self.assertEqual(child_details['workspace_id'], (await self.client.get(f'/api/conversations/{conversation_id}')).json()['workspace_id'])

    async def test_delete_conversation_keeps_workspace_space(self):
        created = await self._create_conversation()
        conversation_id = created['id']
        space = paths.SPACES_ROOT / str(created['workspace_id'])
        sandbox = paths.SANDBOXES_ROOT / str(conversation_id)
        self.assertTrue(space.exists())
        response = await self.client.delete(f'/api/conversations/{conversation_id}')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(space.exists())
        self.assertFalse(sandbox.exists())
