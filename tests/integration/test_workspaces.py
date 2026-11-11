from pathlib import Path

import paths
from tests.helpers import BaseTestCase


class TestWorkspaces(BaseTestCase):

    async def test_resolve_creates_workspace_repository_and_space(self):
        response = await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        workspace = payload['workspace']
        self.assertTrue(payload['created'])
        self.assertTrue(payload['repository_created'])
        self.assertEqual(len(workspace['sources']), 1)
        self.assertTrue((paths.SPACES_ROOT / str(workspace['id'])).is_dir())

    async def test_resolve_is_idempotent(self):
        first = (await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})).json()
        second = (await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})).json()
        self.assertEqual(first['workspace']['id'], second['workspace']['id'])
        self.assertFalse(second['created'])
        self.assertFalse(second['repository_created'])

    async def test_resolve_rejects_myai_data_directory(self):
        response = await self.client.post('/api/workspaces/resolve', json={'path': str(paths.MYAI_DIR)})
        self.assertEqual(response.status_code, 400)

    async def test_add_and_remove_source(self):
        workspace = (await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})).json()['workspace']
        other = Path(self._myai_dir.name) / 'other'
        other.mkdir()
        repo = (await self.client.post('/api/repositories', json={
            'display_name': 'Other',
            'internal_name': 'other',
            'path': str(other),
        })).json()
        added = await self.client.post(
            f"/api/workspaces/{workspace['id']}/sources",
            json={'repository_id': repo['id']},
        )
        self.assertEqual(added.status_code, 200)
        self.assertEqual(len(added.json()['sources']), 2)
        removed = await self.client.delete(
            f"/api/workspaces/{workspace['id']}/sources/{repo['id']}"
        )
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(len(removed.json()['sources']), 1)

    async def test_delete_workspace_reassigns_conversation_and_keeps_space(self):
        workspace = (await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})).json()['workspace']
        other_path = Path(str(self._myai_dir.name) + '-other')
        other_path.mkdir()
        await self.client.post('/api/workspaces/resolve', json={'path': str(other_path)})
        created = await self.client.post('/api/conversations', json={'workspace_id': workspace['id']})
        conversation_id = created.json()['id']
        space = paths.SPACES_ROOT / str(workspace['id'])
        response = await self.client.delete(f"/api/workspaces/{workspace['id']}")
        self.assertEqual(response.status_code, 200)
        details = (await self.client.get(f'/api/conversations/{conversation_id}')).json()
        self.assertIsNotNone(details['workspace_id'])
        self.assertNotEqual(details['workspace_id'], workspace['id'])
        self.assertTrue(space.exists())

    async def test_cannot_delete_last_workspace(self):
        workspace = (await self.client.post('/api/workspaces/resolve', json={'path': self._repo_dir.name})).json()['workspace']
        workspaces = (await self.client.get('/api/workspaces')).json()['workspaces']
        scratch = next(item for item in workspaces if item['name'] == 'Scratch')
        self.assertEqual((await self.client.delete(f"/api/workspaces/{workspace['id']}")).status_code, 200)
        response = await self.client.delete(f"/api/workspaces/{scratch['id']}")
        self.assertEqual(response.status_code, 409)

    async def test_fresh_database_has_scratch(self):
        response = await self.client.get('/api/workspaces')
        self.assertEqual(response.status_code, 200)
        workspaces = response.json()['workspaces']
        self.assertEqual(len(workspaces), 1)
        self.assertEqual(workspaces[0]['name'], 'Scratch')
        self.assertEqual(workspaces[0]['sources'], [])
