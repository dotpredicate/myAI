from pathlib import Path

from tests.helpers import BaseTestCase

class TestRepositories(BaseTestCase):
    async def test_create_repository(self):
        payload = {
            "display_name": "Test Repo",
            "internal_name": "test_repo",
            "path": self._repo_dir.name,
        }

        create_res = await self.client.post("/api/repositories", json=payload)
        self.assertEqual(create_res.status_code, 201)

        created_data = create_res.json()
        self.assertEqual(created_data["internal_name"], "test_repo")
        self.assertEqual(created_data["display_name"], "Test Repo")
        self.assertFalse(created_data["index_enabled"])

        delete_res = await self.client.delete("/api/repositories/test_repo")
        self.assertEqual(delete_res.status_code, 200)

    async def test_update_repository_toggles_indexing(self):
        created = (await self.client.post("/api/repositories", json={
            "display_name": "Toggle Repo",
            "internal_name": "toggle_repo",
            "path": self._repo_dir.name,
            "index_enabled": True,
        })).json()
        self.assertTrue(created["index_enabled"])

        updated = await self.client.put(
            "/api/repositories/toggle_repo",
            json={"index_enabled": False},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertFalse(updated.json()["index_enabled"])

        invalid = await self.client.put(
            "/api/repositories/toggle_repo",
            json={"index_enabled": "yes"},
        )
        self.assertEqual(invalid.status_code, 400)

        repositories = (await self.client.get("/api/repositories")).json()["repositories"]
        self.assertFalse(repositories[0]["index_enabled"])

    async def test_delete_repository_detaches_it_from_workspaces(self):
        workspace = await self._helper_create_workspace()
        source = workspace.sources[0]

        delete_res = await self.client.delete(f"/api/repositories/{source.internal_name}")
        self.assertEqual(delete_res.status_code, 200, delete_res.text)

        refreshed = (await self.client.get(f"/api/workspaces/{workspace.id}")).json()
        self.assertEqual(refreshed["sources"], [])

        repositories = (await self.client.get("/api/repositories")).json()["repositories"]
        self.assertFalse(any(repo["internal_name"] == source.internal_name for repo in repositories))

        # Files on disk are never touched.
        self.assertTrue(Path(self._repo_dir.name).is_dir())
