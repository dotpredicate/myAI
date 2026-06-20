import unittest
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

        delete_res = await self.client.delete("/api/repositories/test_repo")
        self.assertEqual(delete_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()