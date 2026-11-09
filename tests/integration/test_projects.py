import unittest

from agents import provider_in_use
from tests.helpers import BaseTestCase


class TestProjects(BaseTestCase):

    async def _create_agent(self) -> dict:
        response = await self.client.post("/api/agents", json={
            "display_name": "Project Agent",
            "internal_name": "project_agent",
            "description": "",
            "instructions": "Agent rules",
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
            "inference_config": {},
            "repository_access": [],
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def test_create_and_list_project(self):
        agent = await self._create_agent()
        response = await self.client.post("/api/projects", json={
            "display_name": "My Project",
            "internal_name": "my_project",
            "description": "Project description",
            "instructions": "Project rules",
            "default_agent": agent["id"],
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
            "inference_config": {"temperature": 0.2},
            "repository_access": [],
        })
        self.assertEqual(response.status_code, 201, response.text)
        project = response.json()
        self.assertEqual(project["internal_name"], "my_project")
        self.assertEqual(project["default_agent"], agent["id"])
        self.assertEqual(project["instructions"], "Project rules")
        self.assertEqual(project["inference_config"], {"temperature": 0.2})

        list_response = await self.client.get("/api/projects")
        self.assertEqual(list_response.status_code, 200)
        projects = list_response.json()["projects"]
        self.assertTrue(any(item["internal_name"] == "my_project" for item in projects))

    async def test_update_and_clear_project_fields(self):
        agent = await self._create_agent()
        create_response = await self.client.post("/api/projects", json={
            "display_name": "Editable",
            "internal_name": "editable",
            "instructions": "Old rules",
            "default_agent": agent["id"],
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
        })
        self.assertEqual(create_response.status_code, 201, create_response.text)

        update_response = await self.client.put("/api/projects/editable", json={
            "display_name": "Updated",
            "instructions": None,
            "default_agent": None,
            "provider_key": None,
            "model_id": None,
        })
        self.assertEqual(update_response.status_code, 200, update_response.text)
        project = update_response.json()
        self.assertEqual(project["display_name"], "Updated")
        self.assertIsNone(project["instructions"])
        self.assertIsNone(project["default_agent"])
        self.assertIsNone(project["provider_key"])
        self.assertIsNone(project["model_id"])

    async def test_delete_project(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Delete Me",
            "internal_name": "delete_me",
        })
        self.assertEqual(response.status_code, 201, response.text)

        delete_response = await self.client.delete("/api/projects/delete_me")
        self.assertEqual(delete_response.status_code, 200)

        list_response = await self.client.get("/api/projects")
        self.assertFalse(any(
            item["internal_name"] == "delete_me"
            for item in list_response.json()["projects"]
        ))

    async def test_rejects_invalid_key(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Bad",
            "internal_name": "bad project!",
        })
        self.assertEqual(response.status_code, 400)

    async def test_rejects_duplicate_key(self):
        first = await self.client.post("/api/projects", json={
            "display_name": "First",
            "internal_name": "duplicate",
        })
        self.assertEqual(first.status_code, 201, first.text)

        second = await self.client.post("/api/projects", json={
            "display_name": "Second",
            "internal_name": "duplicate",
        })
        self.assertEqual(second.status_code, 409)

    async def test_rejects_unknown_default_agent(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Missing Agent",
            "internal_name": "missing_agent",
            "default_agent": 9999,
        })
        self.assertEqual(response.status_code, 400)

    async def test_rejects_unknown_provider(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Missing Provider",
            "internal_name": "missing_provider",
            "provider_key": "nonexistent",
            "model_id": "model",
        })
        self.assertEqual(response.status_code, 400)

    async def test_rejects_incomplete_model_settings(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Incomplete",
            "internal_name": "incomplete",
            "provider_key": "mock_e2e",
        })
        self.assertEqual(response.status_code, 400)

    async def test_rejects_policy_escalation(self):
        repo_response = await self._helper_create_repo(
            "read_only_repo", self._repo_dir.name, security="read-only"
        )
        self.assertEqual(repo_response.status_code, 201, repo_response.text)
        repo = repo_response.json()

        response = await self.client.post("/api/projects", json={
            "display_name": "Escalating",
            "internal_name": "escalating",
            "repository_access": [
                {"repository_id": repo["id"], "security_policy_override": "write"},
            ],
        })
        self.assertEqual(response.status_code, 400)

    async def test_repository_access_round_trip(self):
        repo_response = await self._helper_create_repo(
            "project_repo", self._repo_dir.name, security="write"
        )
        self.assertEqual(repo_response.status_code, 201, repo_response.text)
        repo = repo_response.json()

        response = await self.client.post("/api/projects", json={
            "display_name": "With Repo",
            "internal_name": "with_repo",
            "repository_access": [
                {
                    "repository_id": repo["id"],
                    "security_policy_override": "read-only",
                },
            ],
        })
        self.assertEqual(response.status_code, 201, response.text)
        access = response.json()["repository_access"]
        self.assertEqual(len(access), 1)
        self.assertEqual(access[0]["repository_internal_name"], "project_repo")
        self.assertEqual(access[0]["security_policy_override"], "read-only")

    async def test_provider_in_use_counts_projects(self):
        response = await self.client.post("/api/projects", json={
            "display_name": "Provider User",
            "internal_name": "provider_user",
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(await provider_in_use("mock_e2e"))


if __name__ == "__main__":
    unittest.main()
