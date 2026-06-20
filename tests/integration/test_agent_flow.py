import unittest
from tests.helpers import BaseTestCase, MockInferenceProvider
from inference.registry import registry

class TestAgentFlow(BaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        registry.register(
            "mock_provider",
            "Mock Provider",
            "A mock provider for testing",
            MockInferenceProvider(response_content="Mock response")
        )

    async def test_agent_lifecycle(self):
        payload = {
            "display_name": "E2E Test Agent",
            "internal_name": "e2e_agent",
            "description": "Testing the full lifecycle",
            "instructions": "Be a good agent",
            "provider_key": "mock_provider",
            "model_id": "dummy_model",
            "inference_config": {"temp": 0.5},
            "repository_access": []
        }

        create_res = await self.client.post("/api/agents", json=payload)
        self.assertEqual(create_res.status_code, 201, f"Failed to create agent: {create_res.text}")
        
        created_data = create_res.json()
        self.assertEqual(created_data["internal_name"], "e2e_agent")

        list_res = await self.client.get("/api/agents")
        self.assertEqual(list_res.status_code, 200)
        
        agents = list_res.json()["agents"]
        self.assertTrue(any(a["internal_name"] == "e2e_agent" for a in agents))

        agent_in_list = next(a for a in agents if a["internal_name"] == "e2e_agent")
        self.assertEqual(agent_in_list["display_name"], "E2E Test Agent")
        self.assertEqual(agent_in_list["provider_key"], "mock_provider")

if __name__ == "__main__":
    unittest.main()
