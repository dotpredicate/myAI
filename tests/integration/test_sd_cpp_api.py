import unittest

from tests.helpers import BaseTestCase


class TestSdCppApi(BaseTestCase):

    async def test_models_endpoint_lists_the_default(self):
        res = await self.client.get("/api/sd-cpp/models")

        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["default"], "mock-image")
        self.assertEqual(body["models"], [{
            "id": "mock-image",
            "name": "Mock Image",
            "available": True,
            "missing": [],
        }])

    async def test_add_model_is_rejected_for_providers_without_file_catalog(self):
        res = await self.client.post("/api/sd-cpp/models", json={
            "name": "new model",
            "recipe": "qwen-image-21",
            "files": {"diffusion_model": "owner/repo/file.gguf"},
        })

        self.assertEqual(res.status_code, 501)

    async def test_recipes_and_files_are_empty_for_mock_provider(self):
        recipes = await self.client.get("/api/sd-cpp/recipes")
        files = await self.client.get("/api/sd-cpp/files")

        self.assertEqual(recipes.status_code, 200)
        self.assertEqual(recipes.json(), {"recipes": []})
        self.assertEqual(files.status_code, 200)
        self.assertEqual(files.json(), {"files": []})

    async def test_inference_params_are_served_per_model(self):
        res = await self.client.get("/api/sd-cpp/inference-params?model=mock-image")

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual([p["name"] for p in res.json()], ["steps"])

    async def test_inference_params_reject_unknown_model(self):
        res = await self.client.get("/api/sd-cpp/inference-params?model=nope")

        self.assertEqual(res.status_code, 404)

    async def test_inference_params_default_to_the_provider_model(self):
        res = await self.client.get("/api/sd-cpp/inference-params")

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual([p["name"] for p in res.json()], ["steps"])

    async def test_model_management_is_not_served_under_images(self):
        res = await self.client.get("/api/images/models")

        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
