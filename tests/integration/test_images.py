import unittest

from tests.helpers import BaseTestCase


class TestImages(BaseTestCase):

    async def test_generate_passes_model_and_params(self):
        self.mock_image_provider.result = "aGVsbG8="

        res = await self.client.post("/api/images/generate", json={
            "model": "mock-image",
            "prompt": "a lovely cat",
            "params": {"steps": 10, "cfg_scale": 7.0, "width": 512, "height": 512, "seed": 42},
        })

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["image"], "aGVsbG8=")
        self.assertEqual(len(self.mock_image_provider.requests), 1)
        req = self.mock_image_provider.requests[0]
        self.assertEqual(req.model, "mock-image")
        self.assertEqual(req.prompt, "a lovely cat")
        self.assertEqual(req.params, {"steps": 10, "cfg_scale": 7.0, "width": 512, "height": 512, "seed": 42})

    async def test_generate_defaults_params_to_empty_dict(self):
        res = await self.client.post("/api/images/generate", json={
            "model": "mock-image",
            "prompt": "a lovely cat",
        })

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.mock_image_provider.requests[0].params, {})

    async def test_generate_rejects_unknown_model(self):
        res = await self.client.post("/api/images/generate", json={
            "model": "nope",
            "prompt": "a lovely cat",
        })

        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.mock_image_provider.requests, [])

    async def test_generate_rejects_empty_prompt(self):
        res = await self.client.post("/api/images/generate", json={
            "model": "mock-image",
            "prompt": "   ",
        })

        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.mock_image_provider.requests, [])

    async def test_generate_maps_missing_backend_to_503(self):
        self.mock_image_provider.error = FileNotFoundError()

        res = await self.client.post("/api/images/generate", json={
            "model": "mock-image",
            "prompt": "a lovely cat",
        })

        self.assertEqual(res.status_code, 503)

    async def test_generate_maps_provider_failure_to_502(self):
        self.mock_image_provider.error = RuntimeError("boom")

        res = await self.client.post("/api/images/generate", json={
            "model": "mock-image",
            "prompt": "a lovely cat",
        })

        self.assertEqual(res.status_code, 502)
        self.assertIn("boom", res.json()["error"])

    async def test_generate_uses_default_model_when_omitted(self):
        res = await self.client.post("/api/images/generate", json={"prompt": "a lovely cat"})

        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.mock_image_provider.requests[0].model, "mock-image")


if __name__ == "__main__":
    unittest.main()
