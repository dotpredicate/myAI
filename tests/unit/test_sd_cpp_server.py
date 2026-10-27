import unittest
from typing import Any

from inference.sd_cpp import provider as server
from inference.sd_cpp import recipes as server_recipes
from inference.engine import InferenceParamType

DEFAULTS: dict[str, Any] = {
    "width": 2048,
    "height": 2048,
    "steps": 20,
    "cfg_scale": 1.0,
    "sampling_method": "euler",
    "scheduler": "karras",
}

RECIPE: dict[str, Any] = {
    "id": "qwen-image-21",
    "name": "Qwen Image 2.1",
    "slots": [
        {"id": "diffusion_model", "required": True, "label": "Diffusion model", "categories": ["diffusion_models"]},
        {"id": "llm", "required": True, "label": "Text encoder", "categories": ["text_encoders"]},
        {"id": "vae", "required": False, "label": "VAE", "categories": ["vae"]},
    ],
    "required_any": [["vae"]],
    "defaults": DEFAULTS,
}

CATALOG: dict[str, dict[str, Any]] = {
    "diffusion_models/model.gguf": {"id": "diffusion_models/model.gguf", "category": "diffusion_models"},
    "text_encoders/text.gguf": {"id": "text_encoders/text.gguf", "category": "text_encoders"},
    "vae/vae.safetensors": {"id": "vae/vae.safetensors", "category": "vae"},
}


def _body(overrides: dict[str, Any]) -> dict[str, Any]:
    return server_recipes.build_generation_request(overrides)


class TestGenerationParams(unittest.TestCase):
    def test_scalar_recipe_defaults_become_controls(self) -> None:
        params = {param.name: param for param in server_recipes._generation_params(DEFAULTS)}

        self.assertEqual((params["width"].default, params["height"].default), (2048, 2048))
        self.assertEqual(params["steps"].default, 20)
        self.assertEqual(params["cfg_scale"].default, 1.0)
        self.assertEqual(params["sampling_method"].type, InferenceParamType.STR)
        self.assertEqual(params["sampling_method"].default, "euler")
        self.assertEqual(params["seed"].default, -1)

    def test_lists_are_skipped(self) -> None:
        params = server_recipes._generation_params({"custom_sigmas": [1.0, 0.5]})

        self.assertNotIn("custom_sigmas", {param.name for param in params})


class TestGenerationRequest(unittest.TestCase):
    def test_recipe_defaults_are_mapped_to_the_api_shape(self) -> None:
        body = _body(DEFAULTS)

        self.assertEqual((body["width"], body["height"]), (2048, 2048))
        sample_params = body["sample_params"]
        assert isinstance(sample_params, dict)
        self.assertEqual(sample_params["sample_steps"], 20)
        self.assertEqual(sample_params["sample_method"], "euler")
        self.assertEqual(sample_params["scheduler"], "karras")
        guidance = sample_params["guidance"]
        assert isinstance(guidance, dict)
        self.assertEqual(guidance["txt_cfg"], 1.0)

    def test_requested_values_override_recipe_defaults(self) -> None:
        body = _body({
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg_scale": 7.5,
            "seed": 42,
        })

        self.assertEqual(body["negative_prompt"], "blurry")
        self.assertEqual((body["width"], body["height"]), (512, 512))
        self.assertEqual(body["seed"], 42)
        sample_params = body["sample_params"]
        assert isinstance(sample_params, dict)
        self.assertEqual(sample_params["sample_steps"], 10)
        guidance = sample_params["guidance"]
        assert isinstance(guidance, dict)
        self.assertEqual(guidance["txt_cfg"], 7.5)


class TestMissingSlots(unittest.TestCase):
    def test_required_and_required_any(self) -> None:
        missing = server_recipes._missing_selection(RECIPE, {"diffusion_model": "x"})

        self.assertIn("Text encoder", missing)
        self.assertIn("one of: vae", missing)
        self.assertNotIn("Diffusion model", missing)

    def test_files_must_exist_in_the_server_catalog(self) -> None:
        missing = server_recipes._missing_slots(
            RECIPE,
            {
                "diffusion_model": "diffusion_models/model.gguf",
                "llm": "text_encoders/text.gguf",
                "vae": "vae/vae.safetensors",
            },
            CATALOG,
        )

        self.assertEqual(missing, [])

    def test_unknown_file_is_reported(self) -> None:
        missing = server_recipes._missing_slots(
            RECIPE,
            {
                "diffusion_model": "diffusion_models/model.gguf",
                "llm": "text_encoders/missing.gguf",
                "vae": "vae/vae.safetensors",
            },
            CATALOG,
        )

        self.assertTrue(any("not in model directory" in item for item in missing))

    def test_wrong_category_is_reported(self) -> None:
        missing = server_recipes._missing_slots(
            RECIPE,
            {
                "diffusion_model": "vae/vae.safetensors",
                "llm": "text_encoders/text.gguf",
                "vae": "vae/vae.safetensors",
            },
            CATALOG,
        )

        self.assertTrue(any("wrong category" in item for item in missing))


class TestParseJobResult(unittest.TestCase):
    def test_returns_first_image(self) -> None:
        self.assertEqual(server.parse_job_result({"status": "completed", "result": {"images": [{"b64_json": "aGk="}]}}), "aGk=")

    def test_rejects_incomplete_job(self) -> None:
        with self.assertRaises(RuntimeError):
            server.parse_job_result({"status": "failed", "result": {"images": []}})


if __name__ == "__main__":
    unittest.main()
