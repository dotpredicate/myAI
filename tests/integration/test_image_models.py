import unittest

import psycopg.errors

from inference.sd_cpp import image_models
from tests.helpers import BaseTestCase


class TestImageModelStore(BaseTestCase):

    async def test_create_list_get_and_delete(self):
        created = await image_models.create_image_model(
            "Qwen 2.1 Q8",
            "qwen-image-21",
            {"diffusion_model": "diffusion_models/qwen/file.gguf"},
        )

        self.assertGreater(created.id, 0)
        self.assertEqual(created.files, {"diffusion_model": "diffusion_models/qwen/file.gguf"})

        listed = await image_models.list_image_models()
        self.assertEqual([model.name for model in listed], ["Qwen 2.1 Q8"])

        by_id = await image_models.get_image_model(created.id)
        self.assertIsNotNone(by_id)
        by_name = await image_models.get_image_model_by_name("Qwen 2.1 Q8")
        self.assertEqual(by_name, created)
        by_ref = await image_models.get_image_model_by_ref("Qwen 2.1 Q8")
        self.assertEqual(by_ref, created)

        self.assertTrue(await image_models.delete_image_model(created.id))
        self.assertFalse(await image_models.delete_image_model(created.id))
        self.assertEqual(await image_models.list_image_models(), [])

    async def test_name_is_unique(self):
        await image_models.create_image_model("same", "qwen-image-21", {})

        with self.assertRaises(psycopg.errors.UniqueViolation):
            await image_models.create_image_model("same", "flux", {})


if __name__ == "__main__":
    unittest.main()
