import unittest
from tests.helpers import BaseTestCase, MockE2EProvider, Yield
from inference.registry import registry
from inference.engine import FinishedMessage, StreamingMessage


class TestInference(BaseTestCase):
    async def test_registry_register(self):
        registry.register(
            "test_provider",
            "Test Provider",
            "A test provider",
            MockE2EProvider()
        )
        provider = registry.get("test_provider")
        self.assertIsNotNone(provider)

    async def test_registry_register_duplicate(self):
        registry.register(
            "unique_provider",
            "Unique Provider",
            "",
            MockE2EProvider()
        )
        with self.assertRaises(ValueError):
            registry.register(
                "unique_provider",
                "Duplicate",
                "",
                MockE2EProvider()
            )

    async def test_registry_get_missing(self):
        with self.assertRaises(KeyError):
            registry.get("nonexistent")

    async def test_registry_list_registrations(self):
        registry.register(
            "listable_provider",
            "Listable",
            "Can be listed",
            MockE2EProvider()
        )
        registrations = registry.list_registrations()
        keys = [r.key for r in registrations]
        self.assertIn("listable_provider", keys)

    async def test_mock_provider_stream(self):
        provider = MockE2EProvider()
        provider.schedule_stream(
            Yield(streaming=StreamingMessage(content="hello world")),
            Yield(finished=FinishedMessage(content="hello world")),
        )
        from inference.engine import ChatContext
        from domain import Message
        context = ChatContext(messages=[(1, Message(author="user", content="hi"))], scopes=[], tools=[], instructions="")
        async for delta, finished in provider.run_chat_completion_stream("test_model", {}, context):
            if finished is not None:
                self.assertIsInstance(finished, FinishedMessage)
                self.assertEqual(finished.content, "hello world")

    async def test_mock_provider_list_models(self):
        from inference.engine import Model
        models = [Model(id="model-1", created=123, owned_by="test")]
        provider = MockE2EProvider(models=models)
        result = await provider.list_models()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "model-1")


if __name__ == "__main__":
    unittest.main()