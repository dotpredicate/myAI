import unittest
from types import SimpleNamespace
from typing import Any, cast

from domain import InferenceProviderConfig, Message, Thinking
from inference.engine import ChatContext, FinishedMessage, FinishedThinking, FinishedToolCall
from inference.openai_chat import DeltaProcessor, OpenAICompatibleProvider, _to_oai_messages


def _chunk(**delta: Any) -> Any:
    values: dict[str, Any] = {"tool_calls": None, "reasoning_content": None, "content": None}
    values.update(delta)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**values), finish_reason=None)])


def _finish_chunk(reason: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(tool_calls=None, reasoning_content=None, content=None),
        finish_reason=reason,
    )])


def _truncated_chunk(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(tool_calls=None, reasoning_content=None, content=content),
        finish_reason="length",
    )])


def _tool_chunk(name: str | None, arguments: str) -> Any:
    return _chunk(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
    ])


class FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


class FakeModels:
    def __init__(self, models: list[Any] | None = None, error: Exception | None = None) -> None:
        self.models = models or []
        self.error = error

    async def list(self) -> Any:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.models)


class FakeClient:
    def __init__(self, chunks: list[Any], models: list[Any] | None = None, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(chunks))
        self.models = FakeModels(models, error)


def _config(**overrides: Any) -> InferenceProviderConfig:
    values: dict[str, Any] = {
        "key": "remote",
        "display_name": "Remote",
        "base_url": "https://example.com/v1",
        "api_key": "secret",
    }
    values.update(overrides)
    return InferenceProviderConfig(**values)


def _context(messages: list[Any] | None = None) -> ChatContext:
    return ChatContext(messages=messages or [], scopes=[], tools=[], instructions=None)


class TestDeltaProcessor(unittest.TestCase):
    def test_flushes_message_text(self):
        processor = DeltaProcessor()
        first = processor.process(_chunk(content="Hello"))
        second = processor.process(_chunk(content=" world"))
        finished = processor.flush()

        self.assertEqual(first[1], [])
        self.assertEqual(second[1], [])
        self.assertEqual(finished, [FinishedMessage(content="Hello world")])

    def test_merges_tool_call_arguments(self):
        processor = DeltaProcessor()
        processor.process(_tool_chunk("bash", '{"command":'))
        processor.process(_tool_chunk(None, ' "ls"}'))
        finished = processor.flush()

        self.assertEqual(finished, [FinishedToolCall(name="bash", parameters='{"command": "ls"}')])

    def test_accumulates_multiple_tool_calls_by_index(self):
        processor = DeltaProcessor()
        processor.process(_chunk(tool_calls=[
            SimpleNamespace(index=0, function=SimpleNamespace(name="bash", arguments='{"command":"a"}')),
            SimpleNamespace(index=1, function=SimpleNamespace(name="read", arguments='{"path":"b"}')),
        ]))

        self.assertEqual(processor.flush(), [
            FinishedToolCall(name="bash", parameters='{"command":"a"}'),
            FinishedToolCall(name="read", parameters='{"path":"b"}'),
        ])

    def test_accepts_name_after_arguments(self):
        processor = DeltaProcessor()
        processor.process(_chunk(tool_calls=[
            SimpleNamespace(index=0, function=SimpleNamespace(name=None, arguments='{"command":"a"')),
        ]))
        processor.process(_chunk(tool_calls=[
            SimpleNamespace(index=0, function=SimpleNamespace(name="bash", arguments='}')),
        ]))

        self.assertEqual(processor.flush(), [
            FinishedToolCall(name="bash", parameters='{"command":"a"}'),
        ])

    def test_prefers_visible_content_over_reasoning(self):
        processor = DeltaProcessor()
        processor.process(_chunk(content="visible", reasoning_content="hidden"))

        self.assertEqual(processor.flush(), [FinishedMessage(content="visible")])

    def test_drops_tool_call_without_name(self):
        processor = DeltaProcessor()
        processor.process(_tool_chunk(None, '{}'))

        self.assertEqual(processor.flush(), [])

    def test_ignores_tool_call_chunk_without_function(self):
        processor = DeltaProcessor()
        processor.process(_tool_chunk("bash", '{"command":'))
        processor.process(_chunk(tool_calls=[SimpleNamespace(function=None)]))

        self.assertEqual(
            processor.flush(),
            [FinishedToolCall(name="bash", parameters='{"command":')],
        )


class TestOpenAICompatibleProvider(unittest.IsolatedAsyncioTestCase):
    async def test_streams_text_and_sends_openai_options(self):
        client = FakeClient([_chunk(content="Hello"), _chunk(content=" world")])
        provider = OpenAICompatibleProvider(
            _config(extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
            client=cast(Any, client),
        )

        events = [
            event
            async for event in provider.run_chat_completion_stream(
                "model-a", {"temperature": 0.2}, _context()
            )
        ]

        self.assertEqual([event[0].content for event in events if event[0] is not None], ["Hello", " world"])
        self.assertEqual(events[-1][1], FinishedMessage(content="Hello world"))
        assert client.chat.completions.kwargs is not None
        kwargs = client.chat.completions.kwargs
        self.assertEqual(kwargs["model"], "model-a")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("parallel_tool_calls", kwargs)
        self.assertEqual(kwargs["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("tools", kwargs)

    async def test_sends_tools_and_disables_parallel_tool_calls(self):
        from tools import TOOL_REGISTRY

        client = FakeClient([_chunk(content="done")])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))
        context = ChatContext(
            messages=[],
            scopes=[],
            tools=[TOOL_REGISTRY[0]],
            instructions=None,
        )

        _ = [
            event
            async for event in provider.run_chat_completion_stream("model-a", {}, context)
        ]

        assert client.chat.completions.kwargs is not None
        kwargs = client.chat.completions.kwargs
        self.assertIn("tools", kwargs)
        self.assertIs(kwargs["parallel_tool_calls"], False)

    async def test_streams_tool_call(self):
        client = FakeClient([_tool_chunk("bash", '{"command":'), _tool_chunk(None, ' "ls"}')])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        events = [
            event
            async for event in provider.run_chat_completion_stream(
                "model-a", {}, _context()
            )
        ]

        self.assertEqual(events[-1][1], FinishedToolCall(name="bash", parameters='{"command": "ls"}'))

    async def test_raises_on_empty_stop(self):
        client = FakeClient([_finish_chunk("stop")])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        with self.assertRaises(ValueError):
            _ = [
                event
                async for event in provider.run_chat_completion_stream(
                    "model-a", {}, _context()
                )
            ]

    async def test_flushes_text_before_rejecting_truncated_stream(self):
        client = FakeClient([_truncated_chunk("partial")])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))
        events = []

        with self.assertRaises(ValueError):
            async for event in provider.run_chat_completion_stream("model-a", {}, _context()):
                events.append(event)

        self.assertEqual(events[-1][1], FinishedMessage(content="partial"))

    async def test_reasoning_only_stop_is_not_an_empty_completion(self):
        client = FakeClient([_chunk(reasoning_content="thinking"), _finish_chunk("stop")])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))
        events = [
            event
            async for event in provider.run_chat_completion_stream("model-a", {}, _context())
        ]

        self.assertEqual(events[-1][1], FinishedThinking(content="thinking"))

    async def test_models_fall_back_to_configured_ids(self):
        client = FakeClient([], error=RuntimeError("offline"))
        provider = OpenAICompatibleProvider(_config(models=["model-a", "model-b"]), client=cast(Any, client))

        models = await provider.list_models()

        self.assertEqual([model.id for model in models], ["model-a", "model-b"])

    async def test_models_are_read_from_the_endpoint(self):
        client = FakeClient([], models=[SimpleNamespace(id="model-a", created=1, owned_by="remote")])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        models = await provider.list_models()

        self.assertEqual([model.id for model in models], ["model-a"])


class TestMessageConversion(unittest.TestCase):
    def test_reasoning_content_is_optional(self):
        context = _context([
            (1, Thinking(content="internal reasoning")),
            (2, Message(author="assistant", content="visible answer")),
        ])

        without = _to_oai_messages(context, include_reasoning_content=False)
        with_reasoning = _to_oai_messages(context, include_reasoning_content=True)

        self.assertNotIn("reasoning_content", without[1])
        self.assertEqual(with_reasoning[1]["reasoning_content"], "internal reasoning")


if __name__ == "__main__":
    unittest.main()
