import json
import unittest
from types import SimpleNamespace
from typing import Any, cast

from backend.domain import (
    ApprovalRequest,
    BashInput,
    BashSuccess,
    InferenceProviderConfig,
    Message,
    Prompt,
    ScopeSpec,
    SearchHit,
    SearchSuccess,
    SpawnAgentSuccess,
    Thinking,
    ToolCall,
    ToolCallDecision,
    ToolError,
    ToolProposal,
)
from backend.inference.engine import ChatContext, FinishedMessage, FinishedThinking, FinishedToolCall, StreamingMessage
from backend.inference.openai_chat import DeltaProcessor, OpenAICompatibleProvider, _build_system_prompt, _to_oai_messages


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

    def test_reads_reasoning_field(self):
        processor = DeltaProcessor()
        processor.process(_chunk(content="", reasoning="thinking"))

        self.assertEqual(processor.flush(), [FinishedThinking(content="thinking")])

    def test_reads_reasoning_text_field(self):
        processor = DeltaProcessor()
        processor.process(_chunk(content="", reasoning_text="thinking"))

        self.assertEqual(processor.flush(), [FinishedThinking(content="thinking")])

    def test_prefers_reasoning_content_over_reasoning(self):
        processor = DeltaProcessor()
        processor.process(_chunk(reasoning_content="native", reasoning="openrouter"))

        self.assertEqual(processor.flush(), [FinishedThinking(content="native")])

    def test_ignores_reasoning_details(self):
        processor = DeltaProcessor()
        processor.process(_chunk(
            content="",
            reasoning="thinking",
            reasoning_details=[{"type": "reasoning.text", "text": "thinking"}],
        ))

        self.assertTrue(processor.has_reasoning_details)
        self.assertEqual(processor.flush(), [FinishedThinking(content="thinking")])

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

    def test_drops_whitespace_only_message(self):
        processor = DeltaProcessor()
        processor.process(_chunk(content=""))
        processor.process(_chunk(content="   "))

        self.assertEqual(processor.flush(), [])


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

        self.assertEqual([event[0].content for event in events if isinstance(event[0], StreamingMessage)], ["Hello", " world"])
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
        from backend.tools import TOOL_REGISTRY

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

    async def test_tool_call_then_empty_content_delta_is_not_an_empty_completion(self):
        # Some providers end the stream with finish_reason=stop after sending an
        # empty content delta next to the tool call. The tool call has already
        # been executed, so the stream must not be reported as empty.
        client = FakeClient([
            _tool_chunk("bash", '{"command": "ls"}'),
            _chunk(content=""),
            _finish_chunk("stop"),
        ])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        events = [
            event
            async for event in provider.run_chat_completion_stream(
                "model-a", {}, _context()
            )
        ]

        self.assertEqual(
            [finished for _, finished in events if finished is not None],
            [FinishedToolCall(name="bash", parameters='{"command": "ls"}')],
        )

    async def test_empty_content_in_tool_call_chunk_is_not_an_empty_completion(self):
        client = FakeClient([
            _chunk(content="", tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name="bash", arguments='{"command": "ls"}'))
            ]),
            _finish_chunk("stop"),
        ])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        events = [
            event
            async for event in provider.run_chat_completion_stream(
                "model-a", {}, _context()
            )
        ]

        self.assertEqual(
            [finished for _, finished in events if finished is not None],
            [FinishedToolCall(name="bash", parameters='{"command": "ls"}')],
        )

    async def test_empty_content_delta_before_tool_call_is_not_persisted(self):
        client = FakeClient([
            _chunk(content=""),
            _tool_chunk("bash", '{"command": "ls"}'),
            _finish_chunk("stop"),
        ])
        provider = OpenAICompatibleProvider(_config(), client=cast(Any, client))

        events = [
            event
            async for event in provider.run_chat_completion_stream(
                "model-a", {}, _context()
            )
        ]

        self.assertEqual(
            [finished for _, finished in events if finished is not None],
            [FinishedToolCall(name="bash", parameters='{"command": "ls"}')],
        )

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

    async def test_openrouter_reasoning_only_stop_is_not_an_empty_completion(self):
        client = FakeClient([_chunk(reasoning="thinking"), _finish_chunk("stop")])
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
            (2, Message(content="visible answer")),
        ])

        without = _to_oai_messages(context, include_reasoning_content=False)
        with_reasoning = _to_oai_messages(context, include_reasoning_content=True)

        self.assertNotIn("reasoning_content", without[1])
        self.assertEqual(cast(dict[str, Any], with_reasoning[1])["reasoning_content"], "internal reasoning")

    def test_a_prompt_is_the_user_turn(self):
        context = _context([
            (1, Prompt(content="what is this?")),
            (2, Thinking(content="internal reasoning")),
            (3, Message(content="visible answer")),
        ])

        converted = _to_oai_messages(context)

        self.assertEqual([message["role"] for message in converted], ["system", "user", "assistant"])
        self.assertEqual(cast(dict[str, Any], converted[2])["reasoning_content"], "internal reasoning")

    def test_a_prompt_drops_thinking_with_nothing_after_it(self):
        context = _context([
            (1, Prompt(content="first question")),
            (2, Thinking(content="internal reasoning")),
            (3, Prompt(content="second question")),
        ])

        converted = _to_oai_messages(context)

        self.assertNotIn("reasoning_content", converted[2])
        self.assertEqual([message["role"] for message in converted], ["system", "user", "user"])

    def test_tool_arguments_are_json_text(self):
        context = _context([
            (7, ToolCall(
                name="bash",
                arguments=BashInput(command="ls"),
                outcome=BashSuccess(returncode=0, stdout="ok", stderr=""),
            )),
        ])

        converted = _to_oai_messages(context)
        call = cast(dict[str, Any], converted[1])["tool_calls"][0]

        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "ls"})

    # The transcript keeps the typed outcome; the model should read it as text
    # instead of one JSON envelope per call.
    def test_bash_result_is_rendered_as_text(self):
        context = _context([
            (7, ToolCall(
                name="bash",
                arguments=BashInput(command="false"),
                outcome=BashSuccess(returncode=1, stdout="before\n", stderr="boom\n"),
            )),
        ])

        content = cast(dict[str, Any], _to_oai_messages(context)[2])["content"]

        self.assertEqual(content, "exit code: 1\nstdout:\nbefore\nstderr:\nboom")

    def test_search_results_are_rendered_as_text(self):
        context = _context([
            (7, ToolCall(
                name="search",
                arguments=None,
                outcome=SearchSuccess(results=[
                    SearchHit(document_id=1, file_path="backend/tools.py", chunk_index=3, cosine_distance=0.21, text="def run_tool_call(...)"),
                ]),
            )),
        ])

        content = cast(dict[str, Any], _to_oai_messages(context)[2])["content"]

        self.assertEqual(content, "1. backend/tools.py (chunk 3, distance 0.21)\ndef run_tool_call(...)")

    def test_error_and_rejection_are_text(self):
        error_context = _context([
            (7, ToolCall(
                name="bash",
                arguments=None,
                outcome=ToolError(message="Invalid arguments for bash: command: Field required"),
            )),
        ])
        rejection_context = _context([
            (7, ToolCall(
                name="apply_patch",
                arguments=None,
                outcome=ToolProposal(format="patch", target="/repositories/a/x.py", source="/workspace/d.diff", diff=""),
            )),
            (8, ApprovalRequest(original_message_id=7)),
            (9, ToolCallDecision(original_message_id=7, decision="reject", comment="too risky")),
        ])

        error_content = cast(dict[str, Any], _to_oai_messages(error_context)[2])["content"]
        rejection_content = cast(dict[str, Any], _to_oai_messages(rejection_context)[2])["content"]

        self.assertEqual(error_content, "Error: Invalid arguments for bash: command: Field required")
        self.assertEqual(rejection_content, "User rejected this tool call with comment: too risky")

    def test_spawn_agent_returns_the_child_answer_as_text(self):
        context = _context([
            (7, ToolCall(
                name="spawn_agent",
                arguments=None,
                outcome=SpawnAgentSuccess(
                    conversation_id=42,
                    agent="reviewer",
                    response="Looks fine.\n",
                    blocked=False,
                    blocking_message_id=None,
                ),
            )),
        ])

        content = cast(dict[str, Any], _to_oai_messages(context)[2])["content"]

        self.assertEqual(content, "Looks fine.")


class TestSystemPrompt(unittest.TestCase):
    def test_names_the_workspace(self):
        context = ChatContext(
            messages=[],
            scopes=[],
            tools=[],
            instructions=None,
            workspace_name="myAI",
        )

        prompt = _build_system_prompt(context)

        self.assertIn('You work in the workspace "myAI"', prompt)

    def test_lists_the_sources_of_the_workspace(self):
        context = ChatContext(
            messages=[],
            scopes=[
                ScopeSpec(internal_name="alpha", index_enabled=True),
                ScopeSpec(internal_name="beta"),
            ],
            tools=[],
            instructions=None,
            workspace_name="myAI",
        )

        prompt = _build_system_prompt(context)

        self.assertIn(
            "Available repositories:\n"
            "- /repositories/alpha (indexed)\n"
            "- /repositories/beta (not indexed)",
            prompt,
        )

    def test_says_when_the_workspace_has_no_sources(self):
        context = ChatContext(
            messages=[],
            scopes=[],
            tools=[],
            instructions=None,
            workspace_name="Scratch",
        )

        prompt = _build_system_prompt(context)

        self.assertIn("Available repositories:\n- none; this workspace has no sources", prompt)

    def test_keeps_agent_instructions(self):
        context = ChatContext(
            messages=[],
            scopes=[],
            tools=[],
            instructions="Be brief.",
            workspace_name="myAI",
        )

        prompt = _build_system_prompt(context)

        self.assertIn("Agent instructions:\nBe brief.", prompt)
