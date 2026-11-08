import datetime
import unittest

from domain import Message, ScopeSpec, SecurityPolicy, ToolCall
from inference.engine import ChatContext
from inference.openai_chat import ENVIRONMENT_PROMPT, _build_system_prompt, _to_oai_messages


class TestSystemPrompt(unittest.TestCase):
    def test_builds_labeled_prompt_sections_without_indentation(self):
        context = ChatContext(
            messages=[(1, Message(author="user", content="hello"))],
            scopes=[
                ScopeSpec(
                    internal_name="project",
                    security_policy=SecurityPolicy.READ_ONLY,
                )
            ],
            tools=[],
            instructions="  Be concise.  ",
        )

        prompt = _build_system_prompt(context, today=datetime.date(2026, 1, 2))

        self.assertEqual(
            prompt,
            "Current date: 2026-01-02\n\n"
            f"{ENVIRONMENT_PROMPT}\n\n"
            "Available repositories:\n"
            "- /repositories/project (policy: read-only)\n\n"
            "Agent instructions:\nBe concise.",
        )
        self.assertIn("bubblewrap (bwrap)", prompt)
        self.assertIn("/workspace", prompt)
        self.assertIn("no network access", prompt)

    def test_tool_results_and_arguments_are_not_truncated(self):
        # Truncating tool results here would corrupt the JSON envelope that the
        # sandbox produced; producers cap their own output instead.
        parameters = '{"command": "ls -R ."}'
        result = '{"returncode": 0, "stdout": "' + "x" * 1000 + '", "stderr": ""}'
        call = ToolCall(name="bash", parameters=parameters, result=result)
        context = ChatContext(messages=[(1, call)], scopes=[], tools=[], instructions=None)

        messages = _to_oai_messages(context)
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["content"], result)
        assistant = next(message for message in messages if message.get("role") == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], parameters)

    def test_omits_empty_optional_sections(self):
        context = ChatContext(messages=[], scopes=[], tools=[], instructions=None)

        self.assertEqual(
            _build_system_prompt(context, today=datetime.date(2026, 1, 2)),
            "Current date: 2026-01-02\n\n" + ENVIRONMENT_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
