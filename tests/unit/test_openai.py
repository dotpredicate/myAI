import datetime
import unittest

from domain import Message, ScopeSpec, SecurityPolicy
from inference.engine import ChatContext
from inference.openai import _build_system_prompt


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
            "Available repositories:\n"
            "- /repositories/project (policy: read-only)\n\n"
            "Agent instructions:\nBe concise.",
        )

    def test_omits_empty_optional_sections(self):
        context = ChatContext(messages=[], scopes=[], tools=[], instructions=None)

        self.assertEqual(
            _build_system_prompt(context, today=datetime.date(2026, 1, 2)),
            "Current date: 2026-01-02",
        )


if __name__ == "__main__":
    unittest.main()
