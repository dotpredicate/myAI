import unittest
from pathlib import Path

from scopes import UserScopeChoice, resolve_scope, resolve_scopes
from domain import AgentConfig, AgentRepositoryAccess, SecurityPolicy
from tests.helpers import BaseTestCase


def _agent(policy: SecurityPolicy | None) -> AgentConfig:
    return AgentConfig(
        id=1,
        display_name="Test Agent",
        internal_name="test_agent",
        description="",
        provider_key="test",
        model_id="test-model",
        inference_config={},
        repository_access=[
            AgentRepositoryAccess(
                repository_id=1,
                repository_internal_name="test_repo",
                security_policy_override=policy,
            )
        ],
    )


class TestScopeResolution(BaseTestCase):
    async def test_empty_request_scopes_expand_to_all_agent_repositories(self):
        first = await self._helper_create_repo("first_repo", self._repo_dir.name, security="read-only")
        second_dir = Path(self._myai_dir.name) / "second_repo"
        second_dir.mkdir()
        second = await self._helper_create_repo("second_repo", str(second_dir), security="read-only")
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)

        agent_res = await self.client.post("/api/agents", json={
            "display_name": "Test Agent",
            "internal_name": "test_agent",
            "description": "",
            "instructions": "Test",
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
            "inference_config": {},
            "repository_access": [
                {"repository_id": first.json()["id"]},
                {"repository_id": second.json()["id"]},
            ],
        })
        self.assertEqual(agent_res.status_code, 201, agent_res.text)

        scopes = await resolve_scopes("test_agent", [])
        self.assertEqual(
            {scope.internal_name for scope in scopes},
            {"first_repo", "second_repo"},
        )

        scopes = await resolve_scopes(
            "test_agent",
            [UserScopeChoice(internal_name="first_repo")],
        )
        self.assertEqual(
            [scope.internal_name for scope in scopes],
            ["first_repo", "second_repo"],
        )

    async def test_user_override_takes_precedence_over_agent_override(self):
        await self._helper_create_repo("test_repo", self._repo_dir.name, security="write")
        resolved = await resolve_scope(
            UserScopeChoice(
                internal_name="test_repo",
                security_policy_override=SecurityPolicy.WRITE,
            ),
            agent=_agent(SecurityPolicy.READ_ONLY),
        )

        self.assertEqual(resolved.security_policy, SecurityPolicy.WRITE)

    async def test_agent_override_takes_precedence_over_repository_policy(self):
        await self._helper_create_repo("test_repo", self._repo_dir.name, security="write")
        resolved = await resolve_scope(
            UserScopeChoice(internal_name="test_repo"),
            agent=_agent(SecurityPolicy.PRIVILEGED_WRITE),
        )

        self.assertEqual(resolved.security_policy, SecurityPolicy.PRIVILEGED_WRITE)

    async def test_repository_policy_is_used_when_no_override_exists(self):
        await self._helper_create_repo("test_repo", self._repo_dir.name, security="read-only")
        resolved = await resolve_scope(UserScopeChoice(internal_name="test_repo"))

        self.assertEqual(resolved.security_policy, SecurityPolicy.READ_ONLY)

    async def test_user_override_cannot_escalate_repository_policy(self):
        await self._helper_create_repo("test_repo", self._repo_dir.name, security="read-only")

        with self.assertRaises(ValueError):
            await resolve_scope(UserScopeChoice(
                internal_name="test_repo",
                security_policy_override=SecurityPolicy.WRITE,
            ))

    async def test_agent_override_cannot_escalate_repository_policy(self):
        await self._helper_create_repo("test_repo", self._repo_dir.name, security="read-only")

        with self.assertRaises(ValueError):
            await resolve_scope(
                UserScopeChoice(internal_name="test_repo"),
                agent=_agent(SecurityPolicy.WRITE),
            )


if __name__ == "__main__":
    unittest.main()
