import unittest

from conversation import UserScopeChoice, resolve_scope
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


if __name__ == "__main__":
    unittest.main()
