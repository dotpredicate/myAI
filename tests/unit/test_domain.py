import unittest
from domain import SecurityPolicy, scope_policy_is_escalation

class TestDomain(unittest.TestCase):
    def test_scope_policy_is_escalation(self):
        self.assertTrue(scope_policy_is_escalation(SecurityPolicy.WRITE, SecurityPolicy.READ_ONLY))
        self.assertFalse(scope_policy_is_escalation(SecurityPolicy.READ_ONLY, SecurityPolicy.WRITE))
        self.assertTrue(scope_policy_is_escalation(SecurityPolicy.PRIVILEGED_WRITE, SecurityPolicy.READ_ONLY))
        self.assertTrue(scope_policy_is_escalation(SecurityPolicy.WRITE, SecurityPolicy.PRIVILEGED_WRITE))
        self.assertFalse(scope_policy_is_escalation(None, SecurityPolicy.READ_ONLY))

    def test_tool_call_finished_or_blocked_with_scopes(self):
        from domain import ScopeSpec, ToolCallFinishedOrBlocked, stored_element_adapter
        scopes = [
            ScopeSpec(internal_name="repo1", security_policy=SecurityPolicy.READ_ONLY),
            ScopeSpec(internal_name="repo2", security_policy=SecurityPolicy.WRITE)
        ]
        elem = ToolCallFinishedOrBlocked(
            name="test_tool",
            parameters='{"p": 1}',
            result="{}",
            is_blocking=True,
            status="pending",
            scopes=scopes
        )
        
        # Test model dump
        dumped = elem.model_dump()
        self.assertEqual(dumped['scopes'], [
            {"internal_name": "repo1", "security_policy": "read-only"},
            {"internal_name": "repo2", "security_policy": "write"}
        ])
        
        # Test deserialization
        parsed = stored_element_adapter.validate_python(dumped)
        self.assertEqual(parsed.name, "test_tool")
        self.assertEqual(parsed.scopes[0].internal_name, "repo1")
        self.assertEqual(parsed.scopes[1].security_policy, SecurityPolicy.WRITE)

    def test_tool_call_finished_or_blocked_no_scopes(self):
        from domain import ToolCallFinishedOrBlocked, stored_element_adapter
        elem = ToolCallFinishedOrBlocked(
            name="test_tool",
            parameters='{"p": 1}',
            result="{}",
            is_blocking=False,
            status="completed"
        )
        self.assertEqual(elem.scopes, [])
        
        dumped = elem.model_dump()
        parsed = stored_element_adapter.validate_python(dumped)
        self.assertEqual(parsed.scopes, [])

    def test_security_policy_values(self):
        self.assertEqual(SecurityPolicy.READ_ONLY, 'read-only')
        self.assertEqual(SecurityPolicy.PRIVILEGED_WRITE, 'privileged-write')
        self.assertEqual(SecurityPolicy.WRITE, 'write')

if __name__ == '__main__':
    unittest.main()
