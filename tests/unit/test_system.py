import unittest
from pathlib import Path

from system import (
    is_safe_vpath,
    vpath_to_realpath,
)
from domain import ScopeSpec, SecurityPolicy


class TestSystem(unittest.TestCase):

    def test_is_safe_vpath_valid(self):
        vpath = Path("/repositories/test_repo/foo.py")
        expected_vroot = Path("/repositories")
        safe, err = is_safe_vpath(vpath, expected_vroot)
        self.assertTrue(safe)
        self.assertEqual(err, "")

    def test_is_safe_vpath_traversal(self):
        vpath = Path("/repositories/../etc/passwd")
        expected_vroot = Path("/repositories")
        safe, err = is_safe_vpath(vpath, expected_vroot)
        self.assertFalse(safe)
        self.assertIn("Path traversal", err)

    def test_is_safe_vpath_outside_vroot(self):
        vpath = Path("/etc/passwd")
        expected_vroot = Path("/repositories")
        safe, err = is_safe_vpath(vpath, expected_vroot)
        self.assertFalse(safe)
        self.assertIn("must be under", err)

    def test_is_safe_vpath_with_scopes_matching(self):
        vpath = Path("/repositories/my_repo/foo.py")
        expected_vroot = Path("/repositories")
        scopes = [ScopeSpec(internal_name="my_repo", security_policy=SecurityPolicy.WRITE)]
        safe, err = is_safe_vpath(vpath, expected_vroot, allowed_scopes=scopes)
        self.assertTrue(safe)
        self.assertEqual(err, "")

    def test_is_safe_vpath_with_scopes_no_match(self):
        vpath = Path("/repositories/other_repo/foo.py")
        expected_vroot = Path("/repositories")
        scopes = [ScopeSpec(internal_name="my_repo", security_policy=SecurityPolicy.WRITE)]
        safe, err = is_safe_vpath(vpath, expected_vroot, allowed_scopes=scopes)
        self.assertFalse(safe)
        self.assertIn("not in allowed scopes", err)

    def test_is_safe_vpath_with_scopes_empty(self):
        vpath = Path("/repositories/my_repo/foo.py")
        expected_vroot = Path("/repositories")
        safe, err = is_safe_vpath(vpath, expected_vroot, allowed_scopes=[])
        self.assertTrue(safe)
        self.assertEqual(err, "")

    def test_vpath_to_realpath(self):
        vpath = Path("/workspace/my_project/src/main.py")
        result = vpath_to_realpath(vpath, "/workspace", "/home/user/.myai/workspace")
        expected = Path("/home/user/.myai/workspace") / "my_project" / "src" / "main.py"
        self.assertEqual(result, expected)

    def test_vpath_to_realpath_no_leading_slash(self):
        vpath = Path("workspace/my_project/src/main.py")
        result = vpath_to_realpath(vpath, "/workspace", "/home/user/.myai/workspace")
        expected = Path("/home/user/.myai/workspace") / "my_project" / "src" / "main.py"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()