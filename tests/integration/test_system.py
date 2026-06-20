import unittest
from pathlib import Path
from tests.helpers import BaseTestCase
from system import (
    run_sandboxed_command,
    get_repo_from_vpath,
    resolve_repo_vpath,
)
from domain import ScopeSpec, SecurityPolicy

class TestSystem(BaseTestCase):
    async def test_run_sandboxed_command_basic(self):
        result = await run_sandboxed_command("echo hello", scopes=[])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "hello")

    async def test_run_sandboxed_command_with_output(self):
        result = await run_sandboxed_command("printf 'line1\\nline2\\n'", scopes=[])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "line1\nline2\n")

    async def test_run_sandboxed_command_with_scopes(self):
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.WRITE)]
        result = await run_sandboxed_command("echo 'hello from scoped'", scopes=scopes)
        self.assertEqual(result.returncode, 0)

    async def test_run_sandboxed_command_readonly_scope(self):
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.READ_ONLY)]
        result = await run_sandboxed_command("echo 'hello from read-only'", scopes=scopes)
        self.assertEqual(result.returncode, 0)

    async def test_run_sandboxed_command_error(self):
        result = await run_sandboxed_command("invalid_command_that_does_not_exist", scopes=[])
        self.assertIsNotNone(result.returncode)
        self.assertNotEqual(result.returncode, 0)

    async def test_get_repo_from_vpath_valid(self):
        repo = await self._helper_create_repo("my_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        vpath = Path("/repositories/my_repo/src/main.py")
        result = await get_repo_from_vpath(vpath)
        self.assertIsNotNone(result)
        self.assertEqual(result.internal_name, "my_repo")

    async def test_get_repo_from_vpath_invalid(self):
        vpath = Path("/workspace/foo.py")
        result = await get_repo_from_vpath(vpath)
        self.assertIsNone(result)

    async def test_resolve_repo_vpath_valid(self):
        repo = await self._helper_create_repo("my_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        vpath = Path("/repositories/my_repo/src/main.py")
        resolved = await resolve_repo_vpath(vpath)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved, Path(self._repo_dir.name) / "src" / "main.py")

    async def test_resolve_repo_vpath_no_repo(self):
        vpath = Path("/repositories/nonexistent/foo.py")
        resolved = await resolve_repo_vpath(vpath)
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()