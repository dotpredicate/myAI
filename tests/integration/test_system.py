import os
import unittest
from pathlib import Path
from tests.helpers import BaseTestCase
from system import (
    MAX_OUTPUT_CHARS,
    is_safe_realpath,
    is_safe_vpath,
    run_sandboxed_command,
    get_repo_from_vpath,
    resolve_repo_vpath,
)
from domain import ScopeSpec, SecurityPolicy


class TestSystem(BaseTestCase):
    @property
    def workspace_dir(self) -> Path:
        path = Path(self._workspace_dir.name) / "1"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def test_run_sandboxed_command_basic(self):
        result = await run_sandboxed_command("echo hello", scopes=[], conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "hello")

    async def test_run_sandboxed_command_with_output(self):
        result = await run_sandboxed_command("printf 'line1\\nline2\\n'", scopes=[], conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "line1\nline2\n")

    async def test_run_sandboxed_command_with_scopes(self):
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.WRITE)]
        result = await run_sandboxed_command("echo 'hello from scoped'", scopes=scopes, conversation_id=1)
        self.assertEqual(result.returncode, 0)

    async def test_run_sandboxed_command_has_empty_repositories_without_scopes(self):
        result = await run_sandboxed_command("ls -A /repositories", scopes=[], conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    async def test_run_sandboxed_command_lists_scoped_repositories(self):
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.READ_ONLY)]
        result = await run_sandboxed_command("ls /repositories", scopes=scopes, conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertIn("test_repo", result.stdout)

    async def test_run_sandboxed_command_accepts_trailing_slash_repository_path(self):
        (Path(self._repo_dir.name) / "note.txt").write_text("hello", encoding="utf-8")
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.READ_ONLY)]
        result = await run_sandboxed_command("ls /repositories/test_repo/", scopes=scopes, conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertIn("note.txt", result.stdout)

    async def test_run_sandboxed_command_readonly_scope(self):
        repo = await self._helper_create_repo("test_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        scopes = [ScopeSpec(internal_name="test_repo", security_policy=SecurityPolicy.READ_ONLY)]
        result = await run_sandboxed_command("echo 'hello from read-only'", scopes=scopes, conversation_id=1)
        self.assertEqual(result.returncode, 0)

    async def test_run_sandboxed_command_error(self):
        result = await run_sandboxed_command("invalid_command_that_does_not_exist", scopes=[], conversation_id=1)
        self.assertIsNotNone(result.returncode)
        self.assertNotEqual(result.returncode, 0)

    async def test_run_sandboxed_command_does_not_leak_host_environment(self):
        os.environ["MYAI_TEST_SECRET"] = "leaked"
        try:
            result = await run_sandboxed_command(
                "printf '%s' \"$MYAI_TEST_SECRET\"",
                scopes=[],
                conversation_id=1,
            )
        finally:
            os.environ.pop("MYAI_TEST_SECRET", None)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    async def test_run_sandboxed_command_sets_home_and_tmpdir(self):
        result = await run_sandboxed_command(
            "printf '%s\\n%s\\n' \"$HOME\" \"$TMPDIR\"; test -d /tmp && echo tmp-ok",
            scopes=[],
            conversation_id=1,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "/workspace\n/tmp\ntmp-ok\n")

    async def test_run_sandboxed_command_starts_in_workspace(self):
        result = await run_sandboxed_command("pwd", scopes=[], conversation_id=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "/workspace")

    async def test_run_sandboxed_command_truncates_large_output(self):
        result = await run_sandboxed_command(
            "python3 -c \"print('x' * 200000)\"",
            scopes=[],
            conversation_id=1,
        )
        self.assertEqual(result.returncode, 0)
        self.assertLessEqual(len(result.stdout), MAX_OUTPUT_CHARS + 200)
        self.assertIn("truncated", result.stdout)

    async def test_run_sandboxed_command_mounts_workspace(self):
        result = await run_sandboxed_command(
            "echo hello > /workspace/note.txt",
            scopes=[],
            conversation_id=1,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual((self.workspace_dir / "note.txt").read_text(encoding="utf-8").strip(), "hello")

    def test_is_safe_vpath_requires_exact_scope_segment(self):
        vroot = Path("/repositories")
        scopes = [ScopeSpec(internal_name="foo", security_policy=SecurityPolicy.READ_ONLY)]

        allowed, _ = is_safe_vpath(Path("/repositories/foo/bar.txt"), vroot, scopes)
        self.assertTrue(allowed)

        allowed, _ = is_safe_vpath(Path("/repositories/foo"), vroot, scopes)
        self.assertTrue(allowed)

        allowed, _ = is_safe_vpath(Path("/repositories/foobar/bar.txt"), vroot, scopes)
        self.assertFalse(allowed)

    def test_is_safe_vpath_rejects_empty_scope_list(self):
        allowed, _ = is_safe_vpath(
            Path("/repositories/foo/bar.txt"),
            Path("/repositories"),
            [],
        )
        self.assertFalse(allowed)

    def test_is_safe_vpath_without_scope_restriction(self):
        allowed, _ = is_safe_vpath(Path("/workspace/bar.txt"), Path("/workspace"))
        self.assertTrue(allowed)

    def test_is_safe_realpath_rejects_symlink_escape(self):
        outside = Path(self._repo_dir.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.workspace_dir / "link.txt"
        link.symlink_to(outside)

        self.assertFalse(is_safe_realpath(link, self.workspace_dir))
        self.assertTrue(is_safe_realpath(self.workspace_dir / "inside.txt", self.workspace_dir))

    async def test_get_repo_from_vpath_valid(self):
        repo = await self._helper_create_repo("my_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        vpath = Path("/repositories/my_repo/src/main.py")
        result = await get_repo_from_vpath(vpath)
        self.assertIsNotNone(result)
        self.assertEqual(result.internal_name, "my_repo")

    async def test_get_repo_from_vpath_accepts_trailing_slash(self):
        repo = await self._helper_create_repo("my_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        result = await get_repo_from_vpath(Path("/repositories/my_repo/"))
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

    async def test_resolve_repo_vpath_accepts_trailing_slash(self):
        repo = await self._helper_create_repo("my_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        resolved = await resolve_repo_vpath(Path("/repositories/my_repo/"))
        self.assertEqual(resolved, Path(self._repo_dir.name))

    async def test_resolve_repo_vpath_no_repo(self):
        vpath = Path("/repositories/nonexistent/foo.py")
        resolved = await resolve_repo_vpath(vpath)
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
