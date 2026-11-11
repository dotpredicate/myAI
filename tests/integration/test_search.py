import json
import subprocess
import unittest
from pathlib import Path

import database
from domain import RepositoryConfig, ScopeSpec
from inference import EmbeddingInput
from repositories import delete_repository, get_repo_documents, scan_repo_documents
from tests.helpers import BaseTestCase, DeterministicEmbeddingProvider
import search
from tools import TOOL_REGISTRY, ToolContext, run_semantic_search


class _DeletingEmbeddingProvider(DeterministicEmbeddingProvider):
    def __init__(self, repo_name: str) -> None:
        super().__init__()
        self.repo_name = repo_name
        self.deleted = False

    async def embed(self, model: str, input: EmbeddingInput) -> list[list[float]]:
        if not self.deleted:
            self.deleted = True
            await delete_repository(self.repo_name)
        return await super().embed(model, input)


class TestSearchToolSchema(unittest.TestCase):
    def test_search_tool_uses_query_parameter(self):
        search_tool = next(tool for tool in TOOL_REGISTRY if tool["name"] == "search")
        schema = search_tool["schema"]["parameters"]
        self.assertIn("query", schema["properties"])
        self.assertNotIn("prompt", schema["properties"])
        self.assertEqual(schema["required"], ["query"])


class TestSearch(BaseTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._original_embedding_provider = search.embedding_provider
        self.embedding_provider = DeterministicEmbeddingProvider()
        search.embedding_provider = self.embedding_provider

    async def asyncTearDown(self):
        search.embedding_provider = self._original_embedding_provider
        search._sync_pending = False
        await super().asyncTearDown()

    async def _make_two_repos(self):
        first_repo_dir = Path(self._repo_dir.name)
        second_repo_dir = Path(self._myai_dir.name) / "second_repo"
        second_repo_dir.mkdir()
        (first_repo_dir / "notes.txt").write_text(
            "content unique to first repository", encoding="utf-8"
        )
        (second_repo_dir / "notes.txt").write_text(
            "content unique to second repository", encoding="utf-8"
        )

        first_repo = await self._helper_create_repo("first_repo", str(first_repo_dir))
        second_repo = await self._helper_create_repo("firstXrepo", str(second_repo_dir))
        self.assertEqual(first_repo.status_code, 201)
        self.assertEqual(second_repo.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        return first_repo.json(), second_repo.json(), self.embedding_provider


    async def test_repo_documents_plain(self):
        repo_res = await self._helper_create_repo("search_repo", self._repo_dir.name, repo_type="plain")
        self.assertEqual(repo_res.status_code, 201)

        (Path(self._repo_dir.name) / "test.py").write_text("def foo(): pass")
        (Path(self._repo_dir.name) / "notes.txt").write_text("hello")

        docs = await get_repo_documents("search_repo")
        self.assertEqual(len(docs), 2)

    async def test_semantic_search_is_scoped_to_repository(self):
        await self._make_two_repos()

        result = await run_semantic_search(
            "search",
            json.dumps({"query": "repository notes", "top_k": 10}),
            ToolContext(
                conversation_id=1,
                workspace_id=1,
                scopes=[ScopeSpec(internal_name="first_repo")],
            ),
        )
        payload = json.loads(result.result)

        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["file_path"], "first_repo/notes.txt")
        self.assertEqual(payload["results"][0]["text"], "content unique to first repository")

    async def test_semantic_search_empty_scopes_returns_no_results(self):
        _, _, provider = await self._make_two_repos()
        calls_before = provider.embed_calls

        self.assertEqual(
            await search.semantic_search("repository notes", 10, scopes=[], provider=provider),
            [],
        )
        self.assertEqual(provider.embed_calls, calls_before)

    async def test_run_semantic_search_empty_scopes_returns_no_results(self):
        await self._make_two_repos()

        result = await run_semantic_search(
            "search",
            json.dumps({"query": "repository notes", "top_k": 10}),
            ToolContext(conversation_id=1, workspace_id=1, scopes=[]),
        )
        self.assertEqual(json.loads(result.result)["results"], [])

    async def test_api_search_uses_workspace_sources(self):
        first_repo, second_repo, _ = await self._make_two_repos()
        resolved = await self.client.post("/api/workspaces/resolve", json={"path": first_repo["path"]})
        self.assertEqual(resolved.status_code, 200, resolved.text)
        workspace_id = resolved.json()["workspace"]["id"]
        added = await self.client.post(
            f"/api/workspaces/{workspace_id}/sources",
            json={"repository_id": second_repo["id"]},
        )
        self.assertEqual(added.status_code, 200, added.text)

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "workspace_id": workspace_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["title"] for item in response.json()["results"]},
            {"first_repo/notes.txt (0)", "firstXrepo/notes.txt (0)"},
        )

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": []})

    async def test_search_normalizes_scope_names(self):
        await self._make_two_repos()

        hits = await search.semantic_search(
            "repository notes", 10, scopes=[" first_repo/ "], provider=self.embedding_provider
        )
        self.assertEqual([hit.file_path for hit in hits], ["first_repo/notes.txt"])


    async def test_scan_repo_documents_marks_git_error(self):
        repo = RepositoryConfig(
            id=1,
            display_name="Not a git repo",
            internal_name="not_git",
            repo_type="git",
            path=self._repo_dir.name,
        )
        scan = await scan_repo_documents(repo)
        self.assertEqual(scan.status, "error")

    async def test_synchronize_prunes_git_tracked_file_deleted_from_worktree(self):
        repo_dir = Path(self._repo_dir.name)
        subprocess.run(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
        (repo_dir / "gone.txt").write_text("gone content", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "gone.txt"],
            check=True,
            capture_output=True,
        )
        response = await self._helper_create_repo(
            "git_delete_repo", str(repo_dir), repo_type="git"
        )
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        hits = await search.semantic_search(
            "gone content", 10, scopes=["git_delete_repo"], provider=self.embedding_provider
        )
        self.assertEqual(len(hits), 1)

        (repo_dir / "gone.txt").unlink()
        await search.synchronize(provider=self.embedding_provider)
        hits = await search.semantic_search(
            "gone content", 10, scopes=["git_delete_repo"], provider=self.embedding_provider
        )
        self.assertEqual(hits, [])

    async def test_sync_accepts_requests_while_another_sync_is_running(self):
        async with search._sync_lock:
            response = await self.client.post("/api/sync")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json(), {"status": "accepted"})

        response = await self.client.post("/api/sync")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"status": "accepted"})

    async def test_synchronize_removes_chunks_when_file_becomes_empty(self):
        repo_dir = Path(self._repo_dir.name)
        notes = repo_dir / "notes.txt"
        notes.write_text("some content", encoding="utf-8")
        response = await self._helper_create_repo("empty_file_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        hits = await search.semantic_search(
            "some content", 10, scopes=["empty_file_repo"], provider=self.embedding_provider
        )
        self.assertEqual(len(hits), 1)

        notes.write_text("", encoding="utf-8")
        await search.synchronize(provider=self.embedding_provider)
        hits = await search.semantic_search(
            "some content", 10, scopes=["empty_file_repo"], provider=self.embedding_provider
        )
        self.assertEqual(hits, [])
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM document_chunks "
                "WHERE document_id IN (SELECT id FROM documents WHERE repository_id = "
                "(SELECT id FROM repositories WHERE internal_name = 'empty_file_repo'))"
            )
            self.assertEqual((await cur.fetchone())[0], 0)

        self.assertEqual(
            await search.semantic_search(
                "some content", 10, scopes=["empty_file_repo"], provider=self.embedding_provider
            ),
            [],
        )

    async def test_synchronize_runs_again_when_a_sync_was_queued(self):
        calls: list[int] = []
        original = search._sync_repositories

        async def fake_sync(provider: object) -> None:
            calls.append(1)
            if len(calls) == 1:
                search._sync_pending = True

        search._sync_repositories = fake_sync
        try:
            await search.synchronize(provider=self.embedding_provider)
        finally:
            search._sync_repositories = original

        self.assertEqual(len(calls), 2)

    async def test_clear_index_removes_documents_and_chunks(self):
        repo = await self._helper_create_repo("clear_repo", self._repo_dir.name)
        self.assertEqual(repo.status_code, 201)
        repo_id = repo.json()["id"]
        from sqlite_vec import serialize_float32  # type: ignore[import-untyped]
        embedding = serialize_float32([0.0] * 768)
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (repository_id, file_path, file_hash, content) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (repo_id, "clear_repo/notes.txt", "0" * 64, "indexed text"),
            )
            row = await cur.fetchone()
            assert row is not None
            await cur.execute(
                "INSERT INTO document_chunks "
                "(document_id, chunk_index, chunk_text, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (row[0], 0, "indexed text", embedding),
            )
            await conn.commit()

        response = await self.client.delete("/api/search/index")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "cleared"})
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM documents")
            self.assertEqual((await cur.fetchone())[0], 0)
            await cur.execute("SELECT COUNT(*) FROM document_chunks")
            self.assertEqual((await cur.fetchone())[0], 0)
