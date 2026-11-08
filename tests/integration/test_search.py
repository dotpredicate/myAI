import json
import subprocess
import unittest
from pathlib import Path

import database
from domain import RepositoryConfig, ScopeSpec, SecurityPolicy
from inference import EmbeddingInput
from repositories import delete_repository, get_repo_documents, scan_repo_documents
from tests.helpers import BaseTestCase, DeterministicEmbeddingProvider
import search
from tools import TOOL_REGISTRY, run_semantic_search


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
        second_repo_dir = Path(self._workspace_dir.name) / "second_repo"
        second_repo_dir.mkdir()
        (first_repo_dir / "notes.txt").write_text(
            "content unique to first repository", encoding="utf-8"
        )
        (second_repo_dir / "notes.txt").write_text(
            "content unique to second repository", encoding="utf-8"
        )

        first_repo = await self._helper_create_repo(
            "first_repo", str(first_repo_dir), security="read-only"
        )
        second_repo = await self._helper_create_repo(
            "firstXrepo", str(second_repo_dir), security="read-only"
        )
        self.assertEqual(first_repo.status_code, 201)
        self.assertEqual(second_repo.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        return first_repo.json(), second_repo.json(), self.embedding_provider

    async def _make_agent(self, repo_ids):
        response = await self.client.post("/api/agents", json={
            "display_name": "Search Agent",
            "internal_name": "search_agent",
            "description": "",
            "instructions": "Search all accessible repositories",
            "provider_key": "mock_e2e",
            "model_id": "dummy_model",
            "inference_config": {},
            "repository_access": [{"repository_id": repo_id} for repo_id in repo_ids],
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

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
            conversation_id=1,
            scopes=[ScopeSpec(internal_name="first_repo", security_policy=SecurityPolicy.READ_ONLY)],
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
            conversation_id=1,
            scopes=[],
        )
        self.assertEqual(json.loads(result.result)["results"], [])

    async def test_api_search_uses_agent_scopes(self):
        first_repo, second_repo, _ = await self._make_two_repos()
        await self._make_agent([first_repo["id"], second_repo["id"]])

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "agent_id": "search_agent"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["title"] for item in response.json()["results"]},
            {"first_repo/notes.txt (0)", "firstXrepo/notes.txt (0)"},
        )

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "agent_id": "search_agent", "scopes": ["first_repo"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["title"] for item in response.json()["results"]},
            {"first_repo/notes.txt (0)", "firstXrepo/notes.txt (0)"},
        )

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "scopes": ["first_repo"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["title"] for item in response.json()["results"]],
            ["first_repo/notes.txt (0)"],
        )

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": []})

    async def test_api_search_rejects_unknown_agent(self):
        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "agent_id": "missing_agent"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_agent", response.json()["error"])

    async def _document_count(self, prefix: str) -> int:
        path_prefix = prefix + "/"
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM documents WHERE left(file_path, char_length(%s)) = %s",
                (path_prefix, path_prefix),
            )
            return (await cur.fetchone())[0]

    async def test_scan_repo_documents_marks_missing_path(self):
        repo = RepositoryConfig(
            id=1,
            display_name="Missing",
            internal_name="missing",
            repo_type="plain",
            path="/definitely/missing",
            security_policy=SecurityPolicy.READ_ONLY,
        )
        scan = await scan_repo_documents(repo)
        self.assertEqual(scan.status, "missing")
        self.assertEqual(scan.files, [])

    async def test_scan_repo_documents_marks_empty_repository_ok(self):
        repo = RepositoryConfig(
            id=1,
            display_name="Empty",
            internal_name="empty",
            repo_type="plain",
            path=self._repo_dir.name,
            security_policy=SecurityPolicy.READ_ONLY,
        )
        scan = await scan_repo_documents(repo)
        self.assertEqual(scan.status, "ok")
        self.assertEqual(scan.files, [])

    async def test_synchronize_ignores_non_indexable_files(self):
        repo_dir = Path(self._repo_dir.name)
        (repo_dir / "notes.txt").write_text("notes content", encoding="utf-8")
        (repo_dir / "blob.bin").write_bytes(b"\x00\x01")
        response = await self._helper_create_repo("extensions_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("extensions_repo"), 1)

    async def test_synchronize_prunes_removed_files(self):
        repo_dir = Path(self._repo_dir.name)
        (repo_dir / "keep.txt").write_text("keep content", encoding="utf-8")
        (repo_dir / "drop.txt").write_text("drop content", encoding="utf-8")
        response = await self._helper_create_repo("prune_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("prune_repo"), 2)

        (repo_dir / "drop.txt").unlink()
        await search.synchronize(provider=self.embedding_provider)

        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute("SELECT file_path FROM documents WHERE file_path LIKE 'prune_repo/%'")
            self.assertEqual([row[0] for row in await cur.fetchall()], ["prune_repo/keep.txt"])

    async def test_synchronize_does_not_prune_when_repository_path_is_missing(self):
        repo_dir = Path(self._repo_dir.name)
        (repo_dir / "keep.txt").write_text("keep content", encoding="utf-8")
        response = await self._helper_create_repo("missing_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("missing_repo"), 1)

        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE repositories SET path = %s WHERE internal_name = %s",
                ("/definitely/missing", "missing_repo"),
            )
            await conn.commit()

        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("missing_repo"), 1)

    async def test_deleting_repository_during_indexing_stops_the_repository(self):
        repo_dir = Path(self._repo_dir.name)
        (repo_dir / "notes.txt").write_text("race content", encoding="utf-8")
        response = await self._helper_create_repo("race_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        provider = _DeletingEmbeddingProvider("race_repo")
        await search.synchronize(provider=provider)

        self.assertTrue(provider.deleted)
        self.assertEqual(await self._document_count("race_repo"), 0)
        response = await self.client.get("/api/repositories")
        names = [repo["internal_name"] for repo in response.json()["repositories"]]
        self.assertNotIn("race_repo", names)

    async def test_deleting_repository_cascades_documents_and_chunks(self):
        repo_dir = Path(self._repo_dir.name)
        (repo_dir / "notes.txt").write_text("cascade content", encoding="utf-8")
        response = await self._helper_create_repo("cascade_repo", str(repo_dir))
        self.assertEqual(response.status_code, 201)

        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("cascade_repo"), 1)
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM documents WHERE file_path = 'cascade_repo/notes.txt'"
            )
            row = await cur.fetchone()
            self.assertIsNotNone(row)
            doc_id = row[0]
            await cur.execute(
                "SELECT count(*) FROM document_chunks WHERE document_id = %s", (doc_id,)
            )
            self.assertEqual((await cur.fetchone())[0], 1)

        response = await self.client.delete("/api/repositories/cascade_repo")
        self.assertEqual(response.status_code, 200)

        self.assertEqual(await self._document_count("cascade_repo"), 0)
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM document_chunks WHERE document_id = %s", (doc_id,)
            )
            self.assertEqual((await cur.fetchone())[0], 0)

        self.assertEqual(
            await search.semantic_search(
                "cascade", 10, scopes=["cascade_repo"], provider=self.embedding_provider
            ),
            [],
        )

    async def test_search_normalizes_scope_names(self):
        await self._make_two_repos()

        hits = await search.semantic_search(
            "repository notes", 10, scopes=[" first_repo/ "], provider=self.embedding_provider
        )
        self.assertEqual([hit.file_path for hit in hits], ["first_repo/notes.txt"])

        response = await self.client.post(
            "/api/search",
            json={"query": "repository notes", "top_k": 10, "scopes": [" first_repo/ "]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["title"] for item in response.json()["results"]],
            ["first_repo/notes.txt (0)"],
        )

    async def test_scan_repo_documents_marks_git_error(self):
        repo = RepositoryConfig(
            id=1,
            display_name="Not a git repo",
            internal_name="not_git",
            repo_type="git",
            path=self._repo_dir.name,
            security_policy=SecurityPolicy.READ_ONLY,
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
        self.assertEqual(await self._document_count("git_delete_repo"), 1)

        (repo_dir / "gone.txt").unlink()
        await search.synchronize(provider=self.embedding_provider)
        self.assertEqual(await self._document_count("git_delete_repo"), 0)

    async def test_sync_accepts_requests_while_another_sync_is_running(self):
        async with search._sync_lock:
            response = await self.client.post("/api/sync")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json(), {"status": "accepted"})

        response = await self.client.post("/api/sync")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"status": "accepted"})

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
        embedding = "[" + ",".join(["0"] * 768) + "]"
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
                "VALUES (%s, %s, %s, %s::vector)",
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


if __name__ == "__main__":
    unittest.main()
