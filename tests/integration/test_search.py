import json
import unittest
from pathlib import Path

import database
from domain import ScopeSpec, SecurityPolicy
from repositories import get_repo_documents
from tests.helpers import BaseTestCase, DeterministicEmbeddingProvider
import search
from tools import TOOL_REGISTRY, run_semantic_search


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

    async def test_clear_index_removes_documents_and_chunks(self):
        embedding = "[" + ",".join(["0"] * 768) + "]"
        async with database.mk_conn() as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (file_path, file_hash, content) "
                "VALUES (%s, %s, %s) RETURNING id",
                ("repo/notes.txt", "0" * 64, "indexed text"),
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
