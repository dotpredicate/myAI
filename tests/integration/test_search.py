import json
import unittest
from pathlib import Path

import database
from domain import ScopeSpec, SecurityPolicy
from repositories import get_repo_documents
from tests.helpers import BaseTestCase, DeterministicEmbeddingProvider
import search
from tools import run_semantic_search

class TestSearch(BaseTestCase):
    async def test_repo_documents_plain(self):
        repo_res = await self._helper_create_repo("search_repo", self._repo_dir.name, repo_type="plain")
        self.assertEqual(repo_res.status_code, 201)

        (Path(self._repo_dir.name) / "test.py").write_text("def foo(): pass")
        (Path(self._repo_dir.name) / "notes.txt").write_text("hello")

        docs = await get_repo_documents("search_repo")
        self.assertEqual(len(docs), 2)

    async def test_semantic_search_is_scoped_to_repository(self):
        first_repo_dir = Path(self._repo_dir.name)
        second_repo_dir = Path(self._workspace_dir.name) / "second_repo"
        second_repo_dir.mkdir()
        first_content = "content unique to first repository"
        second_content = "content unique to second repository"
        (first_repo_dir / "notes.txt").write_text(first_content, encoding="utf-8")
        (second_repo_dir / "notes.txt").write_text(second_content, encoding="utf-8")

        first_repo = await self._helper_create_repo("first_repo", str(first_repo_dir), security="read-only")
        second_repo = await self._helper_create_repo("firstXrepo", str(second_repo_dir), security="read-only")
        self.assertEqual(first_repo.status_code, 201)
        self.assertEqual(second_repo.status_code, 201)

        provider = DeterministicEmbeddingProvider()
        await search.synchronize(provider=provider)
        calls_after_first_sync = provider.embed_calls
        await search.synchronize(provider=provider)

        result = await run_semantic_search(
            "run_semantic_search",
            json.dumps({"prompt": "repository notes", "top_k": 10}),
            scopes=[ScopeSpec(internal_name="first_repo", security_policy=SecurityPolicy.READ_ONLY)],
            embedding_provider=provider,
        )
        payload = json.loads(result.result)

        self.assertEqual(provider.tokenize_calls, 2)
        self.assertEqual(provider.embed_calls, calls_after_first_sync + 1)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["file_path"], "first_repo/notes.txt")
        self.assertEqual(payload["results"][0]["text"], first_content)

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
