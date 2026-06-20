import unittest
from pathlib import Path
from tests.helpers import BaseTestCase
from repositories import get_repo_documents

class TestSearch(BaseTestCase):
    async def test_repo_documents_plain(self):
        repo_res = await self._helper_create_repo("search_repo", self._repo_dir.name, repo_type="plain")
        self.assertEqual(repo_res.status_code, 201)

        (Path(self._repo_dir.name) / "test.py").write_text("def foo(): pass")
        (Path(self._repo_dir.name) / "notes.txt").write_text("hello")

        docs = await get_repo_documents("search_repo")
        self.assertEqual(len(docs), 2)


if __name__ == "__main__":
    unittest.main()