import tempfile
import unittest
from pathlib import Path

from huggingface_hub import scan_cache_dir

from backend.inference.llama_cpp.gguf_cache import (
    delete_cached_model,
    list_cached_model_entries,
    list_cached_models,
)


class CacheFixture:
    """A HuggingFace cache with real blobs and snapshot symlinks."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def add(
        self,
        repo_id: str,
        revision: str,
        files: dict[str, bytes],
        ref: str | None = "main",
    ) -> Path:
        """Store ``files`` as one revision, keyed by their path in the snapshot."""
        repo_path = self.root / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo_path / "snapshots" / revision
        blob_dir = repo_path / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            blob = blob_dir / f"{len(data)}-{abs(hash((repo_id, revision, name)))}"
            blob.write_bytes(data)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(blob)
        if ref:
            ref_path = repo_path / "refs" / ref
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(revision)
        return snapshot


class TestListCachedModelEntries(unittest.TestCase):
    def test_reports_size_of_each_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CacheFixture(Path(tmp)).add(
                "org/model",
                "rev1",
                {
                    "Model-Q4_K_M.gguf": b"a" * 100,
                    "Model-Q8_0.gguf": b"b" * 400,
                    "Model-Q4_K_M-00001-of-00002.gguf": b"c" * 10,
                    "Model-Q4_K_M-00002-of-00002.gguf": b"d" * 20,
                },
            )

            entries = list_cached_model_entries(Path(tmp))

        self.assertEqual(
            entries,
            [
                {"id": "org/model:Q4_K_M", "repo": "org/model", "quant": "Q4_K_M", "size_bytes": 130},
                {"id": "org/model:Q8_0", "repo": "org/model", "quant": "Q8_0", "size_bytes": 400},
            ],
        )

    def test_same_ids_as_the_model_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CacheFixture(Path(tmp)).add(
                "org/model",
                "rev1",
                {"Model-Q4_K_M.gguf": b"x", "mmproj-Model-F16.gguf": b"y", "README.md": b"z"},
            )

            ids = [entry["id"] for entry in list_cached_model_entries(Path(tmp))]
            models = list_cached_models(Path(tmp))

        self.assertEqual(ids, models)
        self.assertEqual(ids, ["org/model:Q4_K_M"])

    def test_missing_cache_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_cached_model_entries(Path(tmp) / "nope"), [])


class TestDeleteCachedModel(unittest.TestCase):
    def test_deletes_only_the_chosen_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = CacheFixture(Path(tmp)).add(
                "org/model",
                "rev1",
                {"Model-Q4_K_M.gguf": b"a" * 100, "Model-Q8_0.gguf": b"b" * 400},
            )

            freed = delete_cached_model("org/model:Q4_K_M", Path(tmp))

            self.assertEqual(freed, 100)
            self.assertFalse((snapshot / "Model-Q4_K_M.gguf").exists())
            self.assertTrue((snapshot / "Model-Q8_0.gguf").exists())
            self.assertEqual([entry["id"] for entry in list_cached_model_entries(Path(tmp))], ["org/model:Q8_0"])

    def test_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CacheFixture(Path(tmp)).add("org/model", "rev1", {"Model-Q4_K_M.gguf": b"a" * 10})

            freed = delete_cached_model("org/model:q4_k_m", Path(tmp))

        self.assertEqual(freed, 10)

    def test_removes_repository_once_no_files_are_left(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = CacheFixture(Path(tmp)).add(
                "org/model", "rev1", {"Model-Q4_K_M.gguf": b"a" * 10}
            ).parent.parent

            delete_cached_model("org/model:Q4_K_M", Path(tmp))

            self.assertFalse(repo.exists())
            self.assertEqual(list_cached_models(Path(tmp)), [])
            self.assertEqual(scan_cache_dir(Path(tmp)).warnings, [])

    def test_keeps_blob_another_quant_still_uses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = CacheFixture(Path(tmp)).add(
                "org/model", "rev1", {"Model-Q4_K_M.gguf": b"a" * 10}
            )
            # Two quant names pointing at one blob, as a hard-linked download.
            shared = (snapshot / "Model-Q4_K_M.gguf").resolve()
            (snapshot / "Model-Q8_0.gguf").symlink_to(shared)

            freed = delete_cached_model("org/model:Q4_K_M", Path(tmp))

            self.assertTrue(shared.exists())
            self.assertFalse((snapshot / "Model-Q4_K_M.gguf").exists())
            self.assertEqual(freed, 0)
            self.assertEqual(
                [entry["id"] for entry in list_cached_model_entries(Path(tmp))],
                ["org/model:Q8_0"],
            )

    def test_keeps_revision_with_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = CacheFixture(Path(tmp)).add(
                "org/model",
                "rev1",
                {"Model-Q4_K_M.gguf": b"a" * 10, "config.json": b"{}"},
            )

            delete_cached_model("org/model:Q4_K_M", Path(tmp))

            self.assertTrue((snapshot / "config.json").exists())
            self.assertEqual(list_cached_models(Path(tmp)), [])
            self.assertEqual(scan_cache_dir(Path(tmp)).warnings, [])

    def test_cache_stays_readable_after_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CacheFixture(Path(tmp)).add(
                "org/model",
                "rev1",
                {"Model-Q4_K_M.gguf": b"a" * 10, "Model-Q8_0.gguf": b"b" * 20},
            )

            delete_cached_model("org/model:Q4_K_M", Path(tmp))

            info = scan_cache_dir(Path(tmp))
            self.assertEqual(info.warnings, [])
            self.assertEqual([repo.repo_id for repo in info.repos], ["org/model"])
            self.assertEqual(list(info.repos)[0].size_on_disk, 20)

    def test_unknown_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            CacheFixture(Path(tmp)).add("org/model", "rev1", {"Model-Q4_K_M.gguf": b"a" * 10})

            with self.assertRaises(FileNotFoundError):
                delete_cached_model("org/other:Q4_K_M", Path(tmp))
            with self.assertRaises(FileNotFoundError):
                delete_cached_model("org/model:Q5_K_M", Path(tmp))

    def test_rejects_alias_without_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                delete_cached_model("org/model", Path(tmp))
