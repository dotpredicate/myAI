import unittest

from inference.hf_gguf import get_gguf_split_info, list_gguf_quants


class FakeApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id: str):
        return self._files


class TestListGgufQuants(unittest.TestCase):
    def test_dedupes_and_skips_non_model_files(self):
        api = FakeApi([
            "Model-Q4_K_M.gguf",
            "Model-Q5_K_M.gguf",
            "Model-Q4_K_M.gguf",
            "mmproj-Model-F16.gguf",
            "dflash-kquant.gguf",
            "README.md",
        ])
        self.assertEqual(
            list_gguf_quants("org/repo", api=api),
            [("Model-Q4_K_M.gguf", "Q4_K_M"), ("Model-Q5_K_M.gguf", "Q5_K_M")],
        )

    def test_empty_repo(self):
        self.assertEqual(list_gguf_quants("org/repo", api=FakeApi([])), [])

    def test_first_shard_represents_split_quant(self):
        api = FakeApi([
            "Q4_K_M/Qwen-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/Qwen-Q4_K_M-00002-of-00003.gguf",
            "Q4_K_M/Qwen-Q4_K_M-00003-of-00003.gguf",
            "Q5_K_M/Qwen-Q5_K_M-00001-of-00002.gguf",
            "Q5_K_M/Qwen-Q5_K_M-00002-of-00002.gguf",
        ])
        self.assertEqual(
            list_gguf_quants("org/repo", api=api),
            [
                ("Q4_K_M/Qwen-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
                ("Q5_K_M/Qwen-Q5_K_M-00001-of-00002.gguf", "Q5_K_M"),
            ],
        )

    def test_split_info_parses_shard_and_quant(self):
        self.assertEqual(
            get_gguf_split_info("Q4_K_M/Qwen-Q4_K_M-00001-of-00003.gguf"),
            {"prefix": "Q4_K_M/Qwen-Q4_K_M", "index": 1, "count": 3, "tag": "Q4_K_M"},
        )

    def test_dynamic_quant_keeps_ud_prefix(self):
        self.assertEqual(
            get_gguf_split_info("Muse-Glimmer-30B-UD-Q4_K_XL.gguf")["tag"],
            "UD-Q4_K_XL",
        )

    def test_regular_and_dynamic_quant_are_not_deduplicated(self):
        api = FakeApi([
            "Model-Q4_K_M.gguf",
            "Model-UD-Q4_K_M.gguf",
        ])
        self.assertEqual(
            list_gguf_quants("org/repo", api=api),
            [
                ("Model-Q4_K_M.gguf", "Q4_K_M"),
                ("Model-UD-Q4_K_M.gguf", "UD-Q4_K_M"),
            ],
        )


if __name__ == "__main__":
    unittest.main()