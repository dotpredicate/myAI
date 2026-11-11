import unittest
from typing import cast

from huggingface_hub import HfApi

from inference.llama_cpp.gguf import (
    SplitInfo,
    get_gguf_split_info,
    list_gguf_quants,
    resolve_hf_alias,
)


class FakeApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id: str):
        return self._files


def fake_api(files: list[str]) -> HfApi:
    return cast(HfApi, FakeApi(files))


class TestListGgufQuants(unittest.TestCase):
    def test_dedupes_and_skips_non_model_files(self):
        api = fake_api([
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
        self.assertEqual(list_gguf_quants("org/repo", api=fake_api([])), [])

    def test_first_shard_represents_split_quant(self):
        api = fake_api([
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
            SplitInfo(prefix="Q4_K_M/Qwen-Q4_K_M", shard=1, shards=3, tag="Q4_K_M"),
        )

    def test_dynamic_quant_keeps_ud_prefix(self):
        self.assertEqual(
            get_gguf_split_info("Muse-Glimmer-30B-UD-Q4_K_XL.gguf").tag,
            "UD-Q4_K_XL",
        )

    def test_regular_and_dynamic_quant_are_not_deduplicated(self):
        api = fake_api([
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


class TestResolveHfAlias(unittest.TestCase):
    def test_skips_mmproj(self):
        api = fake_api(["mmproj-Model-F16.gguf", "Model-F16.gguf"])
        self.assertEqual(
            resolve_hf_alias("org/repo:F16", api=api),
            ("org/repo", "Model-F16.gguf"),
        )

    def test_prefers_regular_quant_over_dynamic_shard(self):
        api = fake_api([
            "UD-Q4_K_M/Qwen-UD-Q4_K_M-00001-of-00002.gguf",
            "Q4_K_M/Qwen-Q4_K_M.gguf",
        ])
        self.assertEqual(
            resolve_hf_alias("org/repo:Q4_K_M", api=api),
            ("org/repo", "Q4_K_M/Qwen-Q4_K_M.gguf"),
        )
