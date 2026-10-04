import asyncio
import unittest
from unittest.mock import patch

from inference.benchmark import model_info, parse_llama_bench_output, _quant_info


HEADER = {
    "tensor_data_offset": 0,
    "size": 100,
    "parameters": 123,
    "architecture": "llama",
    "context_length": 8192,
}


class TestModelInfo(unittest.TestCase):
    def test_quant_info_fields(self):
        async def run():
            with patch("inference.benchmark._gguf_header", return_value=HEADER):
                return await _quant_info("org/repo", "Model-Q4_K_M.gguf", "Q4_K_M")

        r = asyncio.run(run())
        self.assertEqual(r["quant"], "Q4_K_M")
        self.assertEqual(r["filename"], "Model-Q4_K_M.gguf")
        self.assertEqual(r["size_bytes"], 100)
        self.assertEqual(r["parameters"], 123)
        self.assertEqual(r["architecture"], "llama")
        self.assertEqual(r["context_length"], 8192)

    def test_quant_info_error(self):
        async def run():
            with patch("inference.benchmark._gguf_header", return_value={"error": "boom"}):
                return await _quant_info("org/repo", "Model-Q4_K_M.gguf", "Q4_K_M")

        r = asyncio.run(run())
        self.assertEqual(r["error"], "boom")

    def test_model_info_gathers_all_quants(self):
        async def run():
            with (
                patch(
                    "inference.benchmark._resolve_targets",
                    return_value=[("repo", "a.gguf", "Q4_K_M"), ("repo", "b.gguf", "Q5_K_M")],
                ),
                patch("inference.benchmark._gguf_header", return_value=HEADER),
            ):
                return await model_info("repo")

        data = asyncio.run(run())
        self.assertEqual(data["model_id"], "repo")
        self.assertEqual([q["quant"] for q in data["quants"]], ["Q4_K_M", "Q5_K_M"])


class TestParseLlamaBench(unittest.TestCase):
    def test_parse(self):
        data = [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 4126.6, "model_size": 16535223416, "model_n_params": 25233142046},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 83.5, "model_size": 16535223416, "model_n_params": 25233142046},
        ]
        r = parse_llama_bench_output(data)
        self.assertAlmostEqual(r["tps"], 83.5)
        self.assertAlmostEqual(r["pp_tps"], 4126.6)
        self.assertEqual(r["size_bytes"], 16535223416)
        self.assertEqual(r["parameters"], 25233142046)


if __name__ == "__main__":
    unittest.main()
