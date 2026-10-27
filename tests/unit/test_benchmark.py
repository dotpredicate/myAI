import unittest

from inference.llama_cpp.benchmark import (
    ModelQuantInfo,
    _extract_bench_error,
    _model_info_result,
    _quant_info_from_header,
    parse_llama_bench_output,
)


HEADER = {
    "tensor_data_offset": 0,
    "size": 100,
    "parameters": 123,
    "architecture": "llama",
    "context_length": 8192,
}


class TestModelInfoFormatting(unittest.TestCase):
    def test_quant_info_fields(self) -> None:
        r = _quant_info_from_header("Model-Q4_K_M.gguf", "Q4_K_M", HEADER)
        self.assertEqual(r["quant"], "Q4_K_M")
        self.assertEqual(r["filename"], "Model-Q4_K_M.gguf")
        self.assertEqual(r["size_bytes"], 100)
        self.assertEqual(r["parameters"], 123)
        self.assertEqual(r["architecture"], "llama")
        self.assertEqual(r["context_length"], 8192)

    def test_quant_info_error(self) -> None:
        r = _quant_info_from_header("Model-Q4_K_M.gguf", "Q4_K_M", {"error": "boom"})
        self.assertEqual(r["error"], "boom")

    def test_model_info_result_includes_all_quants(self) -> None:
        quants: list[ModelQuantInfo] = [
            _quant_info_from_header("a.gguf", "Q4_K_M", HEADER),
            _quant_info_from_header("b.gguf", "Q5_K_M", HEADER),
        ]
        data = _model_info_result("repo", quants)
        self.assertEqual(data["model_id"], "repo")
        self.assertEqual([q["quant"] for q in data["quants"]], ["Q4_K_M", "Q5_K_M"])


class TestParseLlamaBench(unittest.TestCase):
    def test_parse(self) -> None:
        data = [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 4126.6, "model_size": 16535223416, "model_n_params": 25233142046},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 83.5, "model_size": 16535223416, "model_n_params": 25233142046},
        ]
        r = parse_llama_bench_output(data)
        self.assertAlmostEqual(r["tps"], 83.5)
        self.assertAlmostEqual(r["pp_tps"], 4126.6)
        self.assertEqual(r["size_bytes"], 16535223416)
        self.assertEqual(r["parameters"], 25233142046)


class TestExtractBenchError(unittest.TestCase):
    def test_prefers_model_load_reason(self) -> None:
        stderr = (
            "llama_model_load: error loading model: unknown model architecture: 'flint'\n"
            "llama_model_load_from_file_impl: failed to load model\n"
            "llama_bench: error: failed to load model '/tmp/synthetic.gguf'\n"
        )
        self.assertEqual(_extract_bench_error(stderr), "unknown model architecture: 'flint'")

    def test_skips_generic_load_failure(self) -> None:
        stderr = "llama_bench: error: failed to load model '/tmp/synthetic.gguf'\n"
        self.assertIn("failed to load model", _extract_bench_error(stderr))

    def test_empty_stderr(self) -> None:
        self.assertEqual(_extract_bench_error(""), "unknown error")


if __name__ == "__main__":
    unittest.main()
