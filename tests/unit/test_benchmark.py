import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator, TypeVar
from unittest import mock

from backend.inference.llama_cpp import benchmark as benchmark_module
from backend.inference.llama_cpp.benchmark import (
    HeaderUnavailable,
    ModelQuantInfo,
    BenchmarkResult,
    RemoteFileInfo,
    _BenchEntry,
    _BenchEvent,
    _BenchProcessResult,
    _bench_cmd,
    _benchmark_one_stream,
    _depth_points,
    _extract_bench_error,
    _fetch_header,
    _header_cache_path,
    _load_header,
    _merge_depth_points,
    _model_info_result,
    _point_from_entry,
    _quant_info,
    _quant_info_from_header,
    _resolve_targets,
)
from backend.inference.llama_cpp.gguf_header import GgufHeader, parse_gguf_header
from tests.unit.test_gguf_header import _build_gguf


HEADER = GgufHeader(
    size=100,
    parameters=123,
    architecture="llama",
    context_length=8192,
    tensor_data_offset=0,
    bits_per_weight=None,
)


class TestModelInfoFormatting(unittest.TestCase):
    def test_quant_info_fields(self) -> None:
        r = _quant_info_from_header("Model-Q4_K_M.gguf", "Q4_K_M", HEADER)
        self.assertEqual(r["quant"], "Q4_K_M")
        self.assertEqual(r["filename"], "Model-Q4_K_M.gguf")
        self.assertEqual(r["size_bytes"], 100)
        bits_per_weight = r["bits_per_weight"]
        assert bits_per_weight is not None
        self.assertAlmostEqual(bits_per_weight, 100 * 8 / 123)
        self.assertEqual(r["parameters"], 123)
        self.assertEqual(r["architecture"], "llama")
        self.assertEqual(r["context_length"], 8192)

    def test_quant_info_keeps_exact_bits_per_weight(self) -> None:
        header = GgufHeader(
            size=100,
            parameters=123,
            architecture="llama",
            context_length=8192,
            tensor_data_offset=0,
            bits_per_weight=7.25,
        )

        r = _quant_info_from_header("Model-Q4_K_M.gguf", "Q4_K_M", header)

        self.assertEqual(r["bits_per_weight"], 7.25)

    def test_quant_info_error(self) -> None:
        with mock.patch(
            "backend.inference.llama_cpp.benchmark._load_header",
            new=mock.AsyncMock(side_effect=HeaderUnavailable("boom")),
        ):
            r = asyncio.run(_quant_info("repo", "Model-Q4_K_M.gguf", "Q4_K_M"))
        self.assertEqual(r["error"], "boom")

    def test_model_info_result_includes_all_quants(self) -> None:
        quants: list[ModelQuantInfo] = [
            _quant_info_from_header("a.gguf", "Q4_K_M", HEADER),
            _quant_info_from_header("b.gguf", "Q5_K_M", HEADER),
        ]
        data = _model_info_result("repo", quants)
        self.assertEqual(data["model_id"], "repo")
        self.assertEqual([q["quant"] for q in data["quants"]], ["Q4_K_M", "Q5_K_M"])

    def test_model_info_sorts_quants_by_bits_per_weight_then_size(self) -> None:
        def quant(name: str, size: int, parameters: int) -> ModelQuantInfo:
            header = GgufHeader(
                size=size,
                parameters=parameters,
                architecture="llama",
                context_length=1,
                tensor_data_offset=1,
                bits_per_weight=None,
            )
            return _quant_info_from_header(f"{name}.gguf", name, header)

        data = _model_info_result(
            "repo",
            [
                quant("small", 2_000, 2_000),
                {"quant": "BF16", "error": "unsupported"},
                quant("large", 4_000, 2_000),
                quant("same_bpw_bigger", 4_000, 4_000),
                quant("unknown", 9_000, 0),
            ],
        )

        self.assertEqual(
            [q["quant"] for q in data["quants"]],
            ["large", "same_bpw_bigger", "small", "unknown", "BF16"],
        )


def _header_bytes() -> tuple[bytes, int]:
    body, offset = _build_gguf()
    return body + b"\0" * (offset - len(body)), offset


class TestRemoteHeaderFetch(unittest.TestCase):
    def test_appends_only_missing_bytes(self) -> None:
        header_prefix, offset = _header_bytes()
        data = header_prefix + b"\0" * 4096
        calls: list[tuple[int, int]] = []

        async def fake_slice(repo_id: str, filename: str, start: int, length: int) -> bytes:
            calls.append((start, length))
            return data[start:start + length]

        info = RemoteFileInfo(size=len(data), blob_id="blob")
        with mock.patch.object(benchmark_module, "REMOTE_HEADER_INITIAL_BYTES", 8), \
             mock.patch.object(benchmark_module, "REMOTE_HEADER_MAX_BYTES", 1024), \
             mock.patch.object(benchmark_module, "download_file_slice", new=fake_slice):
            header, header_data = asyncio.run(_fetch_header("repo", "model.gguf", info))

        self.assertEqual(header.tensor_data_offset, offset)
        self.assertEqual(header_data, header_prefix)
        self.assertEqual(calls[0][0], 0)
        for (start, length), (next_start, _) in zip(calls, calls[1:]):
            self.assertEqual(start + length, next_start)


class TestHeaderCache(unittest.TestCase):
    def _info(self, offset: int) -> RemoteFileInfo:
        return RemoteFileInfo(size=offset + 100, blob_id="blob")

    @contextmanager
    def _cache(self, info: RemoteFileInfo, fetch: mock.AsyncMock) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(benchmark_module, "_HEADER_CACHE_DIR", Path(tmp)), \
             mock.patch.object(benchmark_module, "find_cached_gguf", return_value=None), \
             mock.patch.object(benchmark_module, "_remote_file_infos", new=mock.AsyncMock(return_value={"model.gguf": info})), \
             mock.patch.object(benchmark_module, "_fetch_header", new=fetch):
            yield Path(tmp)

    def test_second_load_reads_from_disk(self) -> None:
        header_bytes, offset = _header_bytes()
        info = self._info(offset)
        header = parse_gguf_header(header_bytes, file_size=info.size)
        fetch = mock.AsyncMock(return_value=(header, header_bytes))

        with self._cache(info, fetch):
            first_header, first_bytes = asyncio.run(_load_header("repo", "model.gguf"))
            second_header, second_bytes = asyncio.run(_load_header("repo", "model.gguf"))

        self.assertEqual(first_header, header)
        self.assertEqual(first_bytes, header_bytes)
        self.assertEqual(second_header, header)
        self.assertEqual(second_bytes, header_bytes)
        self.assertEqual(fetch.await_count, 1)

    def test_corrupt_cache_is_replaced(self) -> None:
        header_bytes, offset = _header_bytes()
        info = self._info(offset)
        header = parse_gguf_header(header_bytes, file_size=info.size)
        fetch = mock.AsyncMock(return_value=(header, header_bytes))

        with self._cache(info, fetch):
            cache_path = _header_cache_path("repo", "model.gguf", info)
            cache_path.write_bytes(b"not a gguf")

            loaded_header, loaded_bytes = asyncio.run(_load_header("repo", "model.gguf"))

            self.assertEqual(cache_path.read_bytes(), header_bytes)

        self.assertEqual(loaded_header, header)
        self.assertEqual(loaded_bytes, header_bytes)
        self.assertEqual(fetch.await_count, 1)

    def test_local_file_wins_over_network(self) -> None:
        header_bytes, offset = _header_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(header_bytes + b"\0" * 100)
            remote = mock.AsyncMock(side_effect=AssertionError("network used"))

            with mock.patch.object(benchmark_module, "find_cached_gguf", return_value=path), \
                 mock.patch.object(benchmark_module, "_remote_file_infos", new=remote):
                header, data = asyncio.run(_load_header("repo", "model.gguf"))

        self.assertEqual(header.tensor_data_offset, offset)
        self.assertEqual(data, header_bytes)


T = TypeVar("T")


async def _collect(iterator: AsyncIterator[T]) -> list[T]:
    return [item async for item in iterator]


class TestBenchmarkStream(unittest.TestCase):
    def test_yields_results_in_requested_order(self) -> None:
        targets = (
            "repo",
            [
                ("a.gguf", "A"),
                ("b.gguf", "B"),
                ("c.gguf", "C"),
            ],
        )

        async def fake_stream(repo_id: str, filename: str, quant: str, n_gen: int, n_prompt: int, repetitions: int, *, info: object = None) -> AsyncIterator[BenchmarkResult]:
            yield {"quant": quant, "points": [{"depth": 0, "tg_tps": 1.0}], "done": True}

        with mock.patch.object(benchmark_module, "_resolve_targets", return_value=targets), \
             mock.patch.object(benchmark_module, "_batch_file_infos", return_value={}), \
             mock.patch.object(benchmark_module, "save_benchmark_points", new=mock.AsyncMock()), \
             mock.patch.object(benchmark_module, "_benchmark_one_stream", new=fake_stream):
            results = asyncio.run(_collect(benchmark_module.benchmark_model_stream("repo", quants=["C", "A"])))

        self.assertEqual([result["quant"] for result in results], ["C", "A"])

    def test_reports_resolution_error(self) -> None:
        with mock.patch.object(benchmark_module, "_resolve_targets", side_effect=ValueError("nope")):
            results = asyncio.run(_collect(benchmark_module.benchmark_model_stream("repo")))

        self.assertEqual(results, [{"error": "Failed to resolve model: nope"}])

    def test_reports_empty_selection(self) -> None:
        targets = ("repo", [("a.gguf", "A")])
        with mock.patch.object(benchmark_module, "_resolve_targets", return_value=targets):
            results = asyncio.run(_collect(benchmark_module.benchmark_model_stream("repo", quants=["Z"])))

        self.assertEqual(results, [{"error": "No matching quants selected"}])


class TestMergeDepthPoints(unittest.TestCase):
    def test_keeps_failed_depths_as_empty_points(self) -> None:
        merged = _merge_depth_points([{"depth": 0, "tg_tps": 1.0}], [0, 1024, 4096])

        self.assertEqual([point["depth"] for point in merged], [0, 1024, 4096])
        self.assertNotIn("tg_tps", merged[1])
        self.assertNotIn("tg_tps", merged[2])


class TestResolveTargets(unittest.TestCase):
    def test_prefers_remote_quants_over_cache(self) -> None:
        with mock.patch.object(benchmark_module, "list_cached_quants", return_value=[("cached.gguf", "Q4_K_M")]), \
             mock.patch.object(benchmark_module, "list_gguf_quants", return_value=[("remote.gguf", "Q8_0")]):
            targets = _resolve_targets("repo")

        self.assertEqual(targets, ("repo", [("remote.gguf", "Q8_0")]))

    def test_falls_back_to_cache_when_remote_fails(self) -> None:
        with mock.patch.object(benchmark_module, "list_cached_quants", return_value=[("cached.gguf", "Q4_K_M")]), \
             mock.patch.object(benchmark_module, "list_gguf_quants", side_effect=OSError("offline")):
            targets = _resolve_targets("repo")

        self.assertEqual(targets, ("repo", [("cached.gguf", "Q4_K_M")]))

    def test_falls_back_to_cache_when_remote_is_empty(self) -> None:
        with mock.patch.object(benchmark_module, "list_cached_quants", return_value=[("cached.gguf", "Q4_K_M")]), \
             mock.patch.object(benchmark_module, "list_gguf_quants", return_value=[]):
            targets = _resolve_targets("repo")

        self.assertEqual(targets, ("repo", [("cached.gguf", "Q4_K_M")]))

    def test_raises_when_nothing_is_available(self) -> None:
        with mock.patch.object(benchmark_module, "list_cached_quants", return_value=[]), \
             mock.patch.object(benchmark_module, "list_gguf_quants", return_value=[]):
            with self.assertRaises(ValueError):
                _resolve_targets("repo")


class TestBenchmarkOne(unittest.TestCase):
    def test_split_model_fails_before_fetching_header(self) -> None:
        load = mock.AsyncMock(side_effect=AssertionError("header fetched"))
        with mock.patch.object(benchmark_module, "_load_header", new=load):
            results = asyncio.run(
                _collect(
                    _benchmark_one_stream(
                        "repo", "Model-BF16-00001-of-00002.gguf", "BF16", 1, 1, 1
                    )
                )
            )

        self.assertEqual(results[0]["error"], "Benchmarking split GGUF models is not supported")
        self.assertTrue(results[0]["done"])
        load.assert_not_awaited()

    def test_streams_partial_points_then_final(self) -> None:
        header = GgufHeader(
            size=100,
            parameters=10,
            architecture="llama",
            context_length=8192,
            tensor_data_offset=1,
            bits_per_weight=None,
        )
        events: list[_BenchEvent] = [
            {"entry": {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 100.0}, "vram": 1000},
            {"entry": {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 10.0}, "vram": 1000},
            {"process": _BenchProcessResult(0, "", "", {0: 1000})},
        ]

        async def fake_events(cmd: list[str]) -> AsyncIterator[_BenchEvent]:
            for event in events:
                yield event

        with mock.patch.object(
            benchmark_module, "_load_header", new=mock.AsyncMock(return_value=(header, b""))
        ), mock.patch.object(
            benchmark_module, "_build_synthetic", return_value=Path("/tmp/opencode/missing.gguf")
        ), mock.patch.object(
            benchmark_module, "_run_llama_bench_stream", new=fake_events
        ), mock.patch.object(
            benchmark_module,
            "time",
            mock.Mock(monotonic=mock.Mock(side_effect=[0.0, 0.040, 1.0])),
        ):
            results = asyncio.run(
                _collect(
                    _benchmark_one_stream("repo", "model.gguf", "Q4", 128, 512, 1)
                )
            )

        self.assertEqual(len(results), 3)
        self.assertFalse(results[0]["done"])
        self.assertFalse(results[1]["done"])
        self.assertTrue(results[2]["done"])
        self.assertEqual(len(results[0]["points"]), 1)
        self.assertNotIn("tg_tps", results[0]["points"][0])
        self.assertEqual(results[0]["points"][0]["duration_ms"], 40)
        self.assertEqual(len(results[1]["points"]), 1)
        self.assertIn("tg_tps", results[1]["points"][0])
        self.assertEqual(results[1]["points"][0]["duration_ms"], 1000)
        self.assertEqual(
            [point["depth"] for point in results[2]["points"]],
            [0, 1024, 4096, 8192],
        )


class TestPointFromEntry(unittest.TestCase):
    def test_parses_depth_and_kind(self) -> None:
        self.assertEqual(
            _point_from_entry({"n_prompt": 512, "n_gen": 0, "n_depth": 4096, "avg_ts": 100.0}),
            _BenchEntry(4096, "pp", 100.0),
        )
        self.assertEqual(
            _point_from_entry({"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 50.0}),
            _BenchEntry(0, "tg", 50.0),
        )

    def test_ignores_entries_without_work(self) -> None:
        self.assertIsNone(_point_from_entry({"n_prompt": 0, "n_gen": 0}))


class TestDepthPoints(unittest.TestCase):
    def test_grows_by_powers_of_four_up_to_full_context(self) -> None:
        self.assertEqual(_depth_points(262144), [0, 1024, 4096, 16384, 65536, 262144])

    def test_includes_small_context_verbatim(self) -> None:
        self.assertEqual(_depth_points(8192), [0, 1024, 4096, 8192])

    def test_non_positive_context(self) -> None:
        self.assertEqual(_depth_points(0), [0])




class TestBenchCommand(unittest.TestCase):
    def test_includes_depth_ladder_and_jsonl(self) -> None:
        cmd = _bench_cmd("model.gguf", 128, 512, [0, 1024, 4096], 1)

        self.assertEqual(cmd[cmd.index("-d") + 1], "0,1024,4096")
        self.assertEqual(cmd[cmd.index("-p") + 1], "512")
        self.assertEqual(cmd[cmd.index("-n") + 1], "128")
        self.assertEqual(cmd[cmd.index("-o") + 1], "jsonl")

    def test_uses_the_server_launch_flags(self) -> None:
        cmd = _bench_cmd("model.gguf", 128, 512, [0], 1)

        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")
        self.assertNotIn("--fit", cmd)


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

    def test_prefers_allocation_error(self) -> None:
        stderr = (
            "llama_bench: error: failed to run depth\n"
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 40000 MiB\n"
        )
        self.assertIn("alloc", _extract_bench_error(stderr).lower())

    def test_empty_stderr(self) -> None:
        self.assertEqual(_extract_bench_error(""), "unknown error")
