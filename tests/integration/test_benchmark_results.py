import tempfile
import unittest
from pathlib import Path

from backend import database, paths
from backend.inference.llama_cpp.benchmark import (
    load_all_benchmark_points,
    save_benchmark_points,
)


class TestBenchmarkResultsCache(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._myai_dir = tempfile.TemporaryDirectory()
        paths.use_dir(Path(self._myai_dir.name))
        await database.init_database()

    async def asyncTearDown(self) -> None:
        self._myai_dir.cleanup()

    async def test_upsert_overwrites_the_same_depth(self) -> None:
        await save_benchmark_points("repo", "Q4_K_M", [
            {"depth": 0, "tg_tps": 10.0, "pp_tps": 100.0, "vram_peak_bytes": 1000, "duration_ms": 900},
        ])
        await save_benchmark_points("repo", "Q4_K_M", [
            {"depth": 0, "tg_tps": 20.0, "pp_tps": 200.0, "vram_peak_bytes": 2000, "duration_ms": 700},
        ])

        points = (await load_all_benchmark_points("repo")).get("Q4_K_M", [])

        self.assertEqual(points, [
            {"depth": 0, "tg_tps": 20.0, "pp_tps": 200.0, "vram_peak_bytes": 2000, "duration_ms": 700},
        ])

    async def test_groups_by_quant_and_keeps_depth_order(self) -> None:
        await save_benchmark_points("repo", "Q4_K_M", [
            {"depth": 4096, "tg_tps": 4.0},
            {"depth": 0, "tg_tps": 1.0},
        ])
        await save_benchmark_points("repo", "Q8_0", [{"depth": 0, "tg_tps": 2.0}])

        by_quant = await load_all_benchmark_points("repo")

        self.assertEqual([point["depth"] for point in by_quant["Q4_K_M"]], [0, 4096])
        self.assertEqual([point["depth"] for point in by_quant["Q8_0"]], [0])
