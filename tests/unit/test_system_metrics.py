import unittest
from unittest import mock

import system_metrics


class TestGpuStats(unittest.TestCase):
    def test_gpu_stats_are_parsed_from_gpu_data(self) -> None:
        data = {
            "card0": {
                "GPU use (%)": "12",
                "VRAM Total Memory (B)": "34208743424",
                "VRAM Total Used Memory (B)": "1446617088",
            },
        }

        with mock.patch.object(system_metrics, "read_gpu_data", return_value=data):
            stats = system_metrics._read_gpu_stats()

        self.assertEqual(stats.vram_used_bytes, 1446617088)
        self.assertEqual(stats.vram_total_bytes, 34208743424)
        self.assertEqual(stats.utilization_percent, 12.0)

    def test_gpu_stats_are_unavailable_without_gpu_data(self) -> None:
        for data in (None, {}, {"card0": "unexpected"}):
            with mock.patch.object(system_metrics, "read_gpu_data", return_value=data):
                stats = system_metrics._read_gpu_stats()

            self.assertIsNone(stats.vram_used_bytes)
            self.assertIsNone(stats.vram_total_bytes)
            self.assertIsNone(stats.utilization_percent)


class TestCpuStats(unittest.TestCase):
    def test_cpu_usage_is_derived_from_consecutive_samples(self) -> None:
        system_metrics._previous_cpu = None

        with mock.patch.object(
            system_metrics, "_read_cpu_times", side_effect=[(10, 100), (30, 200)]
        ):
            first = system_metrics._read_cpu_stats()
            second = system_metrics._read_cpu_stats()

        system_metrics._previous_cpu = None
        self.assertIsNone(first.used_cores)
        self.assertGreater(first.max_cores, 0)
        assert second.used_cores is not None
        self.assertAlmostEqual(second.used_cores, 0.2 * second.max_cores)


class TestMemoryStats(unittest.TestCase):
    def test_memory_stats_are_read_from_proc_meminfo(self) -> None:
        stats = system_metrics._read_memory_stats()

        assert stats.total_bytes is not None
        assert stats.used_bytes is not None
        self.assertGreater(stats.total_bytes, 0)
        self.assertGreaterEqual(stats.used_bytes, 0)
        self.assertLessEqual(stats.used_bytes, stats.total_bytes)


class TestSystemStats(unittest.TestCase):
    def test_sample_system_stats_combines_all_sources(self) -> None:
        cpu = system_metrics.CpuStats(used_cores=1.5, max_cores=8)
        ram = system_metrics.MemoryStats(used_bytes=1024, total_bytes=2048)
        gpu = system_metrics.GpuStats(vram_used_bytes=None, vram_total_bytes=None, utilization_percent=None)

        with (
            mock.patch.object(system_metrics, "_read_cpu_stats", return_value=cpu),
            mock.patch.object(system_metrics, "_read_memory_stats", return_value=ram),
            mock.patch.object(system_metrics, "_read_gpu_stats", return_value=gpu),
        ):
            stats = system_metrics.sample_system_stats()

        self.assertEqual(stats, system_metrics.SystemStats(cpu=cpu, ram=ram, gpu=gpu))
