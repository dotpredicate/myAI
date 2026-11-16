import subprocess
import sys
import unittest
from unittest import mock

import backend.inference.process_metrics as process_metrics


class TestProcessMetrics(unittest.TestCase):
    def test_sample_usage_without_pids_skips_gpu_metrics(self) -> None:
        with mock.patch.object(
            process_metrics.gpu_metrics, "read_process_vram_bytes"
        ) as read:
            usage = process_metrics.sample_usage([])

        self.assertEqual(usage, {})
        read.assert_not_called()

    def test_sample_usage_reads_process_vram(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            with mock.patch.object(
                process_metrics.gpu_metrics,
                "read_process_vram_bytes",
                return_value=1234,
            ):
                usage = process_metrics.sample_usage([process.pid])
        finally:
            process.kill()
            process.wait()

        self.assertEqual(usage[process.pid].vram_bytes, 1234)
        self.assertGreater(usage[process.pid].rss_bytes or 0, 0)

    def test_sample_usage_tolerates_missing_gpu_metrics(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            with mock.patch.object(
                process_metrics.gpu_metrics,
                "read_process_vram_bytes",
                return_value=None,
            ):
                usage = process_metrics.sample_usage([process.pid])
        finally:
            process.kill()
            process.wait()

        self.assertIn(process.pid, usage)
        self.assertIsNone(usage[process.pid].vram_bytes)
        self.assertGreater(usage[process.pid].rss_bytes or 0, 0)

    def test_sample_usage_drops_samples_of_vanished_processes(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.01)"])
        try:
            with mock.patch.object(
                process_metrics.gpu_metrics,
                "read_process_vram_bytes",
                return_value=None,
            ):
                process_metrics.sample_usage([process.pid])
        finally:
            process.wait()

        with mock.patch.object(
            process_metrics.gpu_metrics, "read_process_vram_bytes", return_value=None
        ):
            process_metrics.sample_usage([process.pid])

        self.assertEqual(process_metrics._cpu_samples, {})
