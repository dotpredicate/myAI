import subprocess
import sys
import unittest
from unittest import mock

import backend.inference.process_metrics as process_metrics


class TestProcessMetrics(unittest.TestCase):
    def test_parse_vram_by_pid_reads_rocm_smi_data(self) -> None:
        data = {
            "system": {
                "PID1234": "llama-server, 1, 688246784, 0, unknown",
                "PID2345": "my server with spaces, 2, 1048576, 0, 50",
            },
        }

        usage = process_metrics._parse_vram_by_pid(data)

        self.assertEqual(usage, {1234: 688246784, 2345: 1048576})

    def test_parse_vram_by_pid_ignores_invalid_data(self) -> None:
        self.assertEqual(process_metrics._parse_vram_by_pid({}), {})
        self.assertEqual(
            process_metrics._parse_vram_by_pid(
                {"system": {"PID1234": "llama-server, 1, unknown, 0, 0"}}
            ),
            {},
        )

    def test_sample_usage_without_pids_skips_rocm_smi(self) -> None:
        with mock.patch.object(process_metrics, "read_gpu_data") as read:
            usage = process_metrics.sample_usage([])

        self.assertEqual(usage, {})
        read.assert_not_called()

    def test_sample_usage_tolerates_missing_rocm_smi(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            with mock.patch.object(process_metrics, "read_gpu_data", return_value=None):
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
            with mock.patch.object(process_metrics, "read_gpu_data", return_value=None):
                process_metrics.sample_usage([process.pid])
        finally:
            process.wait()

        with mock.patch.object(process_metrics, "read_gpu_data", return_value=None):
            process_metrics.sample_usage([process.pid])

        self.assertEqual(process_metrics._cpu_samples, {})
