import json
import subprocess
import unittest
from unittest import mock

import gpu_metrics


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(gpu_metrics.ROCM_SMI_ARGS, 0, stdout=stdout, stderr="")


class TestGpuMetrics(unittest.TestCase):
    def test_read_gpu_data_returns_rocm_smi_json(self) -> None:
        stdout = json.dumps({"card0": {"GPU use (%)": "12"}, "system": {"PID1": "x, 1, 1, 0, 0"}})

        with mock.patch.object(gpu_metrics.subprocess, "run", return_value=completed(stdout)):
            data = gpu_metrics.read_gpu_data()

        self.assertEqual(data, {"card0": {"GPU use (%)": "12"}, "system": {"PID1": "x, 1, 1, 0, 0"}})

    def test_read_gpu_data_tolerates_missing_rocm_smi(self) -> None:
        with mock.patch.object(
            gpu_metrics.subprocess, "run", side_effect=FileNotFoundError("rocm-smi")
        ):
            self.assertIsNone(gpu_metrics.read_gpu_data())

    def test_read_gpu_data_tolerates_failures_and_invalid_json(self) -> None:
        with mock.patch.object(
            gpu_metrics.subprocess, "run", return_value=completed("not json")
        ):
            self.assertIsNone(gpu_metrics.read_gpu_data())

        failed = subprocess.CompletedProcess(gpu_metrics.ROCM_SMI_ARGS, 1, stdout="{}", stderr="")
        with mock.patch.object(gpu_metrics.subprocess, "run", return_value=failed):
            self.assertIsNone(gpu_metrics.read_gpu_data())

        with mock.patch.object(gpu_metrics.subprocess, "run", return_value=completed("[1, 2]")):
            self.assertIsNone(gpu_metrics.read_gpu_data())
