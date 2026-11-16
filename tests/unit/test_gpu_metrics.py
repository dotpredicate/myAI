import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import gpu_metrics


class TestGpuMetrics(unittest.TestCase):
    def _device(self, root: Path) -> Path:
        device = root / "card0" / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text("1000\n")
        (device / "mem_info_vram_used").write_text("250\n")
        (device / "gpu_busy_percent").write_text("42\n")
        return device

    def test_reads_vram_and_utilization_from_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            device = self._device(Path(tmp))
            with mock.patch.object(gpu_metrics, "_GPU_DEVICE", device):
                self.assertEqual(gpu_metrics.read_vram_total_bytes(), 1000)
                self.assertEqual(gpu_metrics.read_vram_used_bytes(), 250)
                self.assertEqual(gpu_metrics.read_gpu_utilization_percent(), 42.0)

    def test_handles_missing_gpu(self) -> None:
        with mock.patch.object(gpu_metrics, "_GPU_DEVICE", None):
            self.assertIsNone(gpu_metrics.read_vram_total_bytes())
            self.assertIsNone(gpu_metrics.read_vram_used_bytes())
            self.assertIsNone(gpu_metrics.read_gpu_utilization_percent())

    def test_reads_process_vram_from_fdinfo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdinfo = Path(tmp) / "123" / "fdinfo"
            fdinfo.mkdir(parents=True)
            (fdinfo / "3").write_text(
                "drm-pdev:\t0000:03:00.0\n"
                "drm-client-id:\t43\n"
                "drm-memory-vram:\t1000 KiB\n"
                "drm-memory-gtt:\t4 KiB\n"
            )
            (fdinfo / "4").write_text(
                "drm-pdev:\t0000:03:00.0\n"
                "drm-client-id:\t43\n"
                "drm-memory-vram:\t1000 KiB\n"
            )
            (fdinfo / "5").write_text(
                "drm-pdev:\t0000:03:00.0\n"
                "drm-client-id:\t44\n"
                "drm-memory-vram:\t2 MiB\n"
            )
            with mock.patch.object(gpu_metrics, "_fdinfo_dir", return_value=fdinfo):
                vram = gpu_metrics.read_process_vram_bytes(123)

        self.assertEqual(vram, 1000 * 1024 + 2 * 1024**2)

    def test_process_vram_falls_back_to_fd_without_client_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdinfo = Path(tmp)
            (fdinfo / "3").write_text("drm-memory-vram:\t1000 KiB\n")
            (fdinfo / "4").write_text("drm-memory-vram:\t1000 KiB\n")
            with mock.patch.object(gpu_metrics, "_fdinfo_dir", return_value=fdinfo):
                vram = gpu_metrics.read_process_vram_bytes(123)

        self.assertEqual(vram, 2000 * 1024)

    def test_process_vram_is_none_without_drm_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdinfo = Path(tmp)
            (fdinfo / "3").write_text("other: 1\n")
            with mock.patch.object(gpu_metrics, "_fdinfo_dir", return_value=fdinfo):
                self.assertIsNone(gpu_metrics.read_process_vram_bytes(123))

    def test_process_vram_is_none_for_missing_process(self) -> None:
        self.assertIsNone(gpu_metrics.read_process_vram_bytes(999999))
