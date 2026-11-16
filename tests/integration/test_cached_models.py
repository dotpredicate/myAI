from unittest import mock

from backend.inference import managed_processes
from backend.inference.llama_cpp import benchmark as benchmark_module
from backend.inference.llama_cpp import router as router_module
from tests.helpers import BaseTestCase


class TestCachedModelsApi(BaseTestCase):
    async def test_lists_cached_models_with_sizes(self):
        entries = [
            {"id": "org/model:Q4_K_M", "repo": "org/model", "quant": "Q4_K_M", "size_bytes": 4096},
        ]
        with mock.patch.object(router_module, "list_cached_model_entries", return_value=entries):
            response = await self.client.get("/api/llama-cpp/cached-models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": entries})

    async def test_list_survives_a_broken_cache(self):
        with mock.patch.object(router_module, "list_cached_model_entries", side_effect=OSError("boom")):
            response = await self.client.get("/api/llama-cpp/cached-models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": []})

    async def test_delete_reports_freed_bytes(self):
        with mock.patch.object(router_module, "delete_cached_model", return_value=2048), \
                mock.patch.object(benchmark_module, "delete_benchmark_points", new=mock.AsyncMock()):
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel%3AQ4_K_M")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "deleted", "model": "org/model:Q4_K_M", "freed_bytes": 2048},
        )

    async def test_delete_unknown_model_is_not_found(self):
        with mock.patch.object(
            router_module, "delete_cached_model", side_effect=FileNotFoundError("not in the cache")
        ):
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel%3AQ4_K_M")

        self.assertEqual(response.status_code, 404)
        self.assertIn("not in the cache", response.json()["error"])

    async def test_delete_rejects_an_alias_without_quant(self):
        with mock.patch.object(
            router_module,
            "delete_cached_model",
            side_effect=ValueError("Expected a cached model alias like org/model:Q4_K_M"),
        ):
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel")

        self.assertEqual(response.status_code, 400)
        self.assertIn("alias", response.json()["error"])

    async def test_delete_refuses_while_a_server_loads_the_model(self):
        server = managed_processes.ServerInfo(
            port="8080",
            provider="llama.cpp",
            model="org/model:Q4_K_M",
            pid=1,
            status=managed_processes.ServerStatus.RUNNING,
            active_requests=0,
            idle_seconds=0,
            cpu_cores=0.0,
            rss_bytes=0,
            vram_bytes=0,
        )
        with mock.patch.object(router_module, "delete_cached_model") as delete, \
                mock.patch.object(managed_processes, "get_server_status", return_value=[server]):
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel%3AQ4_K_M")

        self.assertEqual(response.status_code, 409)
        self.assertIn("8080", response.json()["error"])
        delete.assert_not_called()

    async def test_delete_drops_benchmark_results(self):
        with mock.patch.object(router_module, "delete_cached_model", return_value=16), \
                mock.patch.object(benchmark_module, "delete_benchmark_points", new=mock.AsyncMock()) as drop:
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel%3AQ4_K_M")

        self.assertEqual(response.status_code, 200)
        drop.assert_awaited_once_with("org/model:Q4_K_M")

    async def test_delete_reports_a_filesystem_failure(self):
        with mock.patch.object(router_module, "delete_cached_model", side_effect=OSError("read-only")):
            response = await self.client.delete("/api/llama-cpp/cached-models/org%2Fmodel%3AQ4_K_M")

        self.assertEqual(response.status_code, 500)
        self.assertIn("read-only", response.json()["error"])
