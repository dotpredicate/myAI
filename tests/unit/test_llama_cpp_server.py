import asyncio
import subprocess
import time
import unittest
from typing import cast

from inference import llama_cpp_server as module
from inference.engine import TokenPiece


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 1234
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self) -> int:
        return self.returncode or 0


def as_process(process: FakeProcess) -> subprocess.Popen[str]:
    return cast(subprocess.Popen[str], process)


class TestTokenizationResponse(unittest.TestCase):
    def test_parses_text_and_byte_pieces(self) -> None:
        result = module._parse_tokenization_response({
            "tokens": [
                {"id": 1, "piece": "hello"},
                {"id": 2, "piece": [195, 169]},
            ]
        })

        self.assertEqual(result, [TokenPiece(1, "hello"), TokenPiece(2, [195, 169])])

    def test_rejects_malformed_token(self) -> None:
        with self.assertRaises(ValueError):
            module._parse_tokenization_response({"tokens": [{"id": 1, "piece": [256]}]})


class TestLlamaCppServerIdleShutdown(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await module.stop_llama_servers()

    async def asyncTearDown(self) -> None:
        await module.stop_llama_servers()

    async def test_idle_server_is_stopped_and_removed(self) -> None:
        process = FakeProcess()
        module._server_processes["1234"] = module.ManagedServer(
            process=as_process(process),
            model="test/model",
            last_used=time.monotonic() - module.IDLE_TIMEOUT_SECONDS - 1,
        )

        monitor = asyncio.create_task(module._idle_monitor(check_interval=0.001))
        await asyncio.sleep(0.02)
        monitor.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await monitor

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertNotIn("1234", module._server_processes)

    async def test_active_server_is_not_stopped(self) -> None:
        process = FakeProcess()
        server = module.ManagedServer(
            process=as_process(process),
            model="test/model",
            last_used=time.monotonic() - module.IDLE_TIMEOUT_SECONDS - 1,
            active_requests=1,
        )
        module._server_processes["1234"] = server

        monitor = asyncio.create_task(module._idle_monitor(check_interval=0.001))
        await asyncio.sleep(0.02)
        monitor.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await monitor

        self.assertFalse(process.terminated)
        self.assertIn("1234", module._server_processes)

    async def test_activity_guard_refreshes_usage_and_releases_request(self) -> None:
        process = FakeProcess()
        server = module.ManagedServer(
            process=as_process(process),
            model="test/model",
            last_used=0,
        )
        module._server_processes["1234"] = server

        before = time.monotonic()
        async with module._track_server_activity("1234"):
            self.assertEqual(server.active_requests, 1)
            self.assertGreaterEqual(server.last_used, before)

        self.assertEqual(server.active_requests, 0)
        self.assertGreaterEqual(server.last_used, before)

    async def test_stop_servers_terminates_all_processes_and_clears_map(self) -> None:
        first = FakeProcess()
        second = FakeProcess()
        module._server_processes["1234"] = module.ManagedServer(
            process=as_process(first),
            model="test/first",
            last_used=time.monotonic(),
        )
        module._server_processes["2345"] = module.ManagedServer(
            process=as_process(second),
            model="test/second",
            last_used=time.monotonic(),
        )
        module._idle_monitor_task = asyncio.create_task(asyncio.sleep(60))

        await module.stop_llama_servers()

        self.assertTrue(first.terminated)
        self.assertTrue(second.terminated)
        self.assertEqual(module._server_processes, {})
        self.assertIsNone(module._idle_monitor_task)

    async def test_server_status_contains_model_and_activity(self) -> None:
        process = FakeProcess()
        module._server_processes["1234"] = module.ManagedServer(
            process=as_process(process),
            model="test/model",
            last_used=time.monotonic() - 4,
            active_requests=2,
        )

        status = module.get_llama_server_status()

        self.assertEqual(status[0]["port"], "1234")
        self.assertEqual(status[0]["model"], "test/model")
        self.assertEqual(status[0]["pid"], 1234)
        self.assertEqual(status[0]["status"], "running")
        self.assertEqual(status[0]["active_requests"], 2)
        self.assertGreaterEqual(cast(int, status[0]["idle_seconds"]), 4)

    async def test_stop_one_server_removes_and_terminates_it(self) -> None:
        process = FakeProcess()
        module._server_processes["1234"] = module.ManagedServer(
            process=as_process(process),
            model="test/model",
            last_used=time.monotonic(),
        )

        stopped = await module.stop_llama_server("1234")

        self.assertTrue(stopped)
        self.assertTrue(process.terminated)
        self.assertNotIn("1234", module._server_processes)
        self.assertFalse(await module.stop_llama_server("1234"))


if __name__ == "__main__":
    unittest.main()
