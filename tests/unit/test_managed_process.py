import asyncio
import socket
import subprocess
import sys
import time
import unittest
from typing import cast

import inference.managed_processes as mp


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


def register_process(
    port: str,
    process: FakeProcess,
    *,
    last_used: float | None = None,
    active_requests: int = 0,
) -> mp.ManagedProcess:
    managed = mp.init_port(port, "test-provider", "test-model")
    managed.process = as_process(process)
    managed.last_used = time.monotonic() if last_used is None else last_used
    managed.active_requests = active_requests
    return managed


class TestManagedProcessLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await mp.stop_servers()

    async def asyncTearDown(self) -> None:
        await mp.stop_servers()

    async def test_idle_server_is_stopped_and_removed(self) -> None:
        process = FakeProcess()
        register_process(
            "1234",
            process,
            last_used=time.monotonic() - mp.IDLE_TIMEOUT_SECONDS - 1,
        )

        monitor = asyncio.create_task(mp._idle_monitor(check_interval=0.001))
        await asyncio.sleep(0.02)
        monitor.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await monitor

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertIsNone(mp.get_managed_process("1234"))

    async def test_active_server_is_not_stopped(self) -> None:
        process = FakeProcess()
        register_process(
            "1234",
            process,
            last_used=time.monotonic() - mp.IDLE_TIMEOUT_SECONDS - 1,
            active_requests=1,
        )

        monitor = asyncio.create_task(mp._idle_monitor(check_interval=0.001))
        await asyncio.sleep(0.02)
        monitor.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await monitor

        self.assertFalse(process.terminated)
        self.assertIsNotNone(mp.get_managed_process("1234"))

    async def test_activity_guard_refreshes_usage_and_releases_request(self) -> None:
        process = FakeProcess()
        managed = register_process("1234", process, last_used=0)

        before = time.monotonic()
        async with mp.track_server_activity("1234"):
            self.assertEqual(managed.active_requests, 1)
            self.assertGreaterEqual(managed.last_used, before)

        self.assertEqual(managed.active_requests, 0)
        self.assertGreaterEqual(managed.last_used, before)

    async def test_stop_servers_terminates_all_processes_and_clears_map(self) -> None:
        first = FakeProcess()
        second = FakeProcess()
        register_process("1234", first)
        register_process("2345", second)
        mp._idle_monitor_task = asyncio.create_task(asyncio.sleep(60))

        await mp.stop_servers()

        self.assertTrue(first.terminated)
        self.assertTrue(second.terminated)
        self.assertEqual(mp._managed_processes, {})
        self.assertIsNone(mp._idle_monitor_task)

    async def test_server_status_exposes_provider_and_model(self) -> None:
        process = FakeProcess()
        register_process(
            "1234",
            process,
            last_used=time.monotonic() - 4,
            active_requests=2,
        )

        status = mp.get_server_status()[0]

        self.assertEqual(status.port, "1234")
        self.assertEqual(status.provider, "test-provider")
        self.assertEqual(status.model, "test-model")
        self.assertEqual(status.pid, 1234)
        self.assertEqual(status.status, mp.ServerStatus.RUNNING)
        self.assertEqual(status.active_requests, 2)
        self.assertGreaterEqual(status.idle_seconds, 4)

    async def test_stop_one_server_removes_and_terminates_it(self) -> None:
        process = FakeProcess()
        register_process("1234", process)

        stopped = await mp.stop_server("1234")

        self.assertTrue(stopped)
        self.assertTrue(process.terminated)
        self.assertIsNone(mp.get_managed_process("1234"))
        self.assertFalse(await mp.stop_server("1234"))


class TestPortLifecycleRecord(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await mp.stop_servers()

    async def asyncTearDown(self) -> None:
        await mp.stop_servers()

    def test_init_port_returns_one_record_per_port(self) -> None:
        record = mp.init_port("5000", "provider", "model")
        same_record = mp.init_port("5000", "other provider", "other model")
        other_record = mp.init_port("5001", "provider", "model")

        self.assertIs(record, same_record)
        self.assertIsNot(record, other_record)
        self.assertIs(record.start_lock, same_record.start_lock)

    async def test_stop_wakes_readiness_waiter(self) -> None:
        record = mp.init_port("5000", "provider", "model")
        record.start_finished = asyncio.Event()
        waiter = asyncio.create_task(mp.wait_server_ready("5000"))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        self.assertTrue(await mp.stop_server("5000"))

        with self.assertRaises(RuntimeError):
            await waiter
        self.assertIsNone(mp.get_managed_process("5000"))

    async def test_reap_removes_exited_process(self) -> None:
        process = FakeProcess()
        process.returncode = -11
        managed = register_process("1234", process)

        await mp._reap_exited(managed)

        self.assertIsNone(mp.get_managed_process("1234"))

    async def test_reap_skips_process_started_meanwhile(self) -> None:
        process = FakeProcess()
        process.returncode = -11
        managed = register_process("1234", process)
        replacement = FakeProcess()
        managed.process = as_process(replacement)

        await mp._reap_exited(managed)

        self.assertIs(mp.get_managed_process("1234"), managed)
        self.assertFalse(replacement.terminated)

    async def test_waiter_returns_for_ready_process(self) -> None:
        register_process("5000", FakeProcess())
        await asyncio.wait_for(mp.wait_server_ready("5000"), timeout=1)

    async def test_concurrent_ensures_share_one_real_process(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])

        code = (
            "import socket,sys,time; "
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
            "s.bind(('127.0.0.1',int(sys.argv[1]))); s.listen(); "
            "time.sleep(60)"
        )

        async def ready() -> bool:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", int(port))
            except OSError:
                return False
            writer.close()
            await writer.wait_closed()
            return True

        start = asyncio.create_task(
            mp.ensure_server_started(
                port, "test", "python server", [sys.executable, "-c", code, port], ready
            )
        )
        await asyncio.sleep(0)
        waiter = asyncio.create_task(mp.wait_server_ready(port))
        second_start = asyncio.create_task(
            mp.ensure_server_started(
                port, "test", "python server", [sys.executable, "-c", code, port], ready
            )
        )

        await asyncio.gather(start, waiter, second_start)
        managed = mp.get_managed_process(port)
        assert managed is not None
        self.assertIsNotNone(managed.process)
        assert managed.start_finished is not None
        self.assertTrue(managed.start_finished.is_set())


if __name__ == "__main__":
    unittest.main()
