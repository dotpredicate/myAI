import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import subprocess
import time
from typing import Any, AsyncIterator

import httpx

from log_config import get_logger

logger = get_logger(__name__)


IDLE_TIMEOUT_SECONDS = 5 * 60
IDLE_CHECK_INTERVAL_SECONDS = 30


@dataclass
class ManagedServer:
    process: subprocess.Popen[Any]
    tag: str
    model: str
    last_used: float
    active_requests: int = 0


_server_processes: dict[str, ManagedServer] = {}
_idle_monitor_task: asyncio.Task[None] | None = None


@asynccontextmanager
async def track_server_activity(port: str) -> AsyncIterator[None]:
    server = _server_processes[port]
    server.active_requests += 1
    server.last_used = time.monotonic()

    try:
        yield
    finally:
        if _server_processes.get(port) is server:
            server.active_requests -= 1
            server.last_used = time.monotonic()


async def _stop_managed_server(port: str, server: ManagedServer) -> None:
    process = server.process
    logger.info("Terminating server process %s on port %s", process.pid, port)
    process.terminate()
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, process.wait),
            timeout=10,
        )
        logger.info("Process %s terminated gracefully.", process.pid)
    except asyncio.TimeoutError:
        logger.warning(
            "Process %s did not terminate within 10s, sending SIGKILL.",
            process.pid,
        )
        process.kill()
        await loop.run_in_executor(None, process.wait)
        logger.info("Process %s killed.", process.pid)


async def _idle_monitor(check_interval: float = IDLE_CHECK_INTERVAL_SECONDS) -> None:
    try:
        while True:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            for port, server in list(_server_processes.items()):
                if server.process.poll() is not None:
                    _server_processes.pop(port, None)
                elif server.active_requests == 0 and now - server.last_used >= IDLE_TIMEOUT_SECONDS:
                    _server_processes.pop(port, None)
                    await _stop_managed_server(port, server)
                    logger.info("Stopped idle server process on port %s.", port)
    except asyncio.CancelledError:
        raise

async def _wait_for_server(port: str, timeout: int, readiness_check: Callable[[], Awaitable[bool]]) -> None:
    logger.info("Waiting for server on port %s to be ready...", port)

    async def _poll() -> None:
        while True:
            try:
                if await readiness_check():
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

    try:
        await asyncio.wait_for(_poll(), timeout=timeout)
        logger.info("Server on port %s is ready.", port)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Server on port {port} did not start within {timeout} seconds.")

async def lazy_start_server(port: str, tag: str, model: str, cmd: list[str], start_lock: asyncio.Lock, start_event: asyncio.Event, readiness_check: Callable[[], Awaitable[bool]]) -> None:
    async with start_lock:
        existing = _server_processes.get(port)
        if existing is not None and existing.process.poll() is None:
            await start_event.wait()
            return
        if existing is not None:
            _server_processes.pop(port, None)

        try:
            logger.info("Starting server on port %s: %s", port, ' '.join(cmd))
            proc = subprocess.Popen(cmd)
            _server_processes[port] = ManagedServer(proc, tag, model, time.monotonic())
        except FileNotFoundError:
            logger.error("%s not found in PATH.", cmd[0])
            raise

        global _idle_monitor_task
        if _idle_monitor_task is None or _idle_monitor_task.done():
            _idle_monitor_task = asyncio.create_task(_idle_monitor())

        try:
            await _wait_for_server(port, 60, readiness_check)
        except Exception:
            _server_processes.pop(port, None)
            proc.terminate()
            raise
        start_event.set()


def get_server_status() -> list[dict[str, object]]:
    """Return the current status of all servers managed by this process."""
    now = time.monotonic()
    return [
        {
            "port": port,
            "tag": server.tag,
            "model": server.model,
            "pid": server.process.pid,
            "status": "running" if server.process.poll() is None else "exited",
            "active_requests": server.active_requests,
            "idle_seconds": max(0, int(now - server.last_used)),
        }
        for port, server in _server_processes.items()
    ]


async def stop_server(port: str) -> bool:
    """Stop one managed server. Return False when it is not registered."""
    server = _server_processes.pop(port, None)
    if server is None:
        return False
    await _stop_managed_server(port, server)
    return True

async def stop_servers() -> None:
    """Terminate all managed server processes concurrently.

    Each process gets up to 10 seconds to shut down gracefully (SIGTERM).
    If a process does not terminate within that time, it is killed with SIGKILL.
    """
    global _idle_monitor_task
    if _idle_monitor_task is not None:
        _idle_monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await _idle_monitor_task
        _idle_monitor_task = None

    servers = list(_server_processes.items())
    _server_processes.clear()
    tasks = [_stop_managed_server(port, server) for port, server in servers]
    await asyncio.gather(*tasks)
