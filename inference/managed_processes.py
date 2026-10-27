import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import StrEnum
import socket
import subprocess
import time
from typing import Any, AsyncIterator

import httpx

from log_config import get_logger

logger = get_logger(__name__)


IDLE_TIMEOUT_SECONDS = 5 * 60
IDLE_CHECK_INTERVAL_SECONDS = 30


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class ManagedProcess:
    port: str
    provider: str
    model: str
    process: subprocess.Popen[Any] | None = None
    last_used: float = field(default_factory=time.monotonic)
    active_requests: int = 0
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    start_finished: asyncio.Event | None = None
    start_error: BaseException | None = None
    stopping: bool = False


class ServerStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"
    FAILED = "failed"


def _public_status(managed: ManagedProcess) -> ServerStatus:
    if managed.process is None:
        return ServerStatus.STOPPED
    if managed.process.poll() is not None:
        return ServerStatus.EXITED
    if managed.stopping:
        return ServerStatus.STOPPING
    if managed.start_error is not None:
        return ServerStatus.FAILED
    if managed.start_finished is not None and not managed.start_finished.is_set():
        return ServerStatus.STARTING
    return ServerStatus.RUNNING


# Records hold loop-bound primitives: clear them with stop_servers() before
# reusing this registry from another event loop.
_managed_processes: dict[str, ManagedProcess] = {}
_idle_monitor_task: asyncio.Task[None] | None = None


def init_port(port: str, provider: str, model: str) -> ManagedProcess:
    managed = _managed_processes.get(port)
    if managed is None:
        managed = ManagedProcess(port=port, provider=provider, model=model)
        _managed_processes[port] = managed
    return managed


def get_managed_process(port: str) -> ManagedProcess | None:
    return _managed_processes.get(port)


async def wait_server_ready(port: str) -> None:
    managed = _managed_processes.get(port)
    if managed is None:
        raise RuntimeError(f"Server on port {port} is not registered")

    if managed.start_finished is not None:
        await managed.start_finished.wait()
    if managed.start_error is not None:
        raise RuntimeError(f"Server on port {port} is not available") from managed.start_error
    if _public_status(managed) is not ServerStatus.RUNNING:
        raise RuntimeError(f"Server on port {port} is not ready")


@asynccontextmanager
async def track_server_activity(port: str) -> AsyncIterator[None]:
    managed = _managed_processes.get(port)
    if managed is None:
        raise RuntimeError(f"Server on port {port} is not registered")

    async with managed.start_lock:
        if _managed_processes.get(port) is not managed:
            raise RuntimeError(f"Server on port {port} is no longer registered")
        if _public_status(managed) is not ServerStatus.RUNNING:
            raise RuntimeError(f"Server on port {port} is not ready")
        managed.active_requests += 1
        managed.last_used = time.monotonic()

    try:
        yield
    finally:
        async with managed.start_lock:
            if _managed_processes.get(port) is managed:
                managed.active_requests -= 1
                managed.last_used = time.monotonic()


async def _stop_managed_process(managed: ManagedProcess) -> None:
    process = managed.process
    if process is None or process.poll() is not None:
        return

    logger.info("Terminating server process %s on port %s", process.pid, managed.port)
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


async def _forget(managed: ManagedProcess) -> None:
    managed.stopping = True
    managed.start_error = RuntimeError(f"Server on port {managed.port} was stopped")
    if managed.start_finished is not None:
        managed.start_finished.set()
    await _stop_managed_process(managed)
    managed.process = None
    del _managed_processes[managed.port]


async def stop_server(port: str, *, only_if_idle: bool = False) -> bool:
    while True:
        managed = _managed_processes.get(port)
        if managed is None:
            return False

        async with managed.start_lock:
            # A stale waiter may hold a record that was already replaced.
            if _managed_processes.get(port) is not managed:
                continue
            if only_if_idle and (
                managed.active_requests > 0
                or time.monotonic() - managed.last_used < IDLE_TIMEOUT_SECONDS
            ):
                return False
            await _forget(managed)
            return True


async def _reap_exited(managed: ManagedProcess) -> None:
    async with managed.start_lock:
        if _managed_processes.get(managed.port) is not managed:
            return
        process = managed.process
        if process is None or process.poll() is None:
            return
        await _forget(managed)


async def _idle_monitor(check_interval: float = IDLE_CHECK_INTERVAL_SECONDS) -> None:
    try:
        while True:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            for port, managed in list(_managed_processes.items()):
                process = managed.process
                if process is not None and process.poll() is not None:
                    await _reap_exited(managed)
                elif (
                    _public_status(managed) is ServerStatus.RUNNING
                    and managed.active_requests == 0
                    and now - managed.last_used >= IDLE_TIMEOUT_SECONDS
                ):
                    if await stop_server(port, only_if_idle=True):
                        logger.info("Stopped idle server process on port %s.", port)
    except asyncio.CancelledError:
        raise


async def _wait_for_server(
    port: str,
    timeout: int,
    readiness_check: Callable[[], Awaitable[bool]],
) -> None:
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


async def ensure_server_started(
    port: str,
    provider: str,
    model: str,
    cmd: list[str],
    readiness_check: Callable[[], Awaitable[bool]],
) -> None:
    """Start on demand. On failure the process is stopped and the record dropped."""
    while True:
        managed = init_port(port, provider, model)
        async with managed.start_lock:
            if _managed_processes.get(port) is not managed:
                # The record was stopped while this task waited for its lock.
                continue

            if managed.process is not None and managed.process.poll() is None:
                return

            managed.start_error = None
            managed.stopping = False
            managed.start_finished = asyncio.Event()
            managed.provider = provider
            managed.model = model
            managed.last_used = time.monotonic()
            process: subprocess.Popen[Any] | None = None

            try:
                logger.info("Starting server on port %s: %s", port, " ".join(cmd))
                process = subprocess.Popen(cmd)
                managed.process = process

                global _idle_monitor_task
                if _idle_monitor_task is None or _idle_monitor_task.done():
                    _idle_monitor_task = asyncio.create_task(_idle_monitor())

                await _wait_for_server(port, 60, readiness_check)
            except BaseException as error:
                managed.start_error = error
                try:
                    if process is not None:
                        await _stop_managed_process(managed)
                finally:
                    managed.process = None
                    managed.start_finished.set()
                    if _managed_processes.get(port) is managed:
                        _managed_processes.pop(port)
                raise
            else:
                managed.start_finished.set()
                return


@dataclass(frozen=True)
class ServerInfo:
    port: str
    provider: str
    model: str
    pid: int | None
    status: ServerStatus
    active_requests: int
    idle_seconds: int


def get_server_status() -> list[ServerInfo]:
    now = time.monotonic()
    return [
        ServerInfo(
            port=port,
            provider=managed.provider,
            model=managed.model,
            pid=managed.process.pid if managed.process is not None else None,
            status=_public_status(managed),
            active_requests=managed.active_requests,
            idle_seconds=max(0, int(now - managed.last_used)),
        )
        for port, managed in _managed_processes.items()
    ]


async def stop_servers() -> None:
    global _idle_monitor_task
    if _idle_monitor_task is not None:
        _idle_monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await _idle_monitor_task
        _idle_monitor_task = None

    await asyncio.gather(*(stop_server(port) for port in list(_managed_processes)))
