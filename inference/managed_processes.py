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

from inference import process_metrics
from log_config import get_logger

logger = get_logger(__name__)


IDLE_TIMEOUT_SECONDS = 5 * 60
IDLE_CHECK_INTERVAL_SECONDS = 30
READINESS_POLL_INTERVAL_SECONDS = 1.0


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
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    start_task: asyncio.Task[None] | None = None
    stopping: bool = False


class ServerStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"


def _public_status(managed: ManagedProcess) -> ServerStatus:
    if managed.stopping:
        return ServerStatus.STOPPING
    if managed.process is None:
        return ServerStatus.STOPPED
    if managed.process.poll() is not None:
        return ServerStatus.EXITED
    if managed.start_task is not None and not managed.start_task.done():
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


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    # Start attempts are shared; a waiter may disappear before observing the
    # failure. Retrieve it here so asyncio does not warn about it.
    if not task.cancelled():
        task.exception()


async def _await_start(task: asyncio.Task[None], port: str) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        if task.cancelled():
            raise RuntimeError(f"Server on port {port} start was cancelled") from None
        raise


async def wait_server_ready(port: str) -> None:
    managed = _managed_processes.get(port)
    if managed is None:
        raise RuntimeError(f"Server on port {port} is not registered")

    task = managed.start_task
    if task is not None:
        await _await_start(task, port)
    if _public_status(managed) is not ServerStatus.RUNNING:
        raise RuntimeError(f"Server on port {port} is not ready")


@asynccontextmanager
async def track_server_activity(port: str) -> AsyncIterator[None]:
    managed = _managed_processes.get(port)
    if managed is None:
        raise RuntimeError(f"Server on port {port} is not registered")

    async with managed.state_lock:
        if _managed_processes.get(port) is not managed:
            raise RuntimeError(f"Server on port {port} is no longer registered")
        if _public_status(managed) is not ServerStatus.RUNNING:
            raise RuntimeError(f"Server on port {port} is not ready")
        managed.active_requests += 1
        managed.last_used = time.monotonic()

    try:
        yield
    finally:
        async with managed.state_lock:
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


async def stop_server(port: str, *, only_if_idle: bool = False) -> bool:
    while True:
        managed = _managed_processes.get(port)
        if managed is None:
            return False

        async with managed.state_lock:
            # A stale waiter may hold a record that was already replaced.
            if _managed_processes.get(port) is not managed:
                continue
            if only_if_idle and (
                managed.active_requests > 0
                or time.monotonic() - managed.last_used < IDLE_TIMEOUT_SECONDS
            ):
                return False
            managed.stopping = True
            task = managed.start_task

        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        await _stop_managed_process(managed)

        async with managed.state_lock:
            if _managed_processes.get(port) is managed:
                _managed_processes.pop(port, None)
        return True


async def _reap_exited(managed: ManagedProcess) -> None:
    async with managed.state_lock:
        if _managed_processes.get(managed.port) is not managed:
            return
        process = managed.process
        if process is None or process.poll() is None:
            return
        if managed.start_task is not None and not managed.start_task.done():
            # The start attempt observes the same exit and reports it.
            return
        _managed_processes.pop(managed.port, None)


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
    managed: ManagedProcess,
    readiness_check: Callable[[], Awaitable[bool]],
) -> None:
    while True:
        process = managed.process
        if process is None:
            raise RuntimeError(f"Server on port {managed.port} is not registered")

        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Server on port {managed.port} exited with code {exit_code} before becoming ready"
            )

        try:
            if await readiness_check():
                return
        except httpx.TransportError:
            pass

        await asyncio.sleep(READINESS_POLL_INTERVAL_SECONDS)


async def _start_attempt(
    managed: ManagedProcess,
    cmd: list[str],
    readiness_check: Callable[[], Awaitable[bool]],
) -> None:
    process: subprocess.Popen[Any] | None = None
    try:
        logger.info("Starting server on port %s: %s", managed.port, " ".join(cmd))
        process = subprocess.Popen(cmd)
        managed.process = process

        global _idle_monitor_task
        if _idle_monitor_task is None or _idle_monitor_task.done():
            _idle_monitor_task = asyncio.create_task(_idle_monitor())

        logger.info("Waiting for server on port %s to be ready...", managed.port)
        await _wait_for_server(managed, readiness_check)
        logger.info(
            "Server on port %s is ready after %.1fs.",
            managed.port,
            time.monotonic() - managed.last_used,
        )
    except asyncio.CancelledError:
        await _stop_managed_process(managed)
        managed.process = None
        raise
    except BaseException:
        await _stop_managed_process(managed)
        managed.process = None
        raise


async def ensure_server_started(
    port: str,
    provider: str,
    model: str,
    cmd: list[str],
    readiness_check: Callable[[], Awaitable[bool]],
) -> None:
    """Start on demand, sharing one attempt between all concurrent callers."""
    while True:
        managed = init_port(port, provider, model)

        async with managed.state_lock:
            if _managed_processes.get(port) is not managed:
                # The record was stopped while this task waited for its lock.
                continue
            if managed.stopping:
                raise RuntimeError(f"Server on port {port} is stopping")

            managed.provider = provider
            managed.model = model
            task = managed.start_task

            if task is not None and not task.done():
                # Join the start attempt that is already in flight.
                pass
            elif (
                task is not None
                and not task.cancelled()
                and task.exception() is None
                and managed.process is not None
                and managed.process.poll() is None
            ):
                managed.last_used = time.monotonic()
                return
            else:
                managed.stopping = False
                managed.last_used = time.monotonic()
                managed.start_task = asyncio.create_task(
                    _start_attempt(managed, cmd, readiness_check)
                )
                managed.start_task.add_done_callback(_consume_task_exception)
                task = managed.start_task

        try:
            await _await_start(task, port)
        except asyncio.CancelledError:
            raise
        except BaseException:
            async with managed.state_lock:
                if (
                    _managed_processes.get(port) is managed
                    and managed.start_task is task
                    and managed.process is None
                ):
                    _managed_processes.pop(port, None)
            raise

        async with managed.state_lock:
            if _managed_processes.get(port) is not managed:
                raise RuntimeError(f"Server on port {port} is no longer registered")
            if managed.process is None or managed.process.poll() is not None:
                # The process exited right after a successful readiness check.
                continue
            managed.last_used = time.monotonic()
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
    cpu_cores: float | None
    rss_bytes: int | None
    vram_bytes: int | None


def get_server_status() -> list[ServerInfo]:
    records = list(_managed_processes.items())
    live_pids = [
        managed.process.pid
        for _, managed in records
        if managed.process is not None and managed.process.poll() is None
    ]
    usage = process_metrics.sample_usage(live_pids)
    now = time.monotonic()
    statuses: list[ServerInfo] = []
    for port, managed in records:
        pid = managed.process.pid if managed.process is not None else None
        if pid is None:
            measured = process_metrics.ProcessUsage()
        else:
            measured = usage.get(pid, process_metrics.ProcessUsage())
        statuses.append(
            ServerInfo(
                port=port,
                provider=managed.provider,
                model=managed.model,
                pid=pid,
                status=_public_status(managed),
                active_requests=managed.active_requests,
                idle_seconds=max(0, int(now - managed.last_used)),
                cpu_cores=measured.cpu_cores,
                rss_bytes=measured.rss_bytes,
                vram_bytes=measured.vram_bytes,
            )
        )
    return statuses


async def stop_servers() -> None:
    global _idle_monitor_task
    if _idle_monitor_task is not None:
        _idle_monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await _idle_monitor_task
        _idle_monitor_task = None

    await asyncio.gather(*(stop_server(port) for port in list(_managed_processes)))
