# Implementation Plan

## Overview

Automatically stop each locally managed llama.cpp server instance after five minutes without use.

The implementation keeps the existing port-keyed `_server_processes` map as the single source of truth for chat and embedding servers. Each entry tracks its subprocess, the time it was last used, and the number of active operations. A small background task periodically stops entries that have been idle for at least five minutes, while active streaming or embedding/tokenization requests remain protected.

The existing application shutdown hook will continue to call `stop_llama_servers`; that function will also stop the idle-monitor task before terminating remaining subprocesses.

## Types

- Add a `ManagedServer` dataclass in `/home/predicate/dev/myai/inference/llama_cpp_server.py` with:
  - `process: subprocess.Popen` — the managed llama.cpp process.
  - `last_used: float` — monotonic timestamp of the latest activity.
- `active_requests: int` — currently running operations using the process.
- Change `_server_processes` to `dict[str, ManagedServer]`.
- Add module constants for the five-minute idle timeout and monitor polling interval.

## Files

- `/home/predicate/dev/myai/implementation_plan.md` — this approved implementation plan.
- `/home/predicate/dev/myai/inference/llama_cpp_server.py` — add process activity tracking, idle monitoring, safe idle shutdown, and request activity guards for chat and embedding servers.
- `/home/predicate/dev/myai/tests/unit/test_llama_cpp_server.py` — add isolated unit tests for idle cleanup, active-request protection, activity refresh, and monitor shutdown behavior.

No dependencies, database migrations, API routes, or frontend changes are required.

## Functions

- Modify `_lazy_start_server` to register `ManagedServer` entries, initialize activity timestamps, and start the idle monitor when needed.
- Add an async activity context manager that increments `active_requests` on entry, refreshes `last_used`, and decrements the count on exit.
- Add an idle-monitor coroutine that periodically scans `_server_processes`, removes exited processes, and terminates processes that have been idle for five minutes with no active requests.
- Add a helper for terminating one managed process and reuse it from idle cleanup and `stop_llama_servers`.
- Modify `stop_llama_servers` to cancel the monitor, clear the map, and stop all remaining managed processes.
- Modify `LlamaCppServerProvider.run_chat_completion_stream` to hold an activity guard for the complete streamed request.
- Modify `LlamaCppEmbeddingServer.embed` and `tokenize` to hold an activity guard for their complete requests.

## Classes

- Add the `ManagedServer` dataclass to `/home/predicate/dev/myai/inference/llama_cpp_server.py`.
- Modify `LlamaCppServerProvider` only at request execution points; its model-to-server map and lazy startup behavior remain unchanged.
- Modify `LlamaCppEmbeddingServer` only at request execution points; its existing lazy startup behavior remains unchanged.

## Dependencies

No new packages or version changes are needed. The implementation uses Python's existing `asyncio`, `time`, `contextlib`, `dataclasses`, and `subprocess` modules.

## Testing

- Use `unittest.IsolatedAsyncioTestCase` and mocked subprocess objects so tests do not require llama.cpp or PostgreSQL.
- Verify that an entry idle for five minutes is terminated and removed.
- Verify that an entry with an active request is not terminated even when its timestamp is old.
- Verify that entering and leaving an activity guard refreshes usage and releases the active-request count.
- Verify that global shutdown stops the monitor and clears the process map.
- Run the focused unit test, the full test suite where available, `ruff check`, and `mypy .`.

## Implementation Order

1. Add the plan document and process-management types/constants.
2. Implement activity tracking and the idle monitor around `_server_processes`.
3. Integrate activity guards into chat and embedding operations.
4. Update global shutdown handling.
5. Add isolated unit tests using mocked subprocesses and time.
6. Run focused and broad validation commands.
7. Re-read all edited files and report any environment-related validation limitations.