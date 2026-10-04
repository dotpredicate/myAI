import asyncio
import unittest

from app import _cancel_and_wait, logger


class TestBackgroundTaskCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_cancels_a_running_task_and_waits_for_its_cleanup(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_until_cancelled() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(wait_until_cancelled())
        await started.wait()

        await _cancel_and_wait(task)

        self.assertTrue(cancelled.is_set())
        self.assertTrue(task.cancelled())

    async def test_logs_failure_from_completed_task(self):
        async def fail() -> None:
            raise RuntimeError("index sync failed")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)

        with self.assertLogs(logger, level="ERROR"):
            await _cancel_and_wait(task)

        self.assertTrue(task.done())


if __name__ == "__main__":
    unittest.main()
