import asyncio
import unittest

from app import app, lifespan
from tests.helpers import BaseTestCase


class TestApplicationLifespan(BaseTestCase):
    async def test_application_starts_and_stops(self):
        async with lifespan(app):
            await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
