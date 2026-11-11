import os

from tests.helpers import BaseTestCase


class TestServerInfo(BaseTestCase):
    async def test_info_reports_process(self):
        response = await self.client.get('/api/info')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'pid': os.getpid()})
