from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from datetime import datetime
import os
import glob
import hashlib
import binascii
from log_config import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

_cached_pool = None
_cached_database_url = DATABASE_URL

async def get_pool():
    global _cached_pool, _cached_database_url
    if _cached_database_url != DATABASE_URL or _cached_pool is None:
        _cached_pool = AsyncConnectionPool(
            DATABASE_URL,
            min_size=2,
            max_size=4,
        )
        _cached_database_url = DATABASE_URL
        await _cached_pool.open()
        await _cached_pool.wait()
    return _cached_pool

@asynccontextmanager
async def mk_conn():
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn

async def init_database():
    async with mk_conn() as conn, conn.cursor() as cur:
        # Ensure pgvector is available
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        await cur.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                order_executed INTEGER GENERATED ALWAYS AS IDENTITY,
                name VARCHAR(255) NOT NULL,
                hash CHARACTER(64) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS migrations_name ON migrations (name);
            """
        )
        await conn.commit()

    migration_dir = 'migrations'
    migration_files = sorted(glob.glob(os.path.join(migration_dir, '*.sql')))
    for migration_file in migration_files:
        migration_name = os.path.basename(migration_file)
        async with mk_conn() as conn, conn.cursor() as cur:
            await conn.set_autocommit(False)
            await cur.execute("SELECT 1 FROM migrations WHERE name = %s", (migration_name,))
            row = await cur.fetchone()
            if row:
                logger.info("Skipping already executed migration: %s", migration_name)
                continue
            with open(migration_file, 'r') as f:
                sql = f.read()
            await cur.execute(sql)
            start_time = datetime.now()
            hash_bytes = hashlib.sha256(migration_name.encode('utf-8')).digest()
            hash_str = binascii.hexlify(hash_bytes).decode('utf-8')
            end_time = datetime.now()
            await cur.execute(
                """
                INSERT INTO migrations (name, hash, start_time, end_time)
                VALUES (%s, %s, %s, %s)
                """,
                (migration_name, hash_str, start_time, end_time)
            )
            await conn.commit()
        logger.info("Applied migration: %s", migration_file)
    logger.info("Migrations completed")