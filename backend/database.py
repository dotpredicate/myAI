import hashlib
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

import aiosqlite
import sqlite_vec  # type: ignore[import-untyped]

from backend import paths
from backend.log_config import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR: Path = Path(__file__).resolve().parent.parent / 'migrations'

_DETECT_TYPES = sqlite3.PARSE_DECLTYPES


def _parse_timestamp(value: bytes) -> datetime:
    parsed = datetime.fromisoformat(value.decode('utf-8'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_json(value: bytes) -> Any:
    return json.loads(value.decode('utf-8'))


sqlite3.register_converter('JSON', _parse_json)
sqlite3.register_converter('TIMESTAMP', _parse_timestamp)
sqlite3.register_converter('TIMESTAMPTZ', _parse_timestamp)


def _to_sqlite(sql: str) -> str:
    return sql.replace('%s', '?')


class Cursor:
    def __init__(self, cursor: aiosqlite.Cursor) -> None:
        self._cursor = cursor

    async def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> 'Cursor':
        await self._cursor.execute(_to_sqlite(sql), params)
        return self

    async def executemany(self, sql: str, params: Iterable[Iterable[Any]]) -> 'Cursor':
        await self._cursor.executemany(_to_sqlite(sql), params)
        return self

    async def fetchone(self):
        return await self._cursor.fetchone()

    async def fetchall(self):
        return await self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> Optional[int]:
        return self._cursor.lastrowid

    def __aiter__(self) -> 'Cursor':
        return self

    async def __anext__(self):
        row = await self._cursor.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row


class _CursorContext:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._cursor: Optional[aiosqlite.Cursor] = None

    async def __aenter__(self) -> Cursor:
        self._cursor = await self._connection.cursor()
        return Cursor(self._cursor)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._cursor is not None:
            await self._cursor.close()


class Connection:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    def cursor(self) -> _CursorContext:
        return _CursorContext(self._connection)

    async def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> Cursor:
        cursor = await self._connection.execute(_to_sqlite(sql), params)
        return Cursor(cursor)

    async def executemany(self, sql: str, params: Iterable[Iterable[Any]]) -> Cursor:
        cursor = await self._connection.executemany(_to_sqlite(sql), params)
        return Cursor(cursor)

    async def commit(self) -> None:
        await self._connection.commit()

    async def rollback(self) -> None:
        await self._connection.rollback()

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    async def begin_immediate(self) -> None:
        if not self._connection.in_transaction:
            await self._connection.execute('BEGIN IMMEDIATE')


AsyncConnection = Connection


async def _configure(connection: aiosqlite.Connection) -> None:
    await connection.execute('PRAGMA foreign_keys = ON')
    await connection.execute('PRAGMA busy_timeout = 5000')
    await connection.execute('PRAGMA synchronous = NORMAL')
    await connection.execute('PRAGMA journal_mode = WAL')
    await connection.enable_load_extension(True)
    try:
        await connection.load_extension(sqlite_vec.loadable_path())
    finally:
        await connection.enable_load_extension(False)


async def _open() -> aiosqlite.Connection:
    paths.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = await aiosqlite.connect(
        str(paths.DATABASE_PATH),
        detect_types=_DETECT_TYPES,
        timeout=5.0,
    )
    await _configure(connection)
    return connection


@asynccontextmanager
async def mk_conn() -> AsyncIterator[Connection]:
    connection = await _open()
    try:
        yield Connection(connection)
    except BaseException:
        await connection.rollback()
        raise
    else:
        if connection.in_transaction:
            await connection.commit()
    finally:
        await connection.close()


async def init_database() -> None:
    paths.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = await _open()
    try:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                order_executed INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                hash TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL
            )
            """
        )
        await connection.commit()

        migration_files = sorted(MIGRATIONS_DIR.glob('*.sql'))
        for migration_file in migration_files:
            migration_name = migration_file.name
            sql = migration_file.read_text(encoding='utf-8')
            migration_hash = hashlib.sha256(sql.encode('utf-8')).hexdigest()
            cursor = await connection.execute(
                'SELECT hash FROM migrations WHERE name = ?', (migration_name,)
            )
            existing = await cursor.fetchone()
            await cursor.close()
            if existing:
                if existing[0] != migration_hash:
                    logger.warning(
                        'Migration %s changed after it was applied; not re-running it',
                        migration_name,
                    )
                logger.info('Skipping already executed migration: %s', migration_name)
                continue

            start_time = datetime.now(timezone.utc)
            try:
                await connection.executescript(f"BEGIN;\n{sql}\nCOMMIT;")
            except BaseException:
                await connection.rollback()
                raise
            end_time = datetime.now(timezone.utc)
            await connection.execute(
                """
                INSERT INTO migrations (name, hash, start_time, end_time)
                VALUES (?, ?, ?, ?)
                """,
                (
                    migration_name,
                    migration_hash,
                    start_time.isoformat(),
                    end_time.isoformat(),
                ),
            )
            await connection.commit()
            logger.info('Applied migration: %s', migration_file)
        logger.info('Migrations completed')
    finally:
        await connection.close()

    try:
        os.chmod(paths.DATABASE_PATH, 0o600)
    except OSError:
        pass
    if paths.DATABASE_PATH.parent != Path.home():
        try:
            os.chmod(paths.DATABASE_PATH.parent, 0o700)
        except OSError:
            pass
