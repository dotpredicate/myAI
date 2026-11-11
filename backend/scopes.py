from typing import Optional

from backend.database import mk_conn
from backend.domain import ScopeSpec


async def resolve_workspace_scopes(workspace_id: Optional[int]) -> list[ScopeSpec]:
    """Return the repositories available to a workspace.

    The workspace is the only source of scopes. Agents never add or remove
    repository access.
    """
    if workspace_id is None:
        return []
    async with mk_conn() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT r.internal_name
            FROM workspace_sources ws
            JOIN repositories r ON r.id = ws.repository_id
            WHERE ws.workspace_id = %s
            ORDER BY r.internal_name
            """,
            (workspace_id,),
        )
        return [ScopeSpec(internal_name=row[0]) for row in await cur.fetchall()]
