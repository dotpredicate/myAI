from typing import Optional

from pydantic import BaseModel

from agents import get_agent_by_name
from domain import AgentConfig, ScopeSpec, SecurityPolicy
from repositories import get_repo_by_id, get_repo_by_name


class UserScopeChoice(BaseModel):
    internal_name: str
    security_policy_override: Optional[SecurityPolicy] = None


async def resolve_scope(choice: UserScopeChoice, agent: Optional[AgentConfig] = None) -> ScopeSpec:
    """Resolve one requested scope: user override > agent override > repo policy."""
    if choice.security_policy_override is not None:
        return ScopeSpec(
            internal_name=choice.internal_name,
            security_policy=choice.security_policy_override,
        )
    if agent is not None:
        for access in agent.repository_access:
            if (
                access.repository_internal_name == choice.internal_name
                and access.security_policy_override is not None
            ):
                return ScopeSpec(
                    internal_name=choice.internal_name,
                    security_policy=access.security_policy_override,
                )
    repo = await get_repo_by_name(choice.internal_name)
    if repo is None:
        raise ValueError(f"Repository '{choice.internal_name}' not found")
    return ScopeSpec(
        internal_name=choice.internal_name,
        security_policy=repo.security_policy,
    )


async def scopes_for_agent(agent: AgentConfig, req_scopes: list[UserScopeChoice]) -> list[ScopeSpec]:
    """Resolve explicit choices plus every repository the agent can access."""
    scopes = [await resolve_scope(choice, agent=agent) for choice in req_scopes]
    seen = {scope.internal_name for scope in scopes}
    for access in agent.repository_access:
        repo_key = access.repository_internal_name
        if repo_key in seen:
            continue
        if access.security_policy_override is None:
            repo = await get_repo_by_id(access.repository_id)
            if repo is None:
                raise ValueError(f"Repository {access.repository_id} not found")
            policy = repo.security_policy
        else:
            policy = access.security_policy_override
        scopes.append(ScopeSpec(internal_name=repo_key, security_policy=policy))
        seen.add(repo_key)
    return scopes


async def resolve_scopes(agent_id: Optional[str], req_scopes: list[UserScopeChoice]) -> list[ScopeSpec]:
    """Resolve effective scopes for an agent or a bare request.

    With an agent, an empty ``req_scopes`` means all repositories the agent can
    access. Without an agent, only explicit choices are available; an empty
    list means no access.
    """
    if agent_id:
        agent = await get_agent_by_name(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found")
        return await scopes_for_agent(agent, req_scopes)
    return [await resolve_scope(choice) for choice in req_scopes]
