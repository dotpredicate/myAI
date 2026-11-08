
# Agents

Agents are how you configure AI to do specific tasks.

## Agent spec

- Name
    - Display name
    - Internal name (key)
- Description
- Inference config:
    - Provider key
    - Model
    - Model and inference-specific configs
- Default repositories
    - Override security policy (can only downgrade)
- Default actions
- Available sub-agents

## Sub-agents

An agent can list the agents it may start. The `spawn_agent(agent_symbol, prompt)`
tool is exposed only when that list is non-empty; `agent_symbol` is the target
agent's internal name.

A spawn starts a new conversation with the target agent's provider, model and
instructions. The child inherits the parent's resolved repository scopes, so it
cannot gain access the parent does not have. The tool returns
`{conversation_id, agent, response}`; if the child stops on an approval, the
result also contains `blocked` and `blocking_message_id`. Nesting is limited to
three levels.