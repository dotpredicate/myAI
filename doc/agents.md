# Agents

Agents are reusable behaviour: a name, description, instructions and optional
model preference. They never carry repository access or private data.

## Agent spec

- Name
    - Display name (`display_name`)
    - Internal name (`internal_name`)
- Description (`description`)
- Instructions (`instructions`)
- Optional model preference
    - Provider key (`provider_key`)
    - Model ID (`model_id`)
    - Inference config (`inference_config`)
- Available sub-agents (`spawnable_agents`)

An agent is chosen when a conversation is created and stays with it.

## Sub-agents

An agent can list the agents it may start. The `spawn_agent(agent_symbol,
prompt)` tool is exposed only when that list is non-empty.

A spawn starts a new conversation in the parent's workspace with the target
agent's instructions. The child inherits the parent's model configuration and
its workspace, so it sees exactly the repositories the parent sees. Spawning is
depth-limited so agents cannot recurse without bound.
