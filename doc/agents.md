# Agents

Agents are reusable behaviour: a name, description, instructions and optional
model preference. They never carry repository access or private data.

## Agent spec

- Name
    - Display name
    - Internal name (key)
- Description
- Instructions
- Optional model preference
    - Provider key
    - Model
    - Inference config
- Available sub-agents

An agent is chosen when a conversation is created and stays with it.

## Sub-agents

An agent can list the agents it may start. The `spawn_agent(agent_symbol,
prompt)` tool is exposed only when that list is non-empty.

A spawn starts a new conversation in the parent's workspace with the target
agent's instructions. The child inherits the parent's model configuration and
its workspace, so it sees exactly the repositories the parent sees. Spawning is
depth-limited so agents cannot recurse without bound.
