# Context

Context is the combined state passed to Agents.

## ChatContext

The single object given to every provider call.

- `scopes` - which repositories are available to the Agent
- `tools` - tools available to the Agent
- `agent_prompt` - configurable Agent prompts
- `messages` - conversation history

## Conversation

A conversation is an ordered sequence of interactions stored in the database.

- User message
- User/system actions such as:
    - acceptation / rejection of a secured action
    - interrupt (i.e. async action result)
    - cancellation of ongoing generation
- Agent message
- Agent thinking
- Agent tool calls, such as:
    - Using semantic search
    - Executing a command
    - File change (diff/replace)
    - Spawning a sub-agent

### Scopes

Available repositories and associated resolved security policies are passed to the Agent.
