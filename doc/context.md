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

### Scopes

User messages and Agents carry security scopes.

Scopes are extracted from the last user message on continuation and forwarded to tool execution.

### Lifecycle

1. **Prompt** — inserts a user `Message` (with scopes) into a new or existing conversation, then streams the assistant response
2. **Generate** — loads messages + scopes, wraps them in `ChatContext`, calls the provider. After a non-blocking tool call the loop continues automatically
3. **Blocking tool calls** — set `blocking_message_id`. The front-end must approve or reject before continuing
4. **Decide** — approve runs the tool (privileged), reject inserts a `ToolCallDecision` with optional comment. Both clear the block
5. **Continue** — re-enters the generation loop with the full updated history
6. **Delete** — removes the conversation and all its messages
