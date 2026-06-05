
# Conversations

An overview of how a conversation is structured.

A conversation consists of a stack of actions such as:
- user message
- user/system actions such as:
    - acceptation / rejection of a secured action
    - interrupt (i.e. async action result)
    - cancellation of ongoing generation
- agent message
- agent thinking
- agent tool calls, such as:
    - using semantic search
    - executing a command
    - file change (diff/replace)
    - spawning a sub-agent
