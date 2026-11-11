# Conversations

A conversation is the unit of work: an ordered transcript bound to one
workspace, run on the model its last prompt chose, optionally guided by an
agent.

## Record

- Workspace - set when the conversation is created; deleting the workspace
  moves it to another one.
- Agent - optional, fixed when the conversation is created.
- Model - the last prompt sets it, together with its inference parameters;
  continuations and sub-agents reuse it.
- Transcript - the ordered elements below.
- Blocking tool call - the message the conversation waits on, if any.

## Context

Every provider call receives a `ChatContext`:

- `messages` - the transcript
- `scopes` - the repositories of the conversation's workspace
- `tools` - the tools the model may call
- `instructions` - the agent's instructions, if an agent is used

## Elements

The transcript is a sequence of elements, each stored with a per-conversation
order that never changes:

- user and assistant messages,
- model thinking,
- tool calls and their results,
- approval requests and the user's decisions.

## Lifecycle

1. A prompt carries the model. It is accepted only when the conversation is
   idle and unblocked; otherwise nothing is written.
2. The accepted prompt stores the model and appends the user message.
3. Generation runs in the background: it calls the provider, persists finalized
   elements and continues after tool calls that need no approval.
4. A tool call that needs approval blocks the conversation until the user
   decides. Approving runs the tool; rejecting records the decision. Neither
   resumes generation on its own - that is what a continuation is for.
5. A continuation re-enters the loop with the full transcript.
6. Deleting a conversation removes its transcript, stops any running generation
   and drops its sandbox.

## Streaming

Generation state lives in memory per conversation: its status, the element
being streamed, pending events and subscribers. Statuses distinguish a running
generation from ones that finished, failed, blocked or were cancelled.

A client that attaches replays what it has not seen yet, then follows live
events. Because it can attach in the middle of a message, anything committed
while it was connecting is replayed before the stream ends.

Clients receive streamed text and thinking, tool calls and results, approval
requests and decisions, a finalization event per stored element, and generation
errors.

## Rules

- One generation per conversation at a time.
- A blocked conversation accepts no prompt until its approval is decided.
- The transcript is append-only; only deleting the conversation removes it.
