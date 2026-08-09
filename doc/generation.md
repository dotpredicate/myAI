# Generation

Generation is stateful: it runs in a background task, so clients can attach to an in-progress generation (F5, another window).

## State

- Per-conversation in-memory state: `status` (`running|finished|failed|blocked|cancelled`), buffered streamed element, pending events, subscribers.
- A registry maps conversation id to its active generation.
- `sequence_id` - per-conversation monotonic message order; the stream sync point (`after_sequence_id`).

## Flow

A generation is started in the background, runs the generation loop, and applies each event to the state.

- The generation loop loads history, calls the provider, persists finalized elements, and continues after non-blocking tool calls.
- The background task wraps the loop, applies events to state, and handles finish/fail/cancel.
- Streaming replays persisted messages and the in-memory snapshot, then live-streams from the subscription queue.

## Stream events

- `MessageChunk` / `ThinkingChunk` - streamed token by token.
- `ToolCallEvent` - tool call (not streamed).
- `ElementFinalized` - element persisted, with `id` and `sequence_id`.
- `GenerationErrorEvent` - generation failed.

The buffer holds only streamed elements (message/thinking). Tool calls go to pending events instead.

## Lifecycle

1. **Prompt** - insert user message, stream assistant response.
2. **Generate** - call provider; continue loop after non-blocking tool calls.
3. **Blocking tool call** - set `blocking_message_id`; front-end must approve/reject.
4. **Decide** - approve runs tool (privileged), reject inserts `ToolCallDecision`; both clear the block. Does not auto-resume; client calls Continue.
5. **Continue** - re-enter the loop with full history.
6. **Delete** - remove conversation and messages, cancel active generation.

## Errors

- Second prompt/continue while generating - `409`.
- Prompt on blocked conversation - `403` with `blocking_message_id`.
- Generation failure - `GenerationErrorEvent` to subscribers.