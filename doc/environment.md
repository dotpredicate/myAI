# Environment

What a model can see and do while it works. The sandbox and the tool layer
enforce this; the prompt only describes it.

## Sandbox

Shell commands run as `bash` inside bubblewrap with a clean environment:

- `/workspace` - the workspace's durable model space; writable, survives across
  conversations
- `/sandbox` - scratch space for this conversation; writable, discarded with it
- `/repositories/<internal_name>` - every source of the conversation's
  workspace, mounted read-only
- `/tmp` - ephemeral scratch space, cleared between commands
- `/usr`, `/lib`, `/lib64`, `/bin` - read-only system paths
- `/proc` and a minimal `/dev` - process and device filesystems

The host filesystem (including home directories), the network, GPUs and audio
devices are not available.

## Tools

- `bash` - run a shell command in the sandbox.
- `search` - semantic search over the workspace's sources; without sources it
  finds nothing.
- `replace_file` - replace a repository file with a file from `/workspace`.
- `apply_patch` - apply a unified diff from `/workspace` to a repository file.
- `spawn_agent` - start a conversation with another agent; offered only to
  agents that list sub-agents.

`search` is the only way to read indexed content without the shell. The
[indexing rules](repositories.md#indexing) define what is indexed.

## Changes to repositories

See [repositories](repositories.md#access) for the access invariant and how a
change is approved. The model stages content under `/workspace`; `/sandbox` is
for scratch work and cannot be the source of a change.

See [conversations](conversations.md) for how approvals fit the transcript.
