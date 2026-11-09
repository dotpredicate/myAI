
# Projects

A project bundles the context for a group of conversations.

## Project spec

- Name
    - Display name
    - Internal name (key)
- Description
- Project instructions
- Default agent
- Model preset
    - Provider key
    - Model
    - Model and inference-specific config
- Repository presets
    - Optional security policy override (can only downgrade)

## Semantics

- **Instructions** are enforced by the back-end. A conversation bound to a
  project receives the project instructions before the agent instructions in
  the system prompt. The back-end reads the project from the conversation row,
  not from the request.
- **Default agent, model preset and repository presets** are applied by the
  front-end when a project is chosen in the chat. The back-end does not add
  project repositories or replace the selected model; API clients must send
  the effective agent, model and scopes explicitly.
- `default_agent` takes precedence over the model preset. When it is set, the
  model preset is ignored in the chat. When it is cleared and a model preset is
  present, the chat switches to that manual model.
- Presets are applied when a project is selected and when an existing
  conversation bound to that project is opened. Changing the agent, model or
  scopes afterwards is allowed for the rest of the conversation.
- Selecting “No project” removes only the scopes provided by the previous
  project. It does not clear the selected agent or model.
- A conversation is bound to its project when it is created. The project
  cannot be changed afterwards.
- Sub-agents spawned from a project conversation inherit the project id, so
  they also receive the project instructions and the parent's resolved scopes.
- Deleting a project keeps its conversations; their `project_id` becomes
  `NULL`.

## API

- `GET /api/projects`
- `POST /api/projects`
- `PUT /api/projects/{internal_name}`
- `DELETE /api/projects/{internal_name}`

`default_agent` is the numeric agent id. `provider_key` and `model_id` must be
provided together. Repository policy overrides are validated against the
repository's base policy.
