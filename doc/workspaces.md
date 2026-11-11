# Workspaces

A workspace is a virtual container for sources and the context of a group of
conversations. It is not a filesystem root: sources may live anywhere.

## Spec

- Name - a label; names are not unique.
- Sources - the repositories conversations may read.
- Model space - the workspace's durable directory under `$MYAI_HOME/spaces/`,
  mounted as `/workspace`.

## Semantics

- Every conversation belongs to exactly one workspace, fixed when it is
  created.
- The workspace is the only source of repository scopes. Agents never add or
  remove repository access.
- A repository can be a source of several workspaces.
- A fresh database is seeded with `Scratch`, a normal workspace with no sources.

## Lifecycle

- **Create** - a workspace is created from a path: the path is canonicalized,
  its git root found, the repository registered and the workspace containing it
  found or created. Creating also prepares the model space and starts
  indexing. An existing workspace that already holds the repository wins over
  creating a new one, and the home directory, `/` and the myAI data directory
  are refused.
- **Rename** - changes the label only; the id and the model space stay.
- **Sources** - added explicitly, one repository at a time; adding one does not
  remove the repository from other workspaces.
- **Delete** - its conversations move to another workspace and its model space
  is kept. The last workspace cannot be deleted.

## Upgrade

A database from before workspaces drops its projects: nothing is carried over.
Every existing conversation joins the first workspace and keeps its messages.

## Related

- [repositories](repositories.md) - what can be a source.
- [environment](environment.md) - what the model sees, including `/workspace`.
- [conversations](conversations.md) - how a workspace is bound to a
  conversation.
