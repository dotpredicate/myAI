# Repositories

Repositories are sources of data shared with workspaces.

## Spec

- Display name and internal name (the key used in paths).
- Path - canonicalized, so symlinks do not create a second repository for the
  same directory.
- Type - a plain folder or a Git repository, detected when not stated.

One path is one repository: registering an existing folder again is refused.

## Access

There are no configurable security levels. The invariant is:

- repositories are mounted read-only for `bash`,
- `/workspace` is writable,
- changes go through `replace_file` or `apply_patch`,
- a change is applied only after the user approves it.

A repository can be a source of several workspaces; the workspace decides what
the model sees. Deleting a repository detaches it from every workspace and drops
its index. Deleting never touches files on disk.

## Indexing

Readable text files are saved into a document index. In Git repositories only
the files known to Git are indexed. Model spaces are not indexed.

## Search

Semantic search returns indexed fragments ordered by similarity. Results are
limited to one scope at a time: a single repository, or all sources of a
workspace. Without a scope nothing is searched.

## Related

- [workspaces](workspaces.md) - what a repository serves.
- [environment](environment.md) - how the model reaches it.
