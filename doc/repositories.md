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
- a change blocks the conversation until the user approves it; nothing is
  written before that.

Deleting a repository detaches it from every workspace and drops its index.
Deleting never touches files on disk.

## Indexing

Indexing is opt-in per repository. A repository is embedded into the document
index only when **Index for search** is checked; unchecked repositories are
skipped and any of their existing entries are removed. New and existing
repositories start unchecked. Server startup and **Update index** synchronize
checked repositories only.

Files with a known text extension are decoded and saved into a document index;
binary files and unknown extensions are skipped. In Git repositories only the
files known to Git are indexed. Model spaces are not indexed, so `/workspace`
and `/sandbox` never appear in search.

## Search

Semantic search returns indexed fragments ordered by similarity. Results are
limited to one scope at a time: a single repository, or all sources of a
workspace. Without a scope nothing is searched.

## Related

- [workspaces](workspaces.md) - what a repository serves.
- [environment](environment.md) - how the model reaches it.
