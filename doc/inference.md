# Inference

Chat providers implement `InferenceProvider`, image providers `ImageProvider`.
Instances live in in-memory registries that are rebuilt when configuration
changes.

## Chat providers

- The embedded llama.cpp server is registered in code.
- OpenAI-compatible endpoints are user-defined and stored in the database.

A provider has a key, a display name, connection details and optional secrets.
Secrets are write-only: they can be set and replaced, never read back. Updates
are partial - omitted settings stay, explicit `null` clears. When a provider
cannot list its models, the configured list is used.

A provider that an agent or a conversation still refers to cannot be deleted.

## Inference parameters

Providers describe the parameters their models accept: name, type, default and
range. Clients render them as model settings, and the chosen values travel with
the request that uses them.

## Image models

Image generation uses an embedded stable-diffusion.cpp server. Models are
user-managed presets: each preset binds a recipe exposed by the server to the
component files it needs.

The files must live under one of the roots listed in `MYAI_SD_MODEL_DIRS`, in
the usual category layout, so an existing ComfyUI or A1111 model directory can
be used directly. myAI never copies or links model weights. Removing a preset
deletes only the database entry; files stay on disk.

See [Image backend comparison](analysis/image_backends.md) for the trade-offs
against ComfyUI and diffusers/Unsloth, and why sd.cpp is kept despite its model
directory and file constraints.
