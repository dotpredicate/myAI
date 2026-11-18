# Inference

Chat providers turn a prompt into a streamed answer; image providers turn a
prompt into an image. They are separate interfaces with separate configuration.

## Chat providers

- The embedded llama.cpp server is built in. It serves GGUF models from the
  HuggingFace cache and starts a server for a model when it is first used.
- OpenAI-compatible endpoints are added by hand: a base URL, an optional key,
  and either a model list fetched from the endpoint or a fixed list.

A provider has a key, a display name, connection details and optional secrets.
Secrets are write-only: they can be set and replaced, never read back. Updates
are partial - omitted settings stay, explicit `null` clears. When an endpoint
cannot list its models, the configured list is used.

A provider that an agent or a conversation still refers to cannot be deleted.

## llama.cpp chat server

Each chat server runs one model on the context window the model was trained
with, read from the GGUF, and serves one request at a time, so concurrent
requests queue. A model that does not fit in VRAM fails at startup instead of
running on a smaller window or on the CPU.

Setting `n_ctx` as an inference parameter gives the model a different window,
and a different window is a second server process: servers are keyed by window
and model.

The throughput benchmark loads the same model with the same offloading and
attention flags, but sizes its own context per depth and starts no server. Its
numbers describe the model, not the server.

## Inference parameters

Providers describe the parameters their models accept: name, type, default and range. 
Clients render them as model settings, and the values chosen travel with the
request that uses them.

## Image models

Image generation uses an embedded stable-diffusion.cpp server. Models are
user-managed presets stored in the `sd_cpp_image_models` table: each preset
binds a recipe exposed by the server to the component files it needs.

The files must live under one of the roots listed in `MYAI_SD_MODEL_DIRS`
(default `~/.cache/myai/models`), in the usual category layout, so an existing
ComfyUI or A1111 model directory can be used directly. myAI never copies or
links model weights. Removing a preset deletes only the database entry; files
stay on disk.

See [Image backend comparison](analysis/image_backends.md) for the trade-offs
against ComfyUI and diffusers/Unsloth, and why sd.cpp is kept despite its model
directory and file constraints.
