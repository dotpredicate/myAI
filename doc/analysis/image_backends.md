# Image backends

myAI uses stable-diffusion.cpp for image generation. The alternatives are ComfyUI and a diffusers-based runtime such as Unsloth.

stable-diffusion.cpp is a standalone C++/ggml engine. It runs GGUF natively, needs no Python at runtime, and ships as a small binary. The cost is its model management: recipes, model directories with category subdirectories, and file IDs instead of plain paths. Its recipes are fixed pipelines, so there is no arbitrary node graph.

ComfyUI is a Python/PyTorch node-graph application. It is far more flexible, but it is not an embeddable library: integration means running the whole app and calling its HTTP API. GGUF support comes from a custom node, not the core. It also adds a large Python dependency tree and is GPL-3.0.

diffusers is a Python library for diffusion pipelines. It loads models by path or repo and can be extended in code, but GGUF needs extra quantization and conversion work. Unsloth does exactly that: diffusers plus GGUF quantization, custom converters and runtime patches. It is the closest to "give it a model and it runs", at the cost of torch/diffusers and a lot of glue. Core Unsloth is Apache-2.0, but its Studio component is AGPL-3.0.

A filled sd.cpp recipe (recipe plus concrete files) is roughly a saved ComfyUI loader graph. The components are fixed; ComfyUI can express much more.

We keep sd.cpp because myAI is a single Python app and images are one subprocess: no torch environment to maintain, native GGUF, and the same ggml family as llama.cpp. MIT licensing also helps.

What we accept: files live in sd.cpp model directories (`MYAI_SD_MODEL_DIRS`), the UI binds recipe slots to catalog file IDs, recipes are fixed pipelines, and users manage downloads themselves.

Revisit this if model management or image performance becomes more important than deployment simplicity. The realistic alternative then is a diffusers-based provider, not ComfyUI.
