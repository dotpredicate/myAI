# Inference Providers

Providers are backends that run AI models. Chat providers implement the `InferenceProvider` interface, image providers implement `ImageProvider`.

## Currently available

- `llama.cpp` — embedded local llama.cpp server (chat, `registry`)
- `stable-diffusion.cpp` — embedded local sd-server / stable-diffusion.cpp (image generation, `image_registry`)

Provider instances live in in-memory registries (`ProviderRegistry`, one per provider kind). Persistent custom providers are planned for the future.

## Image models

Image models are user-managed presets stored in the `sd_cpp_image_models` Postgres table: each row binds an sd-server recipe to component file IDs.

Files must live in an sd-server model directory. `MYAI_SD_MODEL_DIRS` is a colon-separated list of roots (default `~/.cache/myai/models`); each root uses the usual category layout (`diffusion_models/`, `text_encoders/`, `vae/`, …). You can point it at an existing ComfyUI/A1111 model root or download into it, for example with `hf download <repo> <file> --local-dir <root>/<category>`. myAI never copies or links model weights.

Add models through the Images view: pick a recipe exposed by `GET /api/sd-cpp/recipes`, assign a catalog file to each recipe slot (from `GET /api/sd-cpp/files`), and save. The provider loads the preset with `POST /sdcpp/v1/models/load` and sends recipe defaults with the generation request. Removing a model only deletes the database row; files stay on disk.

stable-diffusion.cpp-specific model management lives in `inference/sd_cpp/router.py` under `/api/sd-cpp`. Relevant endpoints: `GET/POST /api/sd-cpp/models`, `DELETE /api/sd-cpp/models/{id}`, `GET /api/sd-cpp/recipes`, `GET /api/sd-cpp/files`, `GET /api/sd-cpp/inference-params`. The generic image endpoint `POST /api/images/generate` stays in `images.py`.

### Why stable-diffusion.cpp

See [Image backend comparison](analysis/image_backends.md) for the trade-offs against ComfyUI and diffusers/Unsloth, and why we keep sd.cpp despite its model-directory and file-ID constraints.
