# Inference Providers

Providers are backends that run AI models. Each implements the `InferenceProvider` interface.

## Currently available

- `llama.cpp` — embedded local llama.cpp server

Provider instances live in an in-memory registry. Persistent custom providers are planned for the future.
