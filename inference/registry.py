from dataclasses import dataclass

from inference.engine import InferenceProvider
from inference.llama_cpp_server import LlamaCppServerProvider


@dataclass(frozen=True)
class ProviderRegistration:
    key: str
    display_name: str
    description: str
    provider: InferenceProvider


class InferenceProviderRegistry:
    """In-memory registry for inference providers.

    Providers are stored by key in insertion order.
    The first registered provider is returned by get_default().
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderRegistration] = {}

    def register(
        self,
        key: str,
        display_name: str,
        description: str,
        provider: InferenceProvider,
    ) -> None:
        """Register a provider. Raises ValueError if the key already exists."""
        if key in self._providers:
            raise ValueError(f"Provider with key '{key}' is already registered")
        self._providers[key] = ProviderRegistration(
            key=key,
            display_name=display_name,
            description=description,
            provider=provider,
        )

    def get(self, key: str) -> InferenceProvider:
        """Return the provider instance for the given key. Raises KeyError if not found."""
        if key not in self._providers:
            raise KeyError(f"Provider '{key}' not found")
        return self._providers[key].provider

    def list_registrations(self) -> list[ProviderRegistration]:
        """Return metadata for all registered providers."""
        return list(self._providers.values())

    def get_default(self) -> InferenceProvider:
        """Return the first registered provider's instance.

        Raises RuntimeError if no providers are registered.
        """
        if not self._providers:
            raise RuntimeError("No inference providers are registered")
        key = next(iter(self._providers))
        return self._providers[key].provider


# Module-level singleton
registry = InferenceProviderRegistry()

# Register the built-in llama.cpp server provider
registry.register(
    "llama.cpp",
    "Llama.cpp Server",
    "Embedded local llama.cpp server",
    LlamaCppServerProvider(),
)