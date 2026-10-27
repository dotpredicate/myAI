from dataclasses import dataclass

from inference.engine import ImageProvider, InferenceProvider
from inference.llama_cpp.server import LlamaCppServerProvider
from inference.sd_cpp.provider import SdCppImageProvider


@dataclass(frozen=True)
class ProviderRegistration[ProviderT]:
    key: str
    display_name: str
    description: str
    provider: ProviderT


class ProviderRegistry[ProviderT]:
    """In-memory registry for providers.

    Providers are stored by key in insertion order.
    The first registered provider is returned by get_default().
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderRegistration[ProviderT]] = {}

    def register(
        self,
        key: str,
        display_name: str,
        description: str,
        provider: ProviderT,
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

    def get(self, key: str) -> ProviderT:
        """Return the provider instance for the given key. Raises KeyError if not found."""
        if key not in self._providers:
            raise KeyError(f"Provider '{key}' not found")
        return self._providers[key].provider

    def list_registrations(self) -> list[ProviderRegistration[ProviderT]]:
        """Return metadata for all registered providers."""
        return list(self._providers.values())

    def get_default(self) -> ProviderT:
        """Return the first registered provider's instance.

        Raises RuntimeError if no providers are registered.
        """
        if not self._providers:
            raise RuntimeError("No providers are registered")
        key = next(iter(self._providers))
        return self._providers[key].provider


# Module-level singletons
registry: ProviderRegistry[InferenceProvider] = ProviderRegistry()

registry.register(
    "llama.cpp",
    "Llama.cpp Server",
    "Embedded local llama.cpp server",
    LlamaCppServerProvider(),
)

image_registry: ProviderRegistry[ImageProvider] = ProviderRegistry()

image_registry.register(
    "stable-diffusion.cpp",
    "stable-diffusion.cpp Server",
    "Embedded local stable-diffusion.cpp server",
    SdCppImageProvider(),
)
