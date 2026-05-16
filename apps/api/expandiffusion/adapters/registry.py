"""Adapter registry."""

from __future__ import annotations

from collections.abc import Iterable

from .. import constants
from ..errors import AppError
from ..schemas import AdapterInfo
from .base import ModelAdapter
from .diffusers_inpaint import (
    ChromaInpaintAdapter,
    FluxFillFp8Adapter,
    FluxFillAdapter,
    Sd2InpaintAdapter,
    Sd15ControlNetInpaintAdapter,
    Sd15Img2ImgAdapter,
    Sd15InpaintAdapter,
    SdxlControlNetInpaintAdapter,
    SdxlFillControlNetUnionAdapter,
    SdxlImg2ImgAdapter,
    SdxlInpaintAdapter,
    ZImageInpaintAdapter,
)


class AdapterRegistry:
    """Mutable registry of available model adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        self._adapter_plugin_ids: dict[str, str | None] = {}

    def register(self, adapter: ModelAdapter, plugin_id: str | None = None) -> None:
        """Register an adapter by id."""
        if adapter.id in self._adapters:
            raise AppError(
                constants.ERROR_PLUGIN_LOAD_FAILED,
                f"Adapter '{adapter.id}' is already registered.",
                status_code=500,
                details={"adapter_id": adapter.id, "plugin_id": plugin_id},
            )
        self._adapters[adapter.id] = adapter
        self._adapter_plugin_ids[adapter.id] = plugin_id

    def get(self, adapter_id: str) -> ModelAdapter:
        """Return an adapter or raise a typed API error."""
        adapter = self._adapters.get(adapter_id)
        if adapter is None:
            raise AppError(
                constants.ERROR_ADAPTER_NOT_FOUND,
                f"Adapter '{adapter_id}' is not registered.",
                status_code=404,
            )
        return adapter

    def list(self) -> list[AdapterInfo]:
        """Return all registered adapters."""
        return [
            adapter.info(plugin_id=self._adapter_plugin_ids.get(adapter.id))
            for adapter in self._adapters.values()
        ]

    def adapters(self) -> Iterable[ModelAdapter]:
        """Yield adapter instances."""
        return self._adapters.values()

    def adapter_ids(self) -> set[str]:
        """Return registered adapter ids."""
        return set(self._adapters)

    def unregister(self, adapter_id: str) -> None:
        """Remove an adapter from the registry."""
        self._adapters.pop(adapter_id, None)
        self._adapter_plugin_ids.pop(adapter_id, None)


def create_default_registry() -> AdapterRegistry:
    """Create the built-in adapter registry."""
    registry = AdapterRegistry()
    registry.register(Sd15Img2ImgAdapter())
    registry.register(Sd15InpaintAdapter())
    registry.register(Sd15ControlNetInpaintAdapter())
    registry.register(Sd2InpaintAdapter())
    registry.register(SdxlImg2ImgAdapter())
    registry.register(SdxlInpaintAdapter())
    registry.register(SdxlControlNetInpaintAdapter())
    registry.register(SdxlFillControlNetUnionAdapter())
    registry.register(FluxFillAdapter())
    registry.register(FluxFillFp8Adapter())
    registry.register(ChromaInpaintAdapter())
    registry.register(ZImageInpaintAdapter())
    return registry
