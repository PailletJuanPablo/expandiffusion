"""Generation postprocessor plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from . import constants
from .adapters.base import CancelCheck, ModelAdapter, ProgressCallback
from .errors import AppError
from .schemas import ControlSchema, GenerationParameters, PostprocessorInfo


@dataclass(slots=True)
class GenerationPostprocessorContext:
    """Runtime inputs passed into generation postprocessors."""

    original: Image.Image
    generated: Image.Image
    mask: Image.Image
    parameters: GenerationParameters
    adapter: ModelAdapter
    progress: ProgressCallback
    is_cancelled: CancelCheck
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def parameter(self, key: str, default: Any = None) -> Any:
        """Return a plugin parameter from the generation payload."""
        extra = self.parameters.model_extra or {}
        if key in extra:
            return extra[key]
        return getattr(self.parameters, key, default)


class GenerationPostprocessor(ABC):
    """Base interface for plugins that refine generated images."""

    id: str
    label: str
    description: str = ""
    category: str = constants.POSTPROCESSOR_CATEGORY_GENERATION
    default_order: int = constants.DEFAULT_POSTPROCESSOR_ORDER

    def generation_controls(self) -> list[ControlSchema]:
        """Return frontend controls contributed by this postprocessor."""
        return []

    def generation_defaults(self) -> dict[str, Any]:
        """Return default generation parameters contributed by this postprocessor."""
        return {}

    @abstractmethod
    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        """Return a refined generated image."""

    def info(self, plugin_id: str | None = None) -> PostprocessorInfo:
        """Return serializable metadata for this postprocessor."""
        return PostprocessorInfo(
            id=self.id,
            label=self.label,
            description=self.description,
            plugin_id=plugin_id,
            category=self.category,
            default_order=self.default_order,
        )


class GenerationPostprocessorRegistry:
    """Mutable registry of plugin-provided generation postprocessors."""

    def __init__(self) -> None:
        self._processors: dict[str, GenerationPostprocessor] = {}
        self._processor_plugin_ids: dict[str, str] = {}

    def register(self, processor: GenerationPostprocessor, plugin_id: str) -> None:
        """Register a postprocessor by id."""
        if processor.id in self._processors:
            raise AppError(
                constants.ERROR_PLUGIN_LOAD_FAILED,
                f"Postprocessor '{processor.id}' is already registered.",
                status_code=500,
                details={"postprocessor_id": processor.id, "plugin_id": plugin_id},
            )
        self._processors[processor.id] = processor
        self._processor_plugin_ids[processor.id] = plugin_id

    def processors(self) -> Iterable[GenerationPostprocessor]:
        """Yield registered postprocessor instances."""
        return self._processors.values()

    def processors_for_category(self, category: str) -> list[GenerationPostprocessor]:
        """Return postprocessors matching a category in deterministic order."""
        return sorted(
            (
                processor
                for processor in self._processors.values()
                if processor.category == category
            ),
            key=lambda processor: (processor.default_order, processor.id),
        )

    def processors_except_category(self, category: str) -> list[GenerationPostprocessor]:
        """Return postprocessors outside a category in deterministic order."""
        return self.processors_except_categories({category})

    def processors_except_categories(
        self,
        categories: set[str],
    ) -> list[GenerationPostprocessor]:
        """Return postprocessors outside the supplied categories in deterministic order."""
        return sorted(
            (
                processor
                for processor in self._processors.values()
                if processor.category not in categories
            ),
            key=lambda processor: (processor.default_order, processor.id),
        )

    def correction_pipeline(self, processor_ids: list[str]) -> list[GenerationPostprocessor]:
        """Resolve and validate selected correction processors."""
        pipeline = []
        for processor_id in processor_ids:
            processor = self._processors.get(processor_id)
            if processor is None:
                raise AppError(
                    constants.ERROR_INVALID_GENERATION_PARAMETERS,
                    f"Correction '{processor_id}' is not available.",
                    status_code=422,
                    details={"processor_id": processor_id},
                )
            if processor.category != constants.POSTPROCESSOR_CATEGORY_CORRECTION:
                raise AppError(
                    constants.ERROR_INVALID_GENERATION_PARAMETERS,
                    f"Postprocessor '{processor_id}' is not a correction.",
                    status_code=422,
                    details={"processor_id": processor_id, "category": processor.category},
                )
            pipeline.append(processor)
        return pipeline

    def processor_infos(self) -> list[PostprocessorInfo]:
        """Return serializable metadata for all registered postprocessors."""
        return [
            processor.info(self.plugin_id_for(processor.id))
            for processor in sorted(
                self._processors.values(),
                key=lambda item: (item.category, item.default_order, item.id),
            )
        ]

    def plugin_id_for(self, processor_id: str) -> str | None:
        """Return the plugin owner for a postprocessor id."""
        return self._processor_plugin_ids.get(processor_id)

    def unregister(self, processor_id: str) -> None:
        """Remove a postprocessor from the registry."""
        self._processors.pop(processor_id, None)
        self._processor_plugin_ids.pop(processor_id, None)

    def unregister_plugin(self, plugin_id: str) -> None:
        """Remove all postprocessors owned by a plugin."""
        for processor_id, owner_id in list(self._processor_plugin_ids.items()):
            if owner_id == plugin_id:
                self.unregister(processor_id)
