"""Mask feather correction postprocessor."""

from __future__ import annotations

from PIL import Image

from expandiffusion import constants
from expandiffusion.image_utils import (
    compose_with_feathered_mask,
    match_generated_lighting_to_preserved_region,
    normalize_mask,
)
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlSchema

PARAM_RADIUS = "correction_mask_feather_radius"
DEFAULT_RADIUS = 12


class MaskFeatherCorrectionPostprocessor(GenerationPostprocessor):
    """Blend the corrected generated region back with a softened mask edge."""

    id = "correction-mask-feather"
    label = "Mask feather"
    description = "Softens the mask edge after matching generated lighting to preserved pixels."
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION
    default_order = 10

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_RADIUS,
                label="Feather radius",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_RADIUS,
                min=1,
                max=96,
                step=1,
            )
        ]

    def generation_defaults(self) -> dict[str, int]:
        return {PARAM_RADIUS: DEFAULT_RADIUS}

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        original = context.original.convert("RGB")
        generated = context.generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        mask = normalize_mask(context.mask, original.size)
        radius = _bounded_int(context.parameter(PARAM_RADIUS, DEFAULT_RADIUS), 1, 96)
        matched = match_generated_lighting_to_preserved_region(original, generated, mask)
        return compose_with_feathered_mask(matched, original, mask, radius)


def _bounded_int(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, int | float):
        return min(max(int(value), minimum), maximum)
    return minimum


def register(context) -> None:
    context.register_generation_postprocessor(MaskFeatherCorrectionPostprocessor())
