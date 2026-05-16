"""Histogram match correction postprocessor."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

from expandiffusion import constants
from expandiffusion.image_utils import get_preserved_boundary_region, normalize_mask
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlSchema

PARAM_STRENGTH = "correction_histogram_match_strength"
DEFAULT_STRENGTH = 0.85


class HistogramMatchCorrectionPostprocessor(GenerationPostprocessor):
    """Match generated RGB histograms to preserved boundary pixels."""

    id = "correction-histogram-match"
    label = "Histogram match"
    description = "Matches generated pixel histograms to preserved boundary samples."
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION
    default_order = 40

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_STRENGTH,
                label="Histogram strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STRENGTH,
                min=0.0,
                max=1.0,
                step=0.01,
            )
        ]

    def generation_defaults(self) -> dict[str, float]:
        return {PARAM_STRENGTH: DEFAULT_STRENGTH}

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        strength = _bounded_float(context.parameter(PARAM_STRENGTH, DEFAULT_STRENGTH), 0.0, 1.0)
        if strength <= 0:
            return context.generated.convert("RGB")
        original = context.original.convert("RGB")
        generated = context.generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        mask = normalize_mask(context.mask, original.size)
        return _match_histogram(original, generated, mask, strength)


def _match_histogram(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    strength: float,
) -> Image.Image:
    mask_array = np.asarray(mask.convert("L"))
    target_region = mask_array >= constants.WHITE_MASK_THRESHOLD
    if target_region.sum() < 32:
        return generated
    preserved_region = get_preserved_boundary_region(mask_array)
    if preserved_region.sum() < 32:
        return generated

    original_array = np.asarray(original).astype(np.float32)
    generated_array = np.asarray(generated).astype(np.float32)
    source = generated_array[target_region]
    reference = original_array[preserved_region]
    matched = match_histograms(source, reference, channel_axis=-1)
    corrected = source * (1.0 - strength) + matched * strength
    output = generated_array.copy()
    output[target_region] = corrected
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _bounded_float(value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, int | float):
        return min(max(float(value), minimum), maximum)
    return minimum


def register(context) -> None:
    context.register_generation_postprocessor(HistogramMatchCorrectionPostprocessor())
