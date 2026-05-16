"""Color match correction postprocessor."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color

from expandiffusion import constants
from expandiffusion.image_utils import (
    get_generated_boundary_region,
    get_preserved_boundary_region,
    normalize_mask,
)
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlSchema

PARAM_STRENGTH = "correction_color_match_strength"
DEFAULT_STRENGTH = 1.0


class ColorMatchCorrectionPostprocessor(GenerationPostprocessor):
    """Transfer LAB boundary color statistics into generated pixels."""

    id = "correction-color-match"
    label = "Color match"
    description = "Matches generated LAB color statistics to the preserved boundary."
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION
    default_order = 30

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_STRENGTH,
                label="Color strength",
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
        return _match_lab_color(original, generated, mask, strength)


def _match_lab_color(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    strength: float,
) -> Image.Image:
    mask_array = np.asarray(mask.convert("L"))
    target_region = mask_array >= constants.WHITE_MASK_THRESHOLD
    if not target_region.any():
        return generated
    preserved_region = get_preserved_boundary_region(mask_array)
    generated_region = get_generated_boundary_region(mask_array)
    if preserved_region.sum() < 32 or generated_region.sum() < 32:
        return generated

    original_array = np.asarray(original).astype(np.float32) / 255.0
    generated_array = np.asarray(generated).astype(np.float32) / 255.0
    original_lab = color.rgb2lab(original_array)
    generated_lab = color.rgb2lab(generated_array)
    original_samples = original_lab[preserved_region]
    generated_samples = generated_lab[generated_region]
    original_mean = original_samples.mean(axis=0)
    generated_mean = generated_samples.mean(axis=0)
    original_std = np.maximum(original_samples.std(axis=0), 0.001)
    generated_std = np.maximum(generated_samples.std(axis=0), 0.001)
    scale = np.clip(original_std / generated_std, 0.35, 2.8)
    matched_lab = (generated_lab - generated_mean) * scale + original_mean
    matched_rgb = np.clip(color.lab2rgb(matched_lab), 0.0, 1.0)
    corrected = generated_array * (1.0 - strength) + matched_rgb * strength
    output = generated_array.copy()
    output[target_region] = corrected[target_region]
    return Image.fromarray(np.clip(output * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _bounded_float(value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, int | float):
        return min(max(float(value), minimum), maximum)
    return minimum


def register(context) -> None:
    context.register_generation_postprocessor(ColorMatchCorrectionPostprocessor())
