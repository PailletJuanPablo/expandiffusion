"""Border blend correction postprocessor."""

from __future__ import annotations

import numpy as np
from PIL import Image

from expandiffusion import constants
from expandiffusion.errors import AppError
from expandiffusion.image_utils import (
    compose_with_feathered_mask,
    match_generated_lighting_to_preserved_region,
    normalize_mask,
)
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlSchema

PARAM_MARGIN = "correction_border_blend_margin"
DEFAULT_MARGIN = 16


class BorderBlendCorrectionPostprocessor(GenerationPostprocessor):
    """Use gradient-domain cloning around the generated/preserved boundary."""

    id = "correction-border-blend"
    label = "Border blend"
    description = "Runs OpenCV mixed seamless clone over the generated mask boundary."
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION
    default_order = 20

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_MARGIN,
                label="Blend margin",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MARGIN,
                min=4,
                max=96,
                step=1,
            )
        ]

    def generation_defaults(self) -> dict[str, int]:
        return {PARAM_MARGIN: DEFAULT_MARGIN}

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        margin = _bounded_int(context.parameter(PARAM_MARGIN, DEFAULT_MARGIN), 4, 96)
        original = context.original.convert("RGB")
        generated = context.generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        mask = normalize_mask(context.mask, original.size)
        matched = match_generated_lighting_to_preserved_region(original, generated, mask)
        return _blend_border(original, matched, mask, margin)


def _blend_border(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    margin: int,
) -> Image.Image:
    mask_array = np.asarray(mask.convert("L"))
    generated_region = mask_array >= constants.WHITE_MASK_THRESHOLD
    if not generated_region.any():
        return original.convert("RGB")
    if generated_region.all():
        return generated.convert("RGB")

    try:
        import cv2
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "OpenCV is required for border blend correction.",
            status_code=500,
        ) from exc

    composite = Image.composite(generated, original, mask).convert("RGB")
    padding = max(32, margin * 2)
    source = np.pad(
        np.asarray(generated),
        ((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    ).astype(np.uint8)
    target = np.pad(
        np.asarray(composite),
        ((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    ).astype(np.uint8)
    padded_mask = np.pad(
        mask_array.astype(np.uint8),
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )
    y_indices, x_indices = np.nonzero(padded_mask >= constants.WHITE_MASK_THRESHOLD)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return composite

    y0 = max(int(y_indices.min()) - margin, 0)
    y1 = min(int(y_indices.max()) + margin + 1, padded_mask.shape[0])
    x0 = max(int(x_indices.min()) - margin, 0)
    x1 = min(int(x_indices.max()) + margin + 1, padded_mask.shape[1])
    source_patch = source[y0:y1, x0:x1]
    mask_patch = padded_mask[y0:y1, x0:x1].copy()
    if source_patch.shape[0] < 3 or source_patch.shape[1] < 3:
        return composite

    mask_patch[0, :] = 0
    mask_patch[-1, :] = 0
    mask_patch[:, 0] = 0
    mask_patch[:, -1] = 0
    if mask_patch.max() < constants.WHITE_MASK_THRESHOLD:
        return composite

    center = (x0 + source_patch.shape[1] // 2, y0 + source_patch.shape[0] // 2)
    try:
        blended = cv2.seamlessClone(source_patch, target, mask_patch, center, cv2.MIXED_CLONE)
    except cv2.error as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "OpenCV seamless clone failed for border blend correction.",
            status_code=500,
            details={"reason": str(exc)},
        ) from exc

    height, width = mask_array.shape
    cropped = blended[padding : padding + height, padding : padding + width]
    blended_image = Image.fromarray(cropped, mode="RGB")
    return compose_with_feathered_mask(blended_image, original, mask)


def _bounded_int(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, int | float):
        return min(max(int(value), minimum), maximum)
    return minimum


def register(context) -> None:
    context.register_generation_postprocessor(BorderBlendCorrectionPostprocessor())
