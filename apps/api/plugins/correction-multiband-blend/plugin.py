"""Multiband blend correction postprocessor."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from expandiffusion import constants
from expandiffusion.errors import AppError
from expandiffusion.image_utils import match_generated_lighting_to_preserved_region, normalize_mask
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlSchema

PARAM_LEVELS = "correction_multiband_levels"
DEFAULT_LEVELS = 4


class MultibandBlendCorrectionPostprocessor(GenerationPostprocessor):
    """Blend original and generated images through a Laplacian pyramid."""

    id = "correction-multiband-blend"
    label = "Multiband blend"
    description = "Uses a multiscale Laplacian pyramid to soften mask transitions."
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION
    default_order = 50

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_LEVELS,
                label="Blend levels",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_LEVELS,
                min=1,
                max=6,
                step=1,
            )
        ]

    def generation_defaults(self) -> dict[str, int]:
        return {PARAM_LEVELS: DEFAULT_LEVELS}

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        levels = _bounded_int(context.parameter(PARAM_LEVELS, DEFAULT_LEVELS), 1, 6)
        original = context.original.convert("RGB")
        generated = context.generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        mask = normalize_mask(context.mask, original.size)
        matched = match_generated_lighting_to_preserved_region(original, generated, mask)
        return _multiband_blend(original, matched, mask, levels)


def _multiband_blend(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    requested_levels: int,
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
            "OpenCV is required for multiband blend correction.",
            status_code=500,
        ) from exc

    original_array = np.asarray(original).astype(np.float32)
    generated_array = np.asarray(generated).astype(np.float32)
    alpha = (mask_array.astype(np.float32) / 255.0)[:, :, np.newaxis]
    max_levels = max(1, int(math.log2(max(2, min(mask_array.shape)))) - 2)
    levels = min(requested_levels, max_levels)
    original_pyramid = _gaussian_pyramid(cv2, original_array, levels)
    generated_pyramid = _gaussian_pyramid(cv2, generated_array, levels)
    alpha_pyramid = _gaussian_pyramid(cv2, alpha, levels)
    original_laplacian = _laplacian_pyramid(cv2, original_pyramid)
    generated_laplacian = _laplacian_pyramid(cv2, generated_pyramid)
    blended = [
        generated_layer * mask_layer + original_layer * (1.0 - mask_layer)
        for generated_layer, original_layer, mask_layer in zip(
            generated_laplacian,
            original_laplacian,
            alpha_pyramid,
            strict=True,
        )
    ]
    reconstructed = blended[-1]
    for layer in reversed(blended[:-1]):
        width, height = layer.shape[1], layer.shape[0]
        reconstructed = cv2.pyrUp(reconstructed, dstsize=(width, height)) + layer
    return Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8), mode="RGB")


def _gaussian_pyramid(cv2, image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    current = image
    for _ in range(levels):
        if current.shape[0] < 2 or current.shape[1] < 2:
            break
        current = cv2.pyrDown(current)
        if image.ndim == 3 and current.ndim == 2:
            current = current[:, :, np.newaxis]
        pyramid.append(current)
    return pyramid


def _laplacian_pyramid(cv2, pyramid: list[np.ndarray]) -> list[np.ndarray]:
    layers = []
    for index in range(len(pyramid) - 1):
        current = pyramid[index]
        expanded = cv2.pyrUp(
            pyramid[index + 1],
            dstsize=(current.shape[1], current.shape[0]),
        )
        layers.append(current - expanded)
    layers.append(pyramid[-1])
    return layers


def _bounded_int(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, int | float):
        return min(max(int(value), minimum), maximum)
    return minimum


def register(context) -> None:
    context.register_generation_postprocessor(MultibandBlendCorrectionPostprocessor())
