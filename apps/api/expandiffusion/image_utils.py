"""Image helpers shared by adapters and API services."""

from __future__ import annotations

import base64
import io
import math
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

from . import constants
from .errors import AppError


def decode_data_url(data_url: str) -> Image.Image:
    """Decode a base64 data URL into a PIL image."""
    try:
        _, payload = data_url.split(constants.DATA_URL_SEPARATOR, 1)
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise AppError(
            constants.ERROR_INVALID_IMAGE,
            "The image payload is not a valid data URL.",
            status_code=422,
        ) from exc


def encode_png_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a PNG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{constants.PNG_MEDIA_TYPE};base64,{payload}"


def normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Normalize a mask so white regenerates and black preserves."""
    grayscale = mask.convert("L").resize(size, Image.Resampling.LANCZOS)
    return grayscale.point(
        lambda pixel: 255 if pixel >= constants.WHITE_MASK_THRESHOLD else 0,
        mode="L",
    )


def generation_mask_from_alpha(image: Image.Image) -> Image.Image:
    """Convert RGBA alpha into a Diffusers generation mask."""
    alpha = image.convert("RGBA").getchannel("A")
    return alpha.point(
        lambda pixel: 255 if pixel < constants.WHITE_MASK_THRESHOLD else 0,
        mode="L",
    )


def combine_generation_masks(*masks: Image.Image) -> Image.Image:
    """Merge generation masks so any white source pixel is regenerated."""
    if not masks:
        raise AppError(
            constants.ERROR_INVALID_IMAGE,
            "At least one generation mask is required.",
            status_code=422,
        )
    size = masks[0].size
    arrays = [np.asarray(normalize_mask(mask, size)) for mask in masks]
    return Image.fromarray(np.maximum.reduce(arrays).astype(np.uint8), mode="L")


def expand_mask_to_block_grid(mask: Image.Image, block_size: int = 8) -> Image.Image:
    """Expand generation mask to the block grid used by Diffusers latents."""
    mask_array = np.array(mask.convert("L"))
    height, width = mask_array.shape
    padded_height = math.ceil(height / block_size) * block_size
    padded_width = math.ceil(width / block_size) * block_size
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = mask_array
    blocks = padded.reshape(
        padded_height // block_size,
        block_size,
        padded_width // block_size,
        block_size,
    )
    expanded = blocks.max(axis=(1, 3)).repeat(block_size, axis=0).repeat(block_size, axis=1)
    return Image.fromarray(expanded[:height, :width], mode="L")


def feather_diffusion_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Soften the Diffusers repaint boundary without changing full empty/full known cases."""
    mask_array = np.asarray(mask.convert("L"))
    if radius <= 0:
        return mask.convert("L")
    if (mask_array >= constants.WHITE_MASK_THRESHOLD).all():
        return mask.convert("L")
    if (mask_array < constants.WHITE_MASK_THRESHOLD).all():
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))


def prepare_source_image(
    image: Image.Image,
    mask: Image.Image,
    fill_mode: str,
    size: tuple[int, int],
) -> Image.Image:
    """Prepare source like stablediffusion-infinity: known alpha guides empty pixels."""
    resized = clear_transparent_rgb(image.convert("RGBA").resize(size, Image.Resampling.LANCZOS))
    generation_mask = normalize_mask(mask, size)
    known_mask = Image.eval(generation_mask, lambda pixel: 255 - pixel)
    image_array = np.array(resized.convert("RGB"))
    known_array = np.array(known_mask)
    fill_mode = normalize_fill_mode(fill_mode)
    if fill_mode == constants.FILL_TRANSPARENT:
        return resized.convert("RGB")
    if fill_mode == constants.FILL_GAUSSIAN_NOISE:
        return Image.fromarray(fill_with_noise(image_array, known_array, gaussian=True), mode="RGB")
    if fill_mode == constants.FILL_PERLIN_NOISE:
        return Image.fromarray(fill_with_perlin(image_array, known_array), mode="RGB")
    if fill_mode == constants.FILL_OPENCV_TELEA:
        return fill_with_opencv(image_array, generation_mask, method="telea")
    if fill_mode == constants.FILL_OPENCV_NS:
        return fill_with_opencv(image_array, generation_mask, method="ns")
    if fill_mode == constants.FILL_PATCHMATCH:
        patched = fill_with_patchmatch(image_array, generation_mask)
        if patched is not None:
            return Image.fromarray(patched, mode="RGB")
    return Image.fromarray(fill_with_edge_pad(image_array, known_array), mode="RGB")


def clear_transparent_rgb(image: Image.Image) -> Image.Image:
    """Match the original canvas buffer: empty alpha carries black RGB, not white."""
    array = np.array(image.convert("RGBA"))
    transparent = array[:, :, 3] < constants.WHITE_MASK_THRESHOLD
    array[transparent, 0:3] = 0
    array[transparent, 3] = 0
    return Image.fromarray(array, mode="RGBA")


def normalize_fill_mode(fill_mode: str) -> str:
    """Map legacy API values to the original stablediffusion-infinity names."""
    aliases = {
        constants.LEGACY_FILL_EDGE_EXTEND: constants.FILL_EDGE_EXTEND,
        constants.LEGACY_FILL_GAUSSIAN_NOISE: constants.FILL_GAUSSIAN_NOISE,
        constants.LEGACY_FILL_PERLIN_NOISE: constants.FILL_PERLIN_NOISE,
        constants.LEGACY_FILL_OPENCV_TELEA: constants.FILL_OPENCV_TELEA,
        constants.LEGACY_FILL_OPENCV_NS: constants.FILL_OPENCV_NS,
    }
    return aliases.get(fill_mode, fill_mode)


def fill_with_edge_pad(image: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Propagate nearest known edge colors into empty pixels."""
    known = known_mask > 0
    if not known.any() or known.all():
        return image
    output = image.astype(np.float32).copy()
    visited = known.copy()
    counts = np.zeros(known.shape, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()
    height, width = known.shape
    for y in range(height):
        for x in range(width):
            if not known[y, x]:
                continue
            for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and not known[neighbor_y, neighbor_x]
                ):
                    queue.append((y, x))
                    break
    while queue:
        y, x = queue.popleft()
        for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_y = y + offset_y
            neighbor_x = x + offset_x
            if (
                0 <= neighbor_y < height
                and 0 <= neighbor_x < width
                and not visited[neighbor_y, neighbor_x]
            ):
                output[neighbor_y, neighbor_x] = (
                    output[neighbor_y, neighbor_x] * counts[neighbor_y, neighbor_x]
                    + output[y, x]
                ) / (counts[neighbor_y, neighbor_x] + 1)
                counts[neighbor_y, neighbor_x] += 1
                visited[neighbor_y, neighbor_x] = True
                queue.append((neighbor_y, neighbor_x))
    return np.clip(output, 0, 255).astype(np.uint8)


def fill_with_noise(image: np.ndarray, known_mask: np.ndarray, gaussian: bool) -> np.ndarray:
    """Fill unknown pixels with noise while preserving known pixels."""
    height, width = known_mask.shape
    if gaussian:
        values = np.random.default_rng().normal(128, 40, (height, width, 3))
    else:
        values = np.random.default_rng().integers(0, 256, (height, width, 3))
    noise = np.clip(values, 0, 255).astype(np.uint8)
    known = (known_mask > 0)[:, :, np.newaxis]
    return np.where(known, image, noise).astype(np.uint8)


def fill_with_perlin(image: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    """Fill unknown pixels with smooth noise while preserving known pixels."""
    height, width = known_mask.shape
    small_width = max(2, math.ceil(width / 32))
    small_height = max(2, math.ceil(height / 32))
    values = np.random.default_rng().integers(48, 208, (small_height, small_width, 3))
    small = Image.fromarray(values.astype(np.uint8), mode="RGB")
    noise = np.array(
        small.resize((width, height), Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(radius=10)
        )
    )
    known = (known_mask > 0)[:, :, np.newaxis]
    return np.where(known, image, noise).astype(np.uint8)


def fill_with_opencv(image: np.ndarray, generation_mask: Image.Image, method: str) -> Image.Image:
    """Fill unknown pixels using OpenCV inpainting."""
    try:
        import cv2
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "OpenCV is required for the selected fill preprocessor.",
            status_code=500,
        ) from exc
    cv_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    mask_array = np.array(generation_mask)
    inpainted = cv2.inpaint(image, mask_array, 5, cv_method)
    return Image.fromarray(inpainted, mode="RGB")


def fill_with_patchmatch(image: np.ndarray, generation_mask: Image.Image) -> np.ndarray | None:
    """Use PyPatchMatch when available, otherwise fall back to edge padding."""
    try:
        from PyPatchMatch import patch_match
    except ImportError:
        try:
            import patch_match
        except ImportError:
            return None
    return patch_match.inpaint(image, mask=np.array(generation_mask), patch_size=3)


def compose_generation_result(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    result_mode: str,
) -> Image.Image:
    """Choose whether known pixels are kept or the full generated selection is used."""
    original_rgb = original.convert("RGB")
    generated_rgb = generated.convert("RGB").resize(original_rgb.size, Image.Resampling.LANCZOS)
    normalized_mask = normalize_mask(mask, original_rgb.size)
    if result_mode == constants.RESULT_MODE_PRESERVE_KNOWN:
        return Image.composite(generated_rgb, original_rgb, normalized_mask).convert("RGB")
    if result_mode == constants.RESULT_MODE_FEATHER_KNOWN:
        return compose_with_feathered_mask(generated_rgb, original_rgb, normalized_mask)
    if result_mode == constants.RESULT_MODE_RESTORE_ORIGINAL_SOFT:
        return compose_with_restored_original(generated_rgb, original, normalized_mask)
    return generated_rgb


def compose_with_restored_original(
    generated: Image.Image,
    original: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Restore original pixels only outside the active generation mask."""
    original_rgba = original.convert("RGBA")
    generated_rgb = generated.convert("RGB").resize(original_rgba.size, Image.Resampling.LANCZOS)
    alpha_array = np.asarray(original_rgba.getchannel("A"))
    generation_array = np.asarray(normalize_mask(mask, original_rgba.size))
    restore_array = np.where(
        (alpha_array >= constants.WHITE_MASK_THRESHOLD)
        & (generation_array < constants.WHITE_MASK_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)
    restore_mask = Image.fromarray(restore_array, mode="L")
    if (restore_array < constants.WHITE_MASK_THRESHOLD).all():
        return generated_rgb

    original_array = np.asarray(original_rgba.convert("RGB"))
    restored_source = Image.fromarray(
        fill_with_edge_pad(original_array, restore_array),
        mode="RGB",
    )
    if (restore_array >= constants.WHITE_MASK_THRESHOLD).all():
        return restored_source

    feathered_mask = restore_mask.filter(
        ImageFilter.GaussianBlur(radius=constants.ORIGINAL_RESTORE_FEATHER_RADIUS)
    )
    return Image.composite(restored_source, generated_rgb, feathered_mask).convert("RGB")


def compose_with_feathered_mask(
    generated: Image.Image,
    original: Image.Image,
    mask: Image.Image,
    max_feather_radius: int = constants.COMPOSITION_FEATHER_RADIUS,
) -> Image.Image:
    """Composite generated pixels over original pixels with a soft seam."""
    mask_array = np.asarray(mask.convert("L"))
    if (mask_array >= constants.WHITE_MASK_THRESHOLD).all():
        return generated.convert("RGB")
    if (mask_array < constants.WHITE_MASK_THRESHOLD).all():
        return original.convert("RGB")
    feather_radius = min(
        max_feather_radius,
        max(1, min(mask_array.shape) // 16),
    )
    feathered_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return Image.composite(
        generated.convert("RGB"),
        original.convert("RGB"),
        feathered_mask,
    ).convert("RGB")


def measure_seam_discontinuity(
    original: Image.Image,
    candidate: Image.Image,
    mask: Image.Image,
) -> dict[str, float | int]:
    """Measure color and luminance mismatch across the preserved/generated seam."""
    original_array = np.asarray(original.convert("RGB")).astype(np.float32)
    candidate_array = np.asarray(
        candidate.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    ).astype(np.float32)
    mask_array = np.asarray(normalize_mask(mask, original.size))
    preserved_region = get_preserved_boundary_region(mask_array)
    generated_region = get_generated_boundary_region(mask_array)
    sample_count = int(min(preserved_region.sum(), generated_region.sum()))
    if sample_count < 32:
        return {"rgb_delta": 0.0, "luma_delta": 0.0, "samples": sample_count}

    preserved_mean = original_array[preserved_region].mean(axis=0)
    generated_mean = candidate_array[generated_region].mean(axis=0)
    rgb_delta = float(np.linalg.norm(preserved_mean - generated_mean))
    preserved_luma = _luminance(preserved_mean)
    generated_luma = _luminance(generated_mean)
    return {
        "rgb_delta": round(rgb_delta, 4),
        "luma_delta": round(abs(float(preserved_luma - generated_luma)), 4),
        "samples": sample_count,
    }


def match_generated_lighting_to_preserved_region(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Match generated lighting/color to preserved pixels adjacent to the generation mask."""
    original_rgba = original.convert("RGBA")
    original_rgb = original_rgba.convert("RGB")
    generated_rgb = generated.convert("RGB").resize(original_rgb.size, Image.Resampling.LANCZOS)
    original_array = np.asarray(original_rgb).astype(np.float32)
    generated_array = np.asarray(generated_rgb).astype(np.float32)
    mask_array = np.asarray(normalize_mask(mask, original_rgb.size))
    target_region = mask_array >= constants.WHITE_MASK_THRESHOLD
    preserved_region = get_preserved_boundary_region(mask_array)
    generated_region = get_generated_boundary_region(mask_array)
    if preserved_region.sum() < 32 or generated_region.sum() < 32:
        return generated
    original_samples = original_array[preserved_region]
    generated_samples = generated_array[generated_region]
    original_mean = original_samples.mean(axis=0)
    generated_mean = generated_samples.mean(axis=0)
    original_std = original_samples.std(axis=0)
    generated_std = generated_samples.std(axis=0)
    scale = np.ones(3, dtype=np.float32)
    textured_channels = (original_std >= 1.0) & (generated_std >= 1.0)
    scale[textured_channels] = np.clip(
        original_std[textured_channels] / generated_std[textured_channels],
        0.45,
        2.2,
    )
    boundary_corrected = (generated_array - generated_mean) * scale + original_mean
    corrected = generated_array.copy()
    corrected[target_region] = boundary_corrected[target_region]
    corrected = _match_generated_context_tone(
        original_array,
        original_rgba,
        corrected,
        mask_array,
    )
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8), mode="RGB")


def _match_generated_context_tone(
    original_array: np.ndarray,
    original_rgba: Image.Image,
    generated_array: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    target_region = mask_array >= constants.WHITE_MASK_THRESHOLD
    if target_region.sum() < 32:
        return generated_array
    alpha_array = np.asarray(original_rgba.getchannel("A"))
    context_region = (
        (mask_array < constants.WHITE_MASK_THRESHOLD)
        & (alpha_array >= constants.WHITE_MASK_THRESHOLD)
    )
    if context_region.sum() < 32:
        context_region = mask_array < constants.WHITE_MASK_THRESHOLD
    if context_region.sum() < 32:
        return generated_array

    context_samples = original_array[context_region]
    generated_samples = generated_array[target_region]
    context_mean = context_samples.mean(axis=0)
    generated_mean = generated_samples.mean(axis=0)
    context_std = context_samples.std(axis=0)
    generated_std = generated_samples.std(axis=0)
    scale = np.ones(3, dtype=np.float32)
    textured_channels = (context_std >= 1.0) & (generated_std >= 1.0)
    scale[textured_channels] = np.clip(
        context_std[textured_channels] / generated_std[textured_channels],
        0.45,
        2.2,
    )
    tone_matched = (generated_array - generated_mean) * scale + context_mean
    distance_alpha = _generated_context_tone_alpha(
        target_region,
        constants.LIGHTING_CONTEXT_TONE_RAMP,
        constants.LIGHTING_CONTEXT_TONE_STRENGTH,
    )
    output = generated_array.copy()
    output[target_region] = (
        generated_array[target_region] * (1.0 - distance_alpha[target_region, np.newaxis])
        + tone_matched[target_region] * distance_alpha[target_region, np.newaxis]
    )
    return output


def _generated_context_tone_alpha(
    target_region: np.ndarray,
    ramp: int,
    strength: float,
) -> np.ndarray:
    try:
        from scipy import ndimage

        distance = ndimage.distance_transform_edt(target_region)
    except Exception:
        distance = _approximate_binary_distance(target_region, ramp)
    return np.clip(distance / max(1, ramp), 0.0, 1.0).astype(np.float32) * strength


def _approximate_binary_distance(target_region: np.ndarray, max_distance: int) -> np.ndarray:
    current = target_region.astype(bool)
    distance = np.zeros(current.shape, dtype=np.float32)
    for step in range(1, max(1, max_distance) + 1):
        if not current.any():
            break
        eroded = (
            np.asarray(
                Image.fromarray(np.where(current, 255, 0).astype(np.uint8), mode="L").filter(
                    ImageFilter.MinFilter(3)
                )
            )
            >= constants.WHITE_MASK_THRESHOLD
        )
        edge = current & ~eroded
        distance[edge] = step
        current = eroded
    distance[current] = max_distance
    return distance


def _luminance(rgb: np.ndarray) -> float:
    return float(rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722)


def get_preserved_boundary_region(mask_array: np.ndarray) -> np.ndarray:
    preserved = mask_array < constants.WHITE_MASK_THRESHOLD
    generated = mask_array >= constants.WHITE_MASK_THRESHOLD
    if not preserved.any() or not generated.any():
        return np.zeros(mask_array.shape, dtype=bool)
    dilated_generated = np.asarray(
        Image.fromarray(mask_array, mode="L").filter(ImageFilter.MaxFilter(33))
    ) >= constants.WHITE_MASK_THRESHOLD
    boundary = preserved & dilated_generated
    if boundary.sum() >= 32:
        return boundary
    return preserved


def get_generated_boundary_region(mask_array: np.ndarray) -> np.ndarray:
    preserved = mask_array < constants.WHITE_MASK_THRESHOLD
    generated = mask_array >= constants.WHITE_MASK_THRESHOLD
    if not preserved.any() or not generated.any():
        return np.zeros(mask_array.shape, dtype=bool)
    preserved_mask = np.where(preserved, 255, 0).astype(np.uint8)
    dilated_preserved = np.asarray(
        Image.fromarray(preserved_mask, mode="L").filter(ImageFilter.MaxFilter(33))
    ) >= constants.WHITE_MASK_THRESHOLD
    boundary = generated & dilated_preserved
    if boundary.sum() >= 32:
        return boundary
    return generated
