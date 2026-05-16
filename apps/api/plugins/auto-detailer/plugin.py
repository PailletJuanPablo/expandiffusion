"""Auto detailer generation postprocessor."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from expandiffusion import constants
from expandiffusion.adapters.base import GenerationContext
from expandiffusion.errors import AppError, GenerationCancelled
from expandiffusion.postprocessors import (
    GenerationPostprocessor,
    GenerationPostprocessorContext,
)
from expandiffusion.schemas import ControlOption, ControlSchema

logger = logging.getLogger("uvicorn.error")

PARAM_ENABLED = "auto_detailer_enabled"
PARAM_TARGETS = "auto_detailer_targets"
PARAM_STRENGTH = "auto_detailer_strength"
PARAM_STEPS = "auto_detailer_steps"
PARAM_PADDING = "auto_detailer_padding"
PARAM_MASK_BLUR = "auto_detailer_mask_blur"
PARAM_MAX_REGIONS = "auto_detailer_max_regions"
PARAM_MIN_SIZE = "auto_detailer_min_size"

TARGET_FACES = "faces"
TARGET_BODIES = "bodies"
TARGET_FACES_AND_BODIES = "faces_and_bodies"
TARGET_OPTIONS = {TARGET_FACES, TARGET_BODIES, TARGET_FACES_AND_BODIES}

DEFAULT_ENABLED = False
DEFAULT_TARGETS = TARGET_FACES_AND_BODIES
DEFAULT_STRENGTH = 0.35
DEFAULT_STEPS = 12
DEFAULT_PADDING = 48
DEFAULT_MASK_BLUR = 8
DEFAULT_MAX_REGIONS = 4
DEFAULT_MIN_SIZE = 64
FACE_DETECTION_CONFIDENCE = 0.97
DETAIL_MAX_MEAN_DELTA = 48.0
DETAIL_MAX_P95_DELTA = 150.0
DETAIL_MAX_P95_SATURATION = 145.0

_FACE_DETECTOR_LOCK = threading.Lock()
_FACE_DETECTORS: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class DetectedRegion:
    """Image region selected for local refinement."""

    box: tuple[int, int, int, int]
    kind: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AutoDetailerSettings:
    """Validated plugin-owned runtime settings."""

    enabled: bool
    targets: str
    strength: float
    steps: int
    padding: int
    mask_blur: int
    max_regions: int
    min_size: int


class AutoDetailerPostprocessor(GenerationPostprocessor):
    """Run a localized second inpaint pass on detected face/body regions."""

    id = "auto-detailer"
    label = "Auto Detailer"
    description = "Second-pass inpaint refinement for faces and body anatomy."

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_ENABLED,
                label="Auto detailer",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_ENABLED,
            ),
            ControlSchema(
                id=PARAM_TARGETS,
                label="Detail targets",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_TARGETS,
                options=[
                    ControlOption(id=TARGET_FACES, label="faces"),
                    ControlOption(id=TARGET_BODIES, label="bodies"),
                    ControlOption(
                        id=TARGET_FACES_AND_BODIES,
                        label="faces + bodies",
                    ),
                ],
            ),
            ControlSchema(
                id=PARAM_STRENGTH,
                label="Detail strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STRENGTH,
                min=0.05,
                max=0.95,
                step=0.01,
            ),
            ControlSchema(
                id=PARAM_STEPS,
                label="Detail steps",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STEPS,
                min=2,
                max=80,
                step=1,
            ),
            ControlSchema(
                id=PARAM_PADDING,
                label="Detail padding",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_PADDING,
                min=0,
                max=512,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MASK_BLUR,
                label="Detail mask blur",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MASK_BLUR,
                min=0,
                max=64,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MAX_REGIONS,
                label="Detail regions",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MAX_REGIONS,
                min=1,
                max=16,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MIN_SIZE,
                label="Detail min size",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MIN_SIZE,
                min=32,
                max=1024,
                step=1,
            ),
        ]

    def generation_defaults(self) -> dict[str, object]:
        return {
            PARAM_ENABLED: DEFAULT_ENABLED,
            PARAM_TARGETS: DEFAULT_TARGETS,
            PARAM_STRENGTH: DEFAULT_STRENGTH,
            PARAM_STEPS: DEFAULT_STEPS,
            PARAM_PADDING: DEFAULT_PADDING,
            PARAM_MASK_BLUR: DEFAULT_MASK_BLUR,
            PARAM_MAX_REGIONS: DEFAULT_MAX_REGIONS,
            PARAM_MIN_SIZE: DEFAULT_MIN_SIZE,
        }

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        settings = _settings_from_context(context)
        if not settings.enabled:
            return context.generated

        generated = context.generated.convert("RGB")
        logger.info(
            "Auto detailer started: job_id=%s targets=%s steps=%s strength=%s",
            context.metadata.get("job_id"),
            settings.targets,
            settings.steps,
            settings.strength,
        )
        detail_regions = _detect_regions(
            generated,
            context.mask,
            settings.targets,
            settings.min_size,
        )[: settings.max_regions]
        if not detail_regions:
            _skip(
                "Auto detailer did not detect any target region inside the generated area.",
                context,
                detected_regions=0,
                refined_regions=0,
            )
            return generated

        refined = generated.copy()
        refined_regions = 0
        for index, region in enumerate(detail_regions):
            if context.is_cancelled():
                raise GenerationCancelled()
            context.progress(
                0.94 + ((index + 1) / len(detail_regions)) * 0.05,
                f"Auto detailer {region.kind}",
            )
            refined = _refine_region(refined, region, context)
            refined_regions += 1
        report = {
            "processor_id": self.id,
            "status": "applied",
            "detected_regions": len(detail_regions),
            "refined_regions": refined_regions,
            "regions": [
                _region_report(region)
                for region in detail_regions
            ],
        }
        context.diagnostics.append(report)
        logger.info(
            "Auto detailer applied: job_id=%s detected_regions=%s refined_regions=%s",
            context.metadata.get("job_id"),
            len(detail_regions),
            refined_regions,
        )
        return refined


def register(context) -> None:
    context.register_generation_postprocessor(AutoDetailerPostprocessor())


def _settings_from_context(context: GenerationPostprocessorContext) -> AutoDetailerSettings:
    targets = _string_parameter(context, PARAM_TARGETS, DEFAULT_TARGETS)
    if targets not in TARGET_OPTIONS:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Auto detailer parameter '{PARAM_TARGETS}' must be one of {sorted(TARGET_OPTIONS)}.",
            status_code=422,
        )
    return AutoDetailerSettings(
        enabled=_bool_parameter(context, PARAM_ENABLED, DEFAULT_ENABLED),
        targets=targets,
        strength=_float_parameter(context, PARAM_STRENGTH, DEFAULT_STRENGTH, 0.05, 0.95),
        steps=_int_parameter(context, PARAM_STEPS, DEFAULT_STEPS, 2, 80),
        padding=_int_parameter(context, PARAM_PADDING, DEFAULT_PADDING, 0, 512),
        mask_blur=_int_parameter(context, PARAM_MASK_BLUR, DEFAULT_MASK_BLUR, 0, 64),
        max_regions=_int_parameter(context, PARAM_MAX_REGIONS, DEFAULT_MAX_REGIONS, 1, 16),
        min_size=_int_parameter(context, PARAM_MIN_SIZE, DEFAULT_MIN_SIZE, 32, 1024),
    )


def _bool_parameter(context: GenerationPostprocessorContext, key: str, default: bool) -> bool:
    value = context.parameter(key, default)
    if isinstance(value, bool):
        return value
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"Auto detailer parameter '{key}' must be a boolean.",
        status_code=422,
    )


def _string_parameter(context: GenerationPostprocessorContext, key: str, default: str) -> str:
    value = context.parameter(key, default)
    if isinstance(value, str):
        return value
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"Auto detailer parameter '{key}' must be a string.",
        status_code=422,
    )


def _int_parameter(
    context: GenerationPostprocessorContext,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = context.parameter(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Auto detailer parameter '{key}' must be an integer.",
            status_code=422,
        )
    if not minimum <= value <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Auto detailer parameter '{key}' must be between {minimum} and {maximum}.",
            status_code=422,
        )
    return value


def _float_parameter(
    context: GenerationPostprocessorContext,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = context.parameter(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Auto detailer parameter '{key}' must be a number.",
            status_code=422,
        )
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Auto detailer parameter '{key}' must be between {minimum} and {maximum}.",
            status_code=422,
        )
    return parsed


def _refine_region(
    image: Image.Image,
    region: DetectedRegion,
    context: GenerationPostprocessorContext,
) -> Image.Image:
    settings = _settings_from_context(context)
    crop_box = _padded_box(region.box, settings.padding, image.size)
    crop = image.crop(crop_box)
    detail_mask = _region_mask(region, crop_box, crop.size)
    generated_mask = context.mask.convert("L").crop(crop_box)
    detail_mask = ImageChops.multiply(detail_mask, generated_mask)
    if not np.asarray(detail_mask).max():
        _fail(
            "Auto detailer detected a region, but its mask does not overlap generated pixels.",
            context,
            detected_regions=1,
            refined_regions=0,
        )

    detail_parameters = context.parameters.model_copy(
        update={
            "steps": settings.steps,
            "strength": settings.strength,
            "sample_count": 1,
            "inpaint_area": constants.INPAINT_AREA_ONLY_MASKED,
            "mask_crop_padding": settings.padding,
            "mask_blur": settings.mask_blur,
        }
    )
    detail_context = GenerationContext(
        source=crop,
        mask=detail_mask,
        parameters=detail_parameters,
        progress=context.progress,
        is_cancelled=context.is_cancelled,
        metadata=context.metadata,
    )
    detail_images = context.adapter.generate(detail_context)
    if not detail_images:
        _fail(
            "Auto detailer detail pass did not return an image.",
            context,
            detected_regions=1,
            refined_regions=0,
        )

    detail_image = detail_images[0].convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
    if _is_black_image(detail_image):
        _fail(
            (
                "Auto detailer detail pass returned a black image. This usually means the "
                "safety checker blocked the detail crop."
            ),
            context,
            detected_regions=1,
            refined_regions=0,
        )
    paste_mask = detail_mask.filter(
        ImageFilter.GaussianBlur(radius=settings.mask_blur)
    )
    blended_crop = Image.composite(detail_image, crop, paste_mask)
    _validate_refinement(region, crop, blended_crop, paste_mask, context)
    output = image.copy()
    output.paste(blended_crop, crop_box[:2])
    return output


def _detect_regions(
    image: Image.Image,
    generation_mask: Image.Image,
    targets: str,
    min_size: int,
) -> list[DetectedRegion]:
    regions: list[DetectedRegion] = []
    if targets in {
        TARGET_FACES,
        TARGET_FACES_AND_BODIES,
    }:
        regions.extend(_detect_faces_with_retinaface(image, min_size))
    if targets in {
        TARGET_BODIES,
        TARGET_FACES_AND_BODIES,
    }:
        try:
            import cv2
        except ImportError as exc:
            raise AppError(
                constants.ERROR_UNSUPPORTED_OPERATION,
                "Auto detailer body detection requires OpenCV, but cv2 is not installed.",
                status_code=500,
            ) from exc

        gray = np.asarray(image.convert("L"))
        regions.extend(
            _detect_with_cascade(cv2, gray, "haarcascade_upperbody.xml", min_size * 2, "body")
        )
        regions.extend(
            _detect_with_cascade(cv2, gray, "haarcascade_fullbody.xml", min_size * 2, "body")
        )

    generated = np.asarray(generation_mask.convert("L")) >= constants.WHITE_MASK_THRESHOLD
    filtered = [region for region in regions if _region_intersects_mask(region.box, generated)]
    return _dedupe_regions(filtered)


def _detect_faces_with_retinaface(image: Image.Image, min_size: int) -> list[DetectedRegion]:
    try:
        import torch
        from facexlib.detection import init_detection_model
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            (
                "Auto detailer face detection requires optional dependency 'facexlib'. "
                "Install the API extra: python -m pip install -e apps/api[face-restoration]"
            ),
            status_code=500,
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _FACE_DETECTOR_LOCK:
        detector = _FACE_DETECTORS.get(device)
        if detector is None:
            model_root = constants.DEFAULT_DATA_DIR / "plugins" / "auto-detailer"
            model_root.mkdir(parents=True, exist_ok=True)
            detector = init_detection_model(
                "retinaface_resnet50",
                half=False,
                device=device,
                model_rootpath=str(model_root),
            )
            _FACE_DETECTORS[device] = detector

    image_bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
    with torch.no_grad():
        detections = detector.detect_faces(image_bgr, FACE_DETECTION_CONFIDENCE)
    regions: list[DetectedRegion] = []
    for detection in detections:
        x0, y0, x1, y1, confidence = detection[:5]
        box = (int(x0), int(y0), int(x1), int(y1))
        if min(box[2] - box[0], box[3] - box[1]) >= min_size:
            regions.append(DetectedRegion(box, "face", float(confidence)))
    return regions


def _detect_with_cascade(
    cv2,
    gray: np.ndarray,
    cascade_name: str,
    min_size: int,
    kind: str,
) -> list[DetectedRegion]:
    cascade_path = Path(cv2.data.haarcascades) / cascade_name
    if not cascade_path.exists():
        return []
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        return []
    detections = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    return [
        DetectedRegion((int(x), int(y), int(x + width), int(y + height)), kind)
        for x, y, width, height in detections
    ]


def _region_intersects_mask(box: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x0, y0, x1, y1 = box
    height, width = mask.shape
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].mean() >= 0.15)


def _dedupe_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    output: list[DetectedRegion] = []
    for region in sorted(regions, key=lambda item: _area(item.box), reverse=True):
        if all(_iou(region.box, existing.box) < 0.45 for existing in output):
            output.append(region)
    return output


def _padded_box(
    box: tuple[int, int, int, int],
    padding: int,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = size
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _region_mask(
    region: DetectedRegion,
    crop_box: tuple[int, int, int, int],
    size: tuple[int, int],
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0, x1, y1 = region.box
    crop_x, crop_y = crop_box[:2]
    local_box = (x0 - crop_x, y0 - crop_y, x1 - crop_x, y1 - crop_y)
    if region.kind == "face":
        draw.ellipse(local_box, fill=255)
    else:
        draw.rounded_rectangle(local_box, radius=12, fill=255)
    return mask


def _region_report(region: DetectedRegion) -> dict[str, object]:
    report: dict[str, object] = {"kind": region.kind, "box": list(region.box)}
    if region.confidence is not None:
        report["confidence"] = round(region.confidence, 4)
    return report


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _is_black_image(image: Image.Image) -> bool:
    array = np.asarray(image.convert("RGB"))
    return bool(array.mean() < 3 and array.std() < 3)


def _validate_refinement(
    region: DetectedRegion,
    original: Image.Image,
    refined: Image.Image,
    mask: Image.Image,
    context: GenerationPostprocessorContext,
) -> None:
    mask_array = np.asarray(mask.convert("L")) >= constants.WHITE_MASK_THRESHOLD
    if not mask_array.any():
        return
    original_array = np.asarray(original.convert("RGB"), dtype=np.int16)
    refined_array = np.asarray(refined.convert("RGB"), dtype=np.int16)
    delta = np.abs(refined_array - original_array).max(axis=2)[mask_array]
    saturation = _rgb_saturation(refined_array.astype(np.uint8))[mask_array]
    mean_delta = float(delta.mean()) if delta.size else 0.0
    p95_delta = float(np.percentile(delta, 95)) if delta.size else 0.0
    p95_saturation = float(np.percentile(saturation, 95)) if saturation.size else 0.0
    if (
        mean_delta >= DETAIL_MAX_MEAN_DELTA
        and p95_delta >= DETAIL_MAX_P95_DELTA
        and p95_saturation >= DETAIL_MAX_P95_SATURATION
    ):
        _fail(
            (
                "Auto detailer rejected a destructive detail pass at "
                f"box={list(region.box)}. The detail crop changed color too much "
                "and would likely create a visible artifact."
            ),
            context,
            detected_regions=1,
            refined_regions=0,
            details={
                "region": {"kind": region.kind, "box": list(region.box)},
                "metrics": {
                    "mean_delta": round(mean_delta, 2),
                    "p95_delta": round(p95_delta, 2),
                    "p95_saturation": round(p95_saturation, 2),
                },
            },
        )


def _rgb_saturation(array: np.ndarray) -> np.ndarray:
    channel_max = array.max(axis=2).astype(np.float32)
    channel_min = array.min(axis=2).astype(np.float32)
    saturation = np.zeros(channel_max.shape, dtype=np.float32)
    nonzero = channel_max > 0
    saturation[nonzero] = (
        (channel_max[nonzero] - channel_min[nonzero]) / channel_max[nonzero]
    ) * 255
    return saturation


def _fail(
    message: str,
    context: GenerationPostprocessorContext,
    detected_regions: int,
    refined_regions: int,
    details: dict[str, object] | None = None,
) -> None:
    report = {
        "processor_id": AutoDetailerPostprocessor.id,
        "status": "failed",
        "message": message,
        "detected_regions": detected_regions,
        "refined_regions": refined_regions,
    }
    if details:
        report.update(details)
    context.diagnostics.append(report)
    logger.error(
        "Auto detailer failed: job_id=%s message=%s",
        context.metadata.get("job_id"),
        message,
    )
    raise AppError(constants.ERROR_GENERATION_FAILED, message, status_code=500)


def _skip(
    message: str,
    context: GenerationPostprocessorContext,
    detected_regions: int,
    refined_regions: int,
) -> None:
    context.diagnostics.append(
        {
            "processor_id": AutoDetailerPostprocessor.id,
            "status": "skipped",
            "message": message,
            "detected_regions": detected_regions,
            "refined_regions": refined_regions,
        }
    )
    logger.warning(
        "Auto detailer skipped: job_id=%s message=%s",
        context.metadata.get("job_id"),
        message,
    )


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = _area((x0, y0, x1, y1))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0
