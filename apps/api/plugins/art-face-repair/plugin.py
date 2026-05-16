"""Art face repair generation postprocessor."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
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
from expandiffusion.schemas import ControlSchema

logger = logging.getLogger("uvicorn.error")

PARAM_ENABLED = "art_face_repair_enabled"
PARAM_STRENGTH = "art_face_repair_strength"
PARAM_STEPS = "art_face_repair_steps"
PARAM_MAX_REGIONS = "art_face_repair_max_regions"
PARAM_MIN_AREA = "art_face_repair_min_area"
PARAM_MASK_SCALE = "art_face_repair_mask_scale"
PARAM_CONTEXT_SCALE = "art_face_repair_context_scale"
PARAM_MASK_BLUR = "art_face_repair_mask_blur"

DEFAULT_ENABLED = False
DEFAULT_STRENGTH = 0.62
DEFAULT_STEPS = 18
DEFAULT_MAX_REGIONS = 2
DEFAULT_MIN_AREA = 1200
DEFAULT_MASK_SCALE = 1.35
DEFAULT_CONTEXT_SCALE = 4.0
DEFAULT_MASK_BLUR = 8

DETECTOR_MODEL_ID = "florence-community/Florence-2-base-ft"
DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"
DETECTION_QUERIES = (
    "human face",
    "face",
    "head",
    "human head",
    "painted face",
    "painted human face",
    "profile face",
    "side profile face",
    "portrait face",
)
MIN_FACE_SIDE = 12
DETAIL_MIN_PROCESS_SIDE = 512
DETAIL_MAX_PROCESS_SIDE = 1024
REGION_IOU_THRESHOLD = 0.25
REGION_CONTAINMENT_THRESHOLD = 0.55
DARK_HALO_LUMA_THRESHOLD = 35.0
DARK_HALO_MEAN_DROP = 28.0
DARK_HALO_RATIO_INCREASE = 0.18

REPAIR_PROMPT_SUFFIX = (
    "coherent painted human face, correct eyes nose and mouth, soft classical "
    "features, same pigment texture, same lighting, same colors, painted artwork"
)
REPAIR_NEGATIVE_SUFFIX = (
    "photorealistic, modern photo, deformed face, distorted eyes, broken mouth, "
    "extra eyes, extra nose, malformed head, blurry face, plastic skin, black halo, "
    "dark stain"
)

_DETECTOR_LOCK = threading.Lock()
_DETECTOR: tuple[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class ArtFaceRepairSettings:
    enabled: bool
    strength: float
    steps: int
    max_regions: int
    min_area: int
    mask_scale: float
    context_scale: float
    mask_blur: int


@dataclass(frozen=True, slots=True)
class ArtFaceRegion:
    box: tuple[int, int, int, int]
    label: str


class ArtFaceRepairPostprocessor(GenerationPostprocessor):
    """Repair malformed painted faces using semantic detection and local repaint."""

    id = "art-face-repair"
    label = "Art Face Repair"
    description = "Florence-2 guided local repaint for malformed faces in paintings."

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_ENABLED,
                label="Art face repair",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_ENABLED,
            ),
            ControlSchema(
                id=PARAM_STRENGTH,
                label="Art face strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STRENGTH,
                min=0.05,
                max=0.95,
                step=0.01,
            ),
            ControlSchema(
                id=PARAM_STEPS,
                label="Art face steps",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STEPS,
                min=2,
                max=80,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MAX_REGIONS,
                label="Art face regions",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MAX_REGIONS,
                min=1,
                max=8,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MIN_AREA,
                label="Art face min area",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MIN_AREA,
                min=64,
                max=20000,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MASK_SCALE,
                label="Art face mask scale",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MASK_SCALE,
                min=1.0,
                max=3.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_CONTEXT_SCALE,
                label="Art face context",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_CONTEXT_SCALE,
                min=1.0,
                max=6.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_MASK_BLUR,
                label="Art face mask blur",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MASK_BLUR,
                min=0,
                max=64,
                step=1,
            ),
        ]

    def generation_defaults(self) -> dict[str, object]:
        return {
            PARAM_ENABLED: DEFAULT_ENABLED,
            PARAM_STRENGTH: DEFAULT_STRENGTH,
            PARAM_STEPS: DEFAULT_STEPS,
            PARAM_MAX_REGIONS: DEFAULT_MAX_REGIONS,
            PARAM_MIN_AREA: DEFAULT_MIN_AREA,
            PARAM_MASK_SCALE: DEFAULT_MASK_SCALE,
            PARAM_CONTEXT_SCALE: DEFAULT_CONTEXT_SCALE,
            PARAM_MASK_BLUR: DEFAULT_MASK_BLUR,
        }

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        settings = _settings_from_context(context)
        if not settings.enabled:
            return context.generated

        generated = context.generated.convert("RGB")
        regions = _detect_art_face_regions(
            generated,
            context.mask,
            settings.min_area,
        )[: settings.max_regions]
        if not regions:
            _skip(
                "Art face repair did not detect any large painted face inside the generated area.",
                context,
                detected_regions=0,
                refined_regions=0,
            )
            return generated

        refined = generated.copy()
        reports: list[dict[str, object]] = []
        refined_count = 0
        for index, region in enumerate(regions):
            if context.is_cancelled():
                raise GenerationCancelled()
            context.progress(
                0.94 + ((index + 1) / len(regions)) * 0.05,
                "Art face repair",
            )
            refined, report = _repair_region(refined, region, settings, context)
            reports.append(report)
            if report["status"] == "applied":
                refined_count += 1

        status = "applied" if refined_count else "skipped"
        context.diagnostics.append(
            {
                "processor_id": self.id,
                "status": status,
                "detected_regions": len(regions),
                "refined_regions": refined_count,
                "regions": reports,
            }
        )
        logger.info(
            "Art face repair finished: job_id=%s detected_regions=%s refined_regions=%s",
            context.metadata.get("job_id"),
            len(regions),
            refined_count,
        )
        return refined


def register(context) -> None:
    context.register_generation_postprocessor(ArtFaceRepairPostprocessor())


def _settings_from_context(context: GenerationPostprocessorContext) -> ArtFaceRepairSettings:
    return ArtFaceRepairSettings(
        enabled=_bool_parameter(context, PARAM_ENABLED, DEFAULT_ENABLED),
        strength=_float_parameter(context, PARAM_STRENGTH, DEFAULT_STRENGTH, 0.05, 0.95),
        steps=_int_parameter(context, PARAM_STEPS, DEFAULT_STEPS, 2, 80),
        max_regions=_int_parameter(context, PARAM_MAX_REGIONS, DEFAULT_MAX_REGIONS, 1, 8),
        min_area=_int_parameter(context, PARAM_MIN_AREA, DEFAULT_MIN_AREA, 64, 20000),
        mask_scale=_float_parameter(context, PARAM_MASK_SCALE, DEFAULT_MASK_SCALE, 1.0, 3.0),
        context_scale=_float_parameter(
            context,
            PARAM_CONTEXT_SCALE,
            DEFAULT_CONTEXT_SCALE,
            1.0,
            6.0,
        ),
        mask_blur=_int_parameter(context, PARAM_MASK_BLUR, DEFAULT_MASK_BLUR, 0, 64),
    )


def _repair_region(
    image: Image.Image,
    region: ArtFaceRegion,
    settings: ArtFaceRepairSettings,
    context: GenerationPostprocessorContext,
) -> tuple[Image.Image, dict[str, object]]:
    crop_box = _scaled_box(region.box, settings.context_scale, image.size)
    repair_box = _scaled_box(region.box, settings.mask_scale, image.size)
    crop = image.crop(crop_box)
    repair_mask = _ellipse_mask(repair_box, crop_box, crop.size)
    generated_mask = context.mask.convert("L").crop(crop_box)
    repair_mask = ImageChops.multiply(repair_mask, generated_mask)
    if not np.asarray(repair_mask).max():
        return image, _region_report(region, "skipped_no_mask_overlap", crop_box)

    detail_crop, detail_mask = _detail_inputs(crop, repair_mask)
    detail_parameters = context.parameters.model_copy(
        update={
            "prompt": _repair_prompt(context.parameters.prompt),
            "negative_prompt": _repair_negative_prompt(context.parameters.negative_prompt),
            "width": detail_crop.width,
            "height": detail_crop.height,
            "steps": settings.steps,
            "strength": settings.strength,
            "sample_count": 1,
            "inpaint_area": constants.INPAINT_AREA_WHOLE_SELECTION,
            "mask_crop_padding": 0,
            "mask_blur": settings.mask_blur,
            PARAM_ENABLED: False,
            "auto_detailer_enabled": False,
            "gfpgan_face_restore_enabled": False,
        }
    )
    detail_context = GenerationContext(
        source=detail_crop,
        mask=detail_mask,
        parameters=detail_parameters,
        progress=context.progress,
        is_cancelled=context.is_cancelled,
        metadata=context.metadata,
    )
    detail_images = context.adapter.generate(detail_context)
    if not detail_images:
        return image, _region_report(region, "skipped_empty_output", crop_box)

    detail_image = detail_images[0].convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
    paste_mask = repair_mask.filter(ImageFilter.GaussianBlur(radius=settings.mask_blur))
    blended_crop = Image.composite(detail_image, crop, paste_mask)
    if _has_dark_halo(crop, blended_crop, paste_mask):
        return image, _region_report(region, "skipped_dark_halo", crop_box)

    output = image.copy()
    output.paste(blended_crop, crop_box[:2])
    return output, _region_report(
        region,
        "applied",
        crop_box,
        detail_size=detail_crop.size,
    )


def _detect_art_face_regions(
    image: Image.Image,
    generation_mask: Image.Image,
    min_area: int,
) -> list[ArtFaceRegion]:
    try:
        import torch
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "Art face repair requires PyTorch.",
            status_code=500,
        ) from exc

    processor, detector = _load_detector()
    detected: list[ArtFaceRegion] = []
    for query in DETECTION_QUERIES:
        inputs = processor(text=DETECTION_TASK + query, images=image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = detector.generate(**inputs, max_new_tokens=512, num_beams=3)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=DETECTION_TASK,
            image_size=image.size,
        )[DETECTION_TASK]
        for box in parsed.get("bboxes", []):
            region = ArtFaceRegion(tuple(int(round(value)) for value in box), query)
            if _is_plausible_region(region.box, image.size, min_area) and _region_intersects_mask(
                region.box,
                generation_mask,
            ):
                detected.append(region)
    return _dedupe_regions(detected)


def _load_detector() -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            (
                "Art face repair requires transformers with Florence-2 support. "
                "Install the API diffusers extra."
            ),
            status_code=500,
        ) from exc

    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
            detector = Florence2ForConditionalGeneration.from_pretrained(DETECTOR_MODEL_ID)
            detector.to("cpu")
            detector.eval()
            _DETECTOR = (processor, detector)
        return _DETECTOR


def _dedupe_regions(regions: list[ArtFaceRegion]) -> list[ArtFaceRegion]:
    output: list[ArtFaceRegion] = []
    for region in sorted(regions, key=lambda item: _area(item.box), reverse=True):
        if all(
            _iou(region.box, existing.box) < REGION_IOU_THRESHOLD
            and _containment(region.box, existing.box) < REGION_CONTAINMENT_THRESHOLD
            for existing in output
        ):
            output.append(region)
    return output


def _is_plausible_region(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    min_area: int,
) -> bool:
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    if min(width, height) < MIN_FACE_SIDE or width * height < min_area:
        return False
    if width > image_size[0] * 0.45 or height > image_size[1] * 0.60:
        return False
    aspect = width / height if height else 0.0
    return 0.35 <= aspect <= 1.85


def _region_intersects_mask(box: tuple[int, int, int, int], mask: Image.Image) -> bool:
    normalized = np.asarray(mask.convert("L")) >= constants.WHITE_MASK_THRESHOLD
    x0, y0, x1, y1 = _clamped_box(box, mask.size)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(normalized[y0:y1, x0:x1].mean() >= 0.15)


def _scaled_box(
    box: tuple[int, int, int, int],
    scale: float,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    half_width = max(16.0, ((x1 - x0) * scale) / 2)
    half_height = max(16.0, ((y1 - y0) * scale) / 2)
    return _clamped_box(
        (
            int(round(center_x - half_width)),
            int(round(center_y - half_height)),
            int(round(center_x + half_width)),
            int(round(center_y + half_height)),
        ),
        image_size,
    )


def _ellipse_mask(
    box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    crop_size: tuple[int, int],
) -> Image.Image:
    mask = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(mask)
    crop_x, crop_y = crop_box[:2]
    draw.ellipse(
        (
            box[0] - crop_x,
            box[1] - crop_y,
            box[2] - crop_x,
            box[3] - crop_y,
        ),
        fill=255,
    )
    return mask


def _detail_inputs(crop: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    max_side = max(crop.size)
    if max_side >= DETAIL_MIN_PROCESS_SIDE:
        return crop, mask
    scale = min(DETAIL_MAX_PROCESS_SIDE / max_side, DETAIL_MIN_PROCESS_SIDE / max_side)
    process_size = (
        max(64, int(round(crop.width * scale))),
        max(64, int(round(crop.height * scale))),
    )
    return (
        crop.resize(process_size, Image.Resampling.LANCZOS),
        mask.resize(process_size, Image.Resampling.NEAREST),
    )


def _repair_prompt(prompt: str) -> str:
    if prompt.strip():
        return f"{prompt.strip()}, {REPAIR_PROMPT_SUFFIX}"
    return REPAIR_PROMPT_SUFFIX


def _repair_negative_prompt(negative_prompt: str) -> str:
    if negative_prompt.strip():
        return f"{negative_prompt.strip()}, {REPAIR_NEGATIVE_SUFFIX}"
    return REPAIR_NEGATIVE_SUFFIX


def _has_dark_halo(original: Image.Image, candidate: Image.Image, mask: Image.Image) -> bool:
    mask_array = np.asarray(mask.convert("L")) > 16
    if int(mask_array.sum()) < 16:
        return False
    original_luma = _luma(original)[mask_array]
    candidate_luma = _luma(candidate)[mask_array]
    original_dark = float((original_luma < DARK_HALO_LUMA_THRESHOLD).mean())
    candidate_dark = float((candidate_luma < DARK_HALO_LUMA_THRESHOLD).mean())
    mean_drop = float(original_luma.mean() - candidate_luma.mean())
    return bool(
        mean_drop >= DARK_HALO_MEAN_DROP
        and candidate_dark >= original_dark + DARK_HALO_RATIO_INCREASE
    )


def _luma(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array[:, :, 0] * 0.2126 + array[:, :, 1] * 0.7152 + array[:, :, 2] * 0.0722


def _region_report(
    region: ArtFaceRegion,
    status: str,
    crop_box: tuple[int, int, int, int],
    detail_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "status": status,
        "label": region.label,
        "box": list(region.box),
        "crop_box": list(crop_box),
    }
    if detail_size is not None:
        report["detail_size"] = list(detail_size)
    return report


def _skip(
    message: str,
    context: GenerationPostprocessorContext,
    detected_regions: int,
    refined_regions: int,
) -> None:
    context.diagnostics.append(
        {
            "processor_id": ArtFaceRepairPostprocessor.id,
            "status": "skipped",
            "message": message,
            "detected_regions": detected_regions,
            "refined_regions": refined_regions,
        }
    )
    logger.warning(
        "Art face repair skipped: job_id=%s message=%s",
        context.metadata.get("job_id"),
        message,
    )


def _bool_parameter(context: GenerationPostprocessorContext, key: str, default: bool) -> bool:
    value = context.parameter(key, default)
    if isinstance(value, bool):
        return value
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"Art face repair parameter '{key}' must be a boolean.",
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
            f"Art face repair parameter '{key}' must be an integer.",
            status_code=422,
        )
    if not minimum <= value <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Art face repair parameter '{key}' must be between {minimum} and {maximum}.",
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
            f"Art face repair parameter '{key}' must be a number.",
            status_code=422,
        )
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"Art face repair parameter '{key}' must be between {minimum} and {maximum}.",
            status_code=422,
        )
    return parsed


def _clamped_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    return (
        max(0, min(box[0], image_size[0])),
        max(0, min(box[1], image_size[1])),
        max(0, min(box[2], image_size[0])),
        max(0, min(box[3], image_size[1])),
    )


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = _area((x0, y0, x1, y1))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0


def _containment(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = _area((x0, y0, x1, y1))
    smaller = min(_area(first), _area(second))
    return intersection / smaller if smaller else 0.0
