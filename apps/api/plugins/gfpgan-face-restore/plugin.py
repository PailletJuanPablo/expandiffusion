"""GFPGAN face restoration generation postprocessor."""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from expandiffusion import constants
from expandiffusion.errors import AppError, GenerationCancelled
from expandiffusion.postprocessors import (
    GenerationPostprocessor,
    GenerationPostprocessorContext,
)
from expandiffusion.schemas import ControlOption, ControlSchema

logger = logging.getLogger("uvicorn.error")

PARAM_ENABLED = "gfpgan_face_restore_enabled"
PARAM_MODEL = "gfpgan_face_restore_model"
PARAM_MODEL_PATH = "gfpgan_face_restore_model_path"
PARAM_WEIGHT = "gfpgan_face_restore_weight"
PARAM_ONLY_CENTER_FACE = "gfpgan_face_restore_only_center_face"
PARAM_GENERATED_AREA_ONLY = "gfpgan_face_restore_generated_area_only"
PARAM_MASK_BLUR = "gfpgan_face_restore_mask_blur"

MODEL_GFPGAN_1_4 = "gfpgan_1_4"
MODEL_GFPGAN_1_3 = "gfpgan_1_3"
MODEL_RESTOREFORMER = "restoreformer"

DEFAULT_ENABLED = False
DEFAULT_MODEL = MODEL_GFPGAN_1_4
DEFAULT_MODEL_PATH = ""
DEFAULT_WEIGHT = 0.5
DEFAULT_ONLY_CENTER_FACE = False
DEFAULT_GENERATED_AREA_ONLY = True
DEFAULT_MASK_BLUR = 8
MIN_CHANGED_PIXELS = 16


@dataclass(frozen=True, slots=True)
class GFPGANModelSpec:
    """Model metadata required by GFPGANer."""

    label: str
    model_path: str
    arch: str
    channel_multiplier: int


@dataclass(frozen=True, slots=True)
class GFPGANSettings:
    """Validated plugin-owned runtime settings."""

    enabled: bool
    model: str
    model_path: str
    weight: float
    only_center_face: bool
    generated_area_only: bool
    mask_blur: int


MODEL_SPECS = {
    MODEL_GFPGAN_1_4: GFPGANModelSpec(
        label="GFPGAN v1.4",
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        arch="clean",
        channel_multiplier=2,
    ),
    MODEL_GFPGAN_1_3: GFPGANModelSpec(
        label="GFPGAN v1.3",
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        arch="clean",
        channel_multiplier=2,
    ),
    MODEL_RESTOREFORMER: GFPGANModelSpec(
        label="RestoreFormer",
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        arch="RestoreFormer",
        channel_multiplier=2,
    ),
}


class GFPGANFaceRestorePostprocessor(GenerationPostprocessor):
    """Run a dedicated blind face restoration pass with GFPGAN."""

    id = "gfpgan-face-restore"
    label = "GFPGAN Face Restore"
    description = "Dedicated face restoration for generated faces."

    def __init__(self) -> None:
        self._restorers: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_ENABLED,
                label="Restore faces",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_ENABLED,
            ),
            ControlSchema(
                id=PARAM_MODEL,
                label="Restorer",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MODEL,
                options=[
                    ControlOption(id=MODEL_GFPGAN_1_4, label="GFPGAN v1.4"),
                    ControlOption(id=MODEL_GFPGAN_1_3, label="GFPGAN v1.3"),
                    ControlOption(id=MODEL_RESTOREFORMER, label="RestoreFormer"),
                ],
            ),
            ControlSchema(
                id=PARAM_MODEL_PATH,
                label="Model path",
                kind=constants.CONTROL_TEXT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_MODEL_PATH,
                placeholder="Optional local .pth path or https URL",
            ),
            ControlSchema(
                id=PARAM_WEIGHT,
                label="Fidelity weight",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_WEIGHT,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_ONLY_CENTER_FACE,
                label="Only center face",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_ONLY_CENTER_FACE,
            ),
            ControlSchema(
                id=PARAM_GENERATED_AREA_ONLY,
                label="Generated area only",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_GENERATED_AREA_ONLY,
            ),
            ControlSchema(
                id=PARAM_MASK_BLUR,
                label="Restore mask blur",
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
            PARAM_MODEL: DEFAULT_MODEL,
            PARAM_MODEL_PATH: DEFAULT_MODEL_PATH,
            PARAM_WEIGHT: DEFAULT_WEIGHT,
            PARAM_ONLY_CENTER_FACE: DEFAULT_ONLY_CENTER_FACE,
            PARAM_GENERATED_AREA_ONLY: DEFAULT_GENERATED_AREA_ONLY,
            PARAM_MASK_BLUR: DEFAULT_MASK_BLUR,
        }

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        settings = _settings_from_context(context)
        if not settings.enabled:
            return context.generated

        if context.is_cancelled():
            raise GenerationCancelled()

        generated = context.generated.convert("RGB")
        logger.info(
            "GFPGAN face restore started: job_id=%s model=%s weight=%s generated_area_only=%s",
            context.metadata.get("job_id"),
            settings.model,
            settings.weight,
            settings.generated_area_only,
        )

        restorer = self._restorer(settings, context)
        input_bgr = np.asarray(generated)[:, :, ::-1]
        output_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_log), contextlib.redirect_stderr(output_log):
                cropped_faces, restored_faces, restored_bgr = restorer.enhance(
                    input_bgr,
                    has_aligned=False,
                    only_center_face=settings.only_center_face,
                    paste_back=True,
                    weight=settings.weight,
                )
        except Exception as exc:
            _fail(f"GFPGAN face restore inference failed: {exc}", context, 0, 0)

        captured_output = output_log.getvalue().strip()
        if "Failed inference for GFPGAN" in captured_output:
            _fail(
                f"GFPGAN reported an internal inference failure: {captured_output}",
                context,
                0,
                0,
            )
        if restored_bgr is None:
            _fail("GFPGAN did not return a restored image.", context, len(cropped_faces), 0)
        if not restored_faces:
            _skip("GFPGAN did not detect any face to restore.", context, 0, 0)
            return generated

        restored = Image.fromarray(restored_bgr[:, :, ::-1]).convert("RGB").resize(generated.size)
        output = _blend_restored_image(generated, restored, context.mask, settings)
        changed_pixels = _changed_pixel_count(generated, output)
        if changed_pixels < MIN_CHANGED_PIXELS:
            """ _fail(
                (
                    "GFPGAN restored faces, but no restored pixels were applied. "
                    "If the face is outside the generated mask, disable 'Generated area only'."
                ),
                context,
                len(cropped_faces),
                len(restored_faces),
            ) """

        report = {
            "processor_id": self.id,
            "status": "applied",
            "detected_faces": len(cropped_faces),
            "restored_faces": len(restored_faces),
            "changed_pixels": changed_pixels,
            "model": settings.model,
        }
        context.diagnostics.append(report)
        logger.info(
            (
                "GFPGAN face restore applied: job_id=%s detected_faces=%s "
                "restored_faces=%s changed_pixels=%s"
            ),
            context.metadata.get("job_id"),
            len(cropped_faces),
            len(restored_faces),
            changed_pixels,
        )
        return output

    def _restorer(self, settings: GFPGANSettings, context: GenerationPostprocessorContext) -> Any:
        spec = MODEL_SPECS[settings.model]
        model_path = settings.model_path or spec.model_path
        _validate_model_path(model_path)
        device = _resolve_device(context)
        cache_key = (settings.model, model_path, device)
        with self._lock:
            if cache_key in self._restorers:
                return self._restorers[cache_key]
            GFPGANer = _load_gfpganer()
            logger.info(
                "GFPGAN face restore loading model: model=%s device=%s path=%s",
                settings.model,
                device,
                model_path,
            )
            try:
                restorer = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch=spec.arch,
                    channel_multiplier=spec.channel_multiplier,
                    bg_upsampler=None,
                    device=device,
                )
            except Exception as exc:
                _fail(f"GFPGAN model could not be loaded: {exc}", context, 0, 0)
            self._restorers[cache_key] = restorer
            return restorer


def register(context) -> None:
    context.register_generation_postprocessor(GFPGANFaceRestorePostprocessor())


def _settings_from_context(context: GenerationPostprocessorContext) -> GFPGANSettings:
    model = _string_parameter(context, PARAM_MODEL, DEFAULT_MODEL)
    if model not in MODEL_SPECS:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"GFPGAN face restore parameter '{PARAM_MODEL}' must be one of {sorted(MODEL_SPECS)}.",
            status_code=422,
        )
    return GFPGANSettings(
        enabled=_bool_parameter(context, PARAM_ENABLED, DEFAULT_ENABLED),
        model=model,
        model_path=_string_parameter(context, PARAM_MODEL_PATH, DEFAULT_MODEL_PATH).strip(),
        weight=_float_parameter(context, PARAM_WEIGHT, DEFAULT_WEIGHT, 0.0, 1.0),
        only_center_face=_bool_parameter(
            context,
            PARAM_ONLY_CENTER_FACE,
            DEFAULT_ONLY_CENTER_FACE,
        ),
        generated_area_only=_bool_parameter(
            context,
            PARAM_GENERATED_AREA_ONLY,
            DEFAULT_GENERATED_AREA_ONLY,
        ),
        mask_blur=_int_parameter(context, PARAM_MASK_BLUR, DEFAULT_MASK_BLUR, 0, 64),
    )


def _bool_parameter(context: GenerationPostprocessorContext, key: str, default: bool) -> bool:
    value = context.parameter(key, default)
    if isinstance(value, bool):
        return value
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"GFPGAN face restore parameter '{key}' must be a boolean.",
        status_code=422,
    )


def _string_parameter(context: GenerationPostprocessorContext, key: str, default: str) -> str:
    value = context.parameter(key, default)
    if isinstance(value, str):
        return value
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"GFPGAN face restore parameter '{key}' must be a string.",
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
            f"GFPGAN face restore parameter '{key}' must be an integer.",
            status_code=422,
        )
    if not minimum <= value <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"GFPGAN face restore parameter '{key}' must be between {minimum} and {maximum}.",
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
            f"GFPGAN face restore parameter '{key}' must be a number.",
            status_code=422,
        )
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise AppError(
            constants.ERROR_GENERATION_FAILED,
            f"GFPGAN face restore parameter '{key}' must be between {minimum} and {maximum}.",
            status_code=422,
        )
    return parsed


def _blend_restored_image(
    generated: Image.Image,
    restored: Image.Image,
    generation_mask: Image.Image,
    settings: GFPGANSettings,
) -> Image.Image:
    if not settings.generated_area_only:
        return restored
    mask = generation_mask.convert("L").resize(generated.size, Image.Resampling.LANCZOS)
    if settings.mask_blur:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=settings.mask_blur))
    return Image.composite(restored, generated, mask)


def _changed_pixel_count(original: Image.Image, output: Image.Image) -> int:
    difference = ImageChops.difference(original.convert("RGB"), output.convert("RGB"))
    return int((np.asarray(difference) > 2).any(axis=2).sum())


def _resolve_device(context: GenerationPostprocessorContext) -> str:
    try:
        import torch
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "GFPGAN face restore requires PyTorch, but torch is not installed.",
            status_code=500,
        ) from exc

    adapter_device = getattr(context.adapter, "device", None)
    if (
        isinstance(adapter_device, str)
        and adapter_device
        and adapter_device != constants.DEFAULT_DEVICE
    ):
        if adapter_device.startswith("cuda") and not torch.cuda.is_available():
            raise AppError(
                constants.ERROR_UNSUPPORTED_OPERATION,
                f"GFPGAN face restore requested '{adapter_device}', but CUDA is not available.",
                status_code=500,
            )
        return adapter_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_gfpganer() -> Any:
    _install_torchvision_compatibility_shim()
    try:
        from gfpgan import GFPGANer
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            (
                "GFPGAN face restore requires optional dependency 'gfpgan'. "
                "Install the API extra: python -m pip install -e apps/api[face-restoration]"
            ),
            status_code=500,
        ) from exc
    return GFPGANer


def _install_torchvision_compatibility_shim() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        from torchvision.transforms.functional import rgb_to_grayscale
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "GFPGAN face restore requires torchvision with rgb_to_grayscale support.",
            status_code=500,
        ) from exc
    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module


def _validate_model_path(model_path: str) -> None:
    if model_path.startswith("https://"):
        return
    if Path(model_path).is_file():
        return
    raise AppError(
        constants.ERROR_GENERATION_FAILED,
        f"GFPGAN model path does not exist: {model_path}",
        status_code=422,
    )


def _fail(
    message: str,
    context: GenerationPostprocessorContext,
    detected_faces: int,
    restored_faces: int,
) -> None:
    report = {
        "processor_id": GFPGANFaceRestorePostprocessor.id,
        "status": "failed",
        "message": message,
        "detected_faces": detected_faces,
        "restored_faces": restored_faces,
    }
    context.diagnostics.append(report)
    logger.error(
        "GFPGAN face restore failed: job_id=%s message=%s",
        context.metadata.get("job_id"),
        message,
    )
    raise AppError(constants.ERROR_GENERATION_FAILED, message, status_code=500)


def _skip(
    message: str,
    context: GenerationPostprocessorContext,
    detected_faces: int,
    restored_faces: int,
) -> None:
    context.diagnostics.append(
        {
            "processor_id": GFPGANFaceRestorePostprocessor.id,
            "status": "skipped",
            "message": message,
            "detected_faces": detected_faces,
            "restored_faces": restored_faces,
        }
    )
    logger.warning(
        "GFPGAN face restore skipped: job_id=%s message=%s",
        context.metadata.get("job_id"),
        message,
    )
