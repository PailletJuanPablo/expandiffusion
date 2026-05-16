"""SDXL Fill outpaint adapter with optional IP-Adapter visual refine."""

from __future__ import annotations

import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from expandiffusion import constants
from expandiffusion.adapters.base import GenerationContext
from expandiffusion.adapters.diffusers_inpaint import SdxlFillControlNetUnionAdapter
from expandiffusion.errors import AppError, GenerationCancelled
from expandiffusion.schemas import AdapterCapabilities, ControlOption, ControlSchema

ADAPTER_ID = "sdxl-fill-ip-refine"

PARAM_ENABLED = "visual_refine_enabled"
PARAM_STRENGTH = "visual_refine_strength"
PARAM_IP_SCALE = "ip_adapter_scale"
PARAM_STEPS = "visual_refine_steps"
PARAM_REFERENCE = "visual_refine_reference"

REFERENCE_NEAR_EDGE = "near_edge"
REFERENCE_VISIBLE_SOURCE = "visible_source"

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
IP_ADAPTER_IMAGE_ENCODER_FOLDER = "models/image_encoder"

DEFAULT_ENABLED = False
DEFAULT_STRENGTH = 0.45
DEFAULT_IP_SCALE = 0.45
DEFAULT_STEPS = 12
DEFAULT_REFERENCE = REFERENCE_NEAR_EDGE


@dataclass(slots=True)
class VisualRefineSettings:
    enabled: bool
    strength: float
    ip_adapter_scale: float
    steps: int
    reference: str


class SdxlFillIpRefineAdapter(SdxlFillControlNetUnionAdapter):
    """Replicate the HF Space outpaint flow and refine with IP-Adapter."""

    id = ADAPTER_ID
    label = "SDXL Fill + IP-Adapter Plus Refine"
    description = "HF Space SDXL outpaint with optional IP-Adapter Plus style harmonization."
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=False,
        controlnet=True,
        ip_adapter=True,
        textual_inversion=False,
        safety_checker=False,
        schedulers=[constants.SCHEDULER_AUTO],
        from_single_file=False,
    )

    def generation_controls(self) -> list[ControlSchema]:
        controls = super().generation_controls()
        controls.extend(
            [
                ControlSchema(
                    id=PARAM_ENABLED,
                    label="Visual refine",
                    kind=constants.CONTROL_SWITCH,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=DEFAULT_ENABLED,
                ),
                ControlSchema(
                    id=PARAM_STRENGTH,
                    label="Refine strength",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=DEFAULT_STRENGTH,
                    min=0.1,
                    max=0.8,
                    step=0.01,
                ),
                ControlSchema(
                    id=PARAM_IP_SCALE,
                    label="IP-Adapter Plus scale",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=DEFAULT_IP_SCALE,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                ControlSchema(
                    id=PARAM_STEPS,
                    label="Refine steps",
                    kind=constants.CONTROL_NUMBER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=DEFAULT_STEPS,
                    min=4,
                    max=40,
                    step=1,
                ),
                ControlSchema(
                    id=PARAM_REFERENCE,
                    label="Refine reference",
                    kind=constants.CONTROL_SELECT,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=DEFAULT_REFERENCE,
                    options=[
                        ControlOption(id=REFERENCE_NEAR_EDGE, label="near edge"),
                        ControlOption(id=REFERENCE_VISIBLE_SOURCE, label="visible source"),
                    ],
                ),
            ]
        )
        return controls

    def generation_defaults(self) -> dict[str, Any]:
        defaults = super().generation_defaults()
        defaults.update(
            {
                PARAM_ENABLED: DEFAULT_ENABLED,
                PARAM_STRENGTH: DEFAULT_STRENGTH,
                PARAM_IP_SCALE: DEFAULT_IP_SCALE,
                PARAM_STEPS: DEFAULT_STEPS,
                PARAM_REFERENCE: DEFAULT_REFERENCE,
            }
        )
        return defaults

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        settings = _settings_from_context(context)
        base_context = GenerationContext(
            source=context.source,
            mask=context.mask,
            parameters=context.parameters,
            progress=lambda value, message: context.progress(value * 0.75, message),
            is_cancelled=context.is_cancelled,
            conditioning_image=context.conditioning_image,
            metadata=context.metadata,
        )
        first_pass_images = super().generate(base_context)
        for index, image in enumerate(first_pass_images):
            _save_image(context.metadata, f"sample_{index:02d}_first_pass.png", image)

        if not settings.enabled:
            _save_json(
                context.metadata,
                "visual_refine_inputs.json",
                _visual_refine_report(settings, context, first_pass_images, enabled=False),
            )
            context.progress(constants.PROGRESS_FINISHED, "Generation complete")
            return first_pass_images

        return self._refine_images(context, first_pass_images, settings)

    def _refine_images(
        self,
        context: GenerationContext,
        first_pass_images: list[Image.Image],
        settings: VisualRefineSettings,
    ) -> list[Image.Image]:
        try:
            import torch
        except ImportError as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "PyTorch is required for IP-Adapter visual refine.",
                status_code=500,
            ) from exc

        reference = _reference_image(context.source, context.mask, settings.reference)
        _save_image(context.metadata, "sample_00_visual_refine_reference.png", reference)
        _save_json(
            context.metadata,
            "visual_refine_inputs.json",
            _visual_refine_report(settings, context, first_pass_images, enabled=True),
        )

        refine_pipeline = None
        try:
            context.progress(0.76, "Loading IP-Adapter refine")
            refine_pipeline = self._build_refine_pipeline()
            refine_pipeline.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT,
                image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER_FOLDER,
            )
            refine_pipeline.set_ip_adapter_scale(settings.ip_adapter_scale)

            refined_images: list[Image.Image] = []
            for index, first_pass in enumerate(first_pass_images):
                if context.is_cancelled():
                    raise GenerationCancelled()
                mask = _refine_mask(context.mask, first_pass.size)
                _save_image(context.metadata, f"sample_{index:02d}_visual_refine_mask.png", mask)
                _save_image(
                    context.metadata,
                    f"sample_{index:02d}_visual_refine_reference.png",
                    reference,
                )
                generator = torch.Generator(device=self.device)
                if not context.parameters.random_seed and context.parameters.seed is not None:
                    generator.manual_seed(context.parameters.seed + index)
                elif context.parameters.random_seed:
                    generator.manual_seed(random.randint(0, 2**31 - 1))
                total_steps = max(1, settings.steps)

                def callback_on_step_end(
                    _pipeline: Any,
                    step: int,
                    _timestep: Any,
                    callback_kwargs: dict[str, Any],
                    _total_steps: int = total_steps,
                ) -> dict[str, Any]:
                    if context.is_cancelled():
                        raise GenerationCancelled()
                    progress = 0.76 + ((step + 1) / _total_steps) * 0.22
                    context.progress(min(0.98, progress), "IP-Adapter visual refine")
                    return callback_kwargs

                output = refine_pipeline(
                    prompt=_refine_prompt(context.parameters.prompt),
                    image=first_pass.convert("RGB"),
                    mask_image=mask,
                    ip_adapter_image=reference,
                    width=first_pass.width,
                    height=first_pass.height,
                    strength=settings.strength,
                    num_inference_steps=settings.steps,
                    guidance_scale=context.parameters.guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                )
                if not output.images:
                    raise RuntimeError("IP-Adapter visual refine produced no image.")
                refined = output.images[0].convert("RGB")
                _save_image(context.metadata, f"sample_{index:02d}_visual_refined.png", refined)
                refined_images.append(refined)

            context.progress(constants.PROGRESS_FINISHED, "Generation complete")
            return refined_images
        except GenerationCancelled:
            raise
        except Exception as exc:
            message = "IP-Adapter visual refine failed after the base outpaint completed."
            if _is_oom(exc):
                message = (
                    "IP-Adapter visual refine ran out of CUDA memory after the base outpaint "
                    "completed."
                )
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                message,
                status_code=500,
                details={
                    "stage": "visual_refine",
                    "base_outpaint_completed": True,
                    "reason": str(exc),
                },
            ) from exc
        finally:
            if refine_pipeline is not None:
                _unload_ip_adapter(refine_pipeline)
                del refine_pipeline
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _build_refine_pipeline(self) -> Any:
        if self.pipeline is None:
            raise AppError(
                constants.ERROR_MODEL_NOT_LOADED,
                "Load the model before starting IP-Adapter visual refine.",
                status_code=409,
            )
        from diffusers import StableDiffusionXLInpaintPipeline

        scheduler = self.pipeline.scheduler.__class__.from_config(self.pipeline.scheduler.config)
        return StableDiffusionXLInpaintPipeline(
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            text_encoder_2=self.pipeline.text_encoder_2,
            tokenizer=self.pipeline.tokenizer,
            tokenizer_2=self.pipeline.tokenizer_2,
            unet=self.pipeline.unet,
            scheduler=scheduler,
            add_watermarker=False,
        ).to(self.device)


def _settings_from_context(context: GenerationContext) -> VisualRefineSettings:
    return VisualRefineSettings(
        enabled=_bool_parameter(context, PARAM_ENABLED, DEFAULT_ENABLED),
        strength=_float_parameter(context, PARAM_STRENGTH, DEFAULT_STRENGTH, 0.1, 0.8),
        ip_adapter_scale=_float_parameter(context, PARAM_IP_SCALE, DEFAULT_IP_SCALE, 0.0, 1.0),
        steps=_int_parameter(context, PARAM_STEPS, DEFAULT_STEPS, 4, 40),
        reference=_choice_parameter(
            context,
            PARAM_REFERENCE,
            DEFAULT_REFERENCE,
            {REFERENCE_NEAR_EDGE, REFERENCE_VISIBLE_SOURCE},
        ),
    )


def _parameter(context: GenerationContext, key: str, default: Any) -> Any:
    extra = context.parameters.model_extra or {}
    if key in extra:
        return extra[key]
    return getattr(context.parameters, key, default)


def _bool_parameter(context: GenerationContext, key: str, default: bool) -> bool:
    value = _parameter(context, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float_parameter(
    context: GenerationContext,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(_parameter(context, key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _int_parameter(
    context: GenerationContext,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(_parameter(context, key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _choice_parameter(
    context: GenerationContext,
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    value = str(_parameter(context, key, default))
    return value if value in allowed else default


def _reference_image(source: Image.Image, mask: Image.Image, reference: str) -> Image.Image:
    source_rgb = source.convert("RGB")
    mask_l = _refine_mask(mask, source_rgb.size)
    known_mask = mask_l.point(
        lambda value: 255 if value < constants.WHITE_MASK_THRESHOLD else 0
    )
    if reference == REFERENCE_VISIBLE_SOURCE:
        return _crop_by_mask(source_rgb, known_mask)

    edge_radius = max(2, min(128, min(source_rgb.size) // 8))
    dilated_generated = mask_l.filter(ImageFilter.MaxFilter(edge_radius * 2 + 1))
    edge_known = ImageChops.multiply(dilated_generated, known_mask)
    return _crop_by_mask(source_rgb, edge_known, fallback_mask=known_mask)


def _crop_by_mask(
    source: Image.Image,
    mask: Image.Image,
    fallback_mask: Image.Image | None = None,
) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None and fallback_mask is not None:
        bbox = fallback_mask.getbbox()
    if bbox is None:
        return source.copy()
    crop = source.crop(bbox)
    if crop.width < 8 or crop.height < 8:
        if fallback_mask is not None and fallback_mask.getbbox() is not None:
            return source.crop(fallback_mask.getbbox())
        return source.copy()
    return crop


def _refine_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.convert("L").resize(size, Image.Resampling.NEAREST).point(
        lambda value: 255 if value >= constants.WHITE_MASK_THRESHOLD else 0
    )


def _refine_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "continue the image, same style, same lighting, same color palette"
    return f"{prompt}, same style, same lighting, same color palette"


def _visual_refine_report(
    settings: VisualRefineSettings,
    context: GenerationContext,
    first_pass_images: list[Image.Image],
    *,
    enabled: bool,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "settings": {
            PARAM_ENABLED: settings.enabled,
            PARAM_STRENGTH: settings.strength,
            PARAM_IP_SCALE: settings.ip_adapter_scale,
            PARAM_STEPS: settings.steps,
            PARAM_REFERENCE: settings.reference,
        },
        "ip_adapter": {
            "repo": IP_ADAPTER_REPO,
            "subfolder": IP_ADAPTER_SUBFOLDER,
            "weight_name": IP_ADAPTER_WEIGHT,
            "image_encoder_folder": IP_ADAPTER_IMAGE_ENCODER_FOLDER,
        },
        "source_size": list(context.source.size),
        "mask_size": list(context.mask.size),
        "first_pass_count": len(first_pass_images),
        "first_pass_sizes": [list(image.size) for image in first_pass_images],
    }


def _adapter_artifact_dir(metadata: dict[str, Any]) -> Path | None:
    artifact_dir = metadata.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_image(metadata: dict[str, Any], filename: str, image: Image.Image) -> None:
    artifact_dir = _adapter_artifact_dir(metadata)
    if artifact_dir is not None:
        image.save(artifact_dir / filename)


def _save_json(metadata: dict[str, Any], filename: str, payload: dict[str, Any]) -> None:
    artifact_dir = _adapter_artifact_dir(metadata)
    if artifact_dir is not None:
        (artifact_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def _unload_ip_adapter(pipeline: Any) -> None:
    unload = getattr(pipeline, "unload_ip_adapter", None)
    if callable(unload):
        unload()


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def register(context: Any) -> None:
    context.register_model_adapter(SdxlFillIpRefineAdapter())
