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
PARAM_INPAINT_STRENGTH = "inpaint_strength"

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
DEFAULT_INPAINT_STRENGTH = 0.65


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
                    id=PARAM_INPAINT_STRENGTH,
                    label="Inpaint strength",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_BASIC,
                    default_value=DEFAULT_INPAINT_STRENGTH,
                    min=0.05,
                    max=1.0,
                    step=0.01,
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
                PARAM_INPAINT_STRENGTH: DEFAULT_INPAINT_STRENGTH,
            }
        )
        return defaults

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        if context.metadata.get("generation_mode") == constants.GENERATION_MODE_INPAINT:
            return self._inpaint_images(context)

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

    def _inpaint_images(self, context: GenerationContext) -> list[Image.Image]:
        try:
            import torch
        except ImportError as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "PyTorch is required for SDXL inpaint.",
                status_code=500,
            ) from exc

        pipeline = None
        try:
            source = context.source.convert("RGB")
            mask = _refine_mask(context.mask, source.size)
            process_size = _multiple_of_eight_size(source.size)
            process_source = _pad_to_size(source, process_size, edge=True)
            process_mask = _pad_to_size(mask, process_size, edge=False)
            strength = _inpaint_strength(context)
            _save_image(context.metadata, "conservative_inpaint_source.png", source)
            _save_image(context.metadata, "conservative_inpaint_mask.png", mask)
            _save_json(
                context.metadata,
                "conservative_inpaint_inputs.json",
                {
                    "source_size": list(source.size),
                    "process_size": list(process_size),
                    "strength": strength,
                    "steps": context.parameters.steps,
                    "guidance_scale": context.parameters.guidance_scale,
                    "sample_count": context.parameters.sample_count,
                    "seed": context.parameters.seed,
                    "random_seed": context.parameters.random_seed,
                },
            )

            context.progress(0.01, "Loading SDXL inpaint")
            pipeline = self._build_refine_pipeline()
            images: list[Image.Image] = []
            sample_count = max(1, context.parameters.sample_count)
            total_steps = max(1, context.parameters.steps)
            for index in range(sample_count):
                if context.is_cancelled():
                    raise GenerationCancelled()
                generator = torch.Generator(device=self.device)
                if not context.parameters.random_seed and context.parameters.seed is not None:
                    generator.manual_seed(context.parameters.seed + index)
                elif context.parameters.random_seed:
                    generator.manual_seed(random.randint(0, 2**31 - 1))

                def callback_on_step_end(
                    _pipeline: Any,
                    step: int,
                    _timestep: Any,
                    callback_kwargs: dict[str, Any],
                    _index: int = index,
                    _sample_count: int = sample_count,
                    _total_steps: int = total_steps,
                ) -> dict[str, Any]:
                    if context.is_cancelled():
                        raise GenerationCancelled()
                    completed = _index * _total_steps + step + 1
                    total = max(1, _sample_count * _total_steps)
                    context.progress(min(0.98, completed / total), "SDXL inpaint")
                    return callback_kwargs

                output = pipeline(
                    prompt=_refine_prompt(context.parameters.prompt),
                    image=process_source,
                    mask_image=process_mask,
                    width=process_size[0],
                    height=process_size[1],
                    strength=strength,
                    num_inference_steps=context.parameters.steps,
                    guidance_scale=context.parameters.guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                )
                if not output.images:
                    raise RuntimeError("SDXL inpaint produced no image.")
                image = output.images[0].convert("RGB").crop((0, 0, *source.size))
                _save_image(context.metadata, f"sample_{index:02d}_conservative_inpaint.png", image)
                images.append(image)
            context.progress(constants.PROGRESS_FINISHED, "Generation complete")
            return images
        except GenerationCancelled:
            raise
        except Exception as exc:
            message = "SDXL inpaint failed."
            if _is_oom(exc):
                message = "SDXL inpaint ran out of CUDA memory."
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                message,
                status_code=500,
                details={"stage": "conservative_inpaint", "reason": str(exc)},
            ) from exc
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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


def _inpaint_strength(context: GenerationContext) -> float:
    return _float_parameter(context, PARAM_INPAINT_STRENGTH, DEFAULT_INPAINT_STRENGTH, 0.05, 1.0)


def _multiple_of_eight_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return (_ceil_to_multiple(width, 8), _ceil_to_multiple(height, 8))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_size(image: Image.Image, size: tuple[int, int], *, edge: bool) -> Image.Image:
    if image.size == size:
        return image
    width, height = image.size
    target_width, target_height = size
    fill = 0 if image.mode == "L" else None
    output = Image.new(image.mode, size, fill)
    output.paste(image, (0, 0))
    if edge and width > 0 and height > 0:
        if target_width > width:
            right_edge = image.crop((width - 1, 0, width, height)).resize(
                (target_width - width, height)
            )
            output.paste(right_edge, (width, 0))
        if target_height > height:
            bottom_edge = output.crop((0, height - 1, target_width, height)).resize(
                (target_width, target_height - height)
            )
            output.paste(bottom_edge, (0, height))
    return output


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
