"""Post-compose seam refine using SDXL inpaint and IP-Adapter Plus."""

from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from expandiffusion import constants
from expandiffusion.errors import AppError, GenerationCancelled
from expandiffusion.postprocessors import GenerationPostprocessor, GenerationPostprocessorContext
from expandiffusion.schemas import ControlOption, ControlSchema

PROCESSOR_ID = "sdxl-seam-refine"

PARAM_ENABLED = "seam_refine_enabled"
PARAM_WIDTH = "seam_refine_width"
PARAM_STRENGTH = "seam_refine_strength"
PARAM_IP_SCALE = "seam_refine_ip_adapter_scale"
PARAM_STEPS = "seam_refine_steps"
PARAM_REFERENCE = "seam_refine_reference"

REFERENCE_NEAR_EDGE = "near_edge"
REFERENCE_VISIBLE_SOURCE = "visible_source"

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
IP_ADAPTER_IMAGE_ENCODER_FOLDER = "models/image_encoder"

DEFAULT_ENABLED = False
DEFAULT_WIDTH = 24
DEFAULT_STRENGTH = 0.18
DEFAULT_IP_SCALE = 0.35
DEFAULT_STEPS = 6
DEFAULT_REFERENCE = REFERENCE_NEAR_EDGE

DEFAULT_PROMPT = (
    "same painting, seamless transition, matching color, matching lighting, "
    "matching brushwork, preserve composition"
)
NEGATIVE_PROMPT = (
    "visible seam, hard edge, border, frame, duplicate image, pasted image, "
    "split image, blur, text, watermark"
)


@dataclass(slots=True)
class SeamRefineSettings:
    enabled: bool
    width: int
    strength: float
    ip_adapter_scale: float
    steps: int
    reference: str


class SdxlSeamRefinePostprocessor(GenerationPostprocessor):
    """Refine only the visible join after the final outpaint composition."""

    id = PROCESSOR_ID
    label = "SDXL Seam Refine"
    description = "Model-based seam repair after final composition using SDXL inpaint and IP-Adapter Plus."
    category = constants.POSTPROCESSOR_CATEGORY_RESULT_REFINE
    default_order = 10

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_ENABLED,
                label="Seam refine",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_ENABLED,
            ),
            ControlSchema(
                id=PARAM_WIDTH,
                label="Seam width",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_WIDTH,
                min=8,
                max=128,
                step=1,
            ),
            ControlSchema(
                id=PARAM_STRENGTH,
                label="Seam strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STRENGTH,
                min=0.05,
                max=0.6,
                step=0.01,
            ),
            ControlSchema(
                id=PARAM_IP_SCALE,
                label="Seam IP-Adapter scale",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_IP_SCALE,
                min=0.0,
                max=1.0,
                step=0.01,
            ),
            ControlSchema(
                id=PARAM_STEPS,
                label="Seam steps",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_STEPS,
                min=2,
                max=24,
                step=1,
            ),
            ControlSchema(
                id=PARAM_REFERENCE,
                label="Seam reference",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=DEFAULT_REFERENCE,
                options=[
                    ControlOption(id=REFERENCE_NEAR_EDGE, label="near edge"),
                    ControlOption(id=REFERENCE_VISIBLE_SOURCE, label="visible source"),
                ],
            ),
        ]

    def generation_defaults(self) -> dict[str, Any]:
        return {
            PARAM_ENABLED: DEFAULT_ENABLED,
            PARAM_WIDTH: DEFAULT_WIDTH,
            PARAM_STRENGTH: DEFAULT_STRENGTH,
            PARAM_IP_SCALE: DEFAULT_IP_SCALE,
            PARAM_STEPS: DEFAULT_STEPS,
            PARAM_REFERENCE: DEFAULT_REFERENCE,
        }

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        settings = _settings_from_context(context)
        composed = context.generated.convert("RGB")
        if not settings.enabled:
            return composed

        seam_mask = _build_seam_mask(context.original, context.mask, composed.size, settings.width)
        if seam_mask.getbbox() is None:
            return composed
        original_size = composed.size
        process_size = _sdxl_process_size(original_size)
        process_image = _pad_to_size(composed, process_size, edge=True)
        process_mask = _pad_to_size(seam_mask, process_size, edge=False)

        reference = _reference_image(
            context.original,
            context.mask,
            original_size,
            settings.width,
            settings.reference,
        )
        _save_image(context.metadata, "seam_refine_input.png", composed)
        _save_image(context.metadata, "seam_refine_mask.png", seam_mask)
        _save_image(context.metadata, "seam_refine_reference.png", reference)
        _save_json(
            context.metadata,
            "seam_refine_inputs.json",
            _seam_refine_report(settings, context, seam_mask, reference),
        )

        pipeline = None
        try:
            import torch

            context.progress(0.98, "Loading SDXL seam refine")
            pipeline = _build_pipeline(torch, context.adapter)
            pipeline.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT,
                image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER_FOLDER,
            )
            pipeline.set_ip_adapter_scale(settings.ip_adapter_scale)
            _prepare_pipeline_runtime(pipeline, torch, context.adapter)

            generator = _torch_generator(torch, context, _resolve_device(context.adapter))
            inference_steps = _effective_inference_steps(settings.steps, settings.strength)
            total_steps = max(1, inference_steps)

            def callback_on_step_end(
                _pipeline: Any,
                step: int,
                _timestep: Any,
                callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                if context.is_cancelled():
                    raise GenerationCancelled()
                context.progress(
                    0.98 + min(0.015, ((step + 1) / total_steps) * 0.015),
                    "SDXL seam refine",
                )
                return callback_kwargs

            output = pipeline(
                prompt=_prompt(context),
                negative_prompt=NEGATIVE_PROMPT,
                image=process_image,
                mask_image=process_mask,
                ip_adapter_image=reference,
                width=process_size[0],
                height=process_size[1],
                strength=settings.strength,
                num_inference_steps=inference_steps,
                guidance_scale=context.parameters.guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            )
            if not output.images:
                raise RuntimeError("SDXL seam refine produced no image.")
            refined = output.images[0].convert("RGB").crop((0, 0, *original_size))
            localized = Image.composite(refined, composed, seam_mask)
            _save_image(context.metadata, "seam_refined_raw.png", refined)
            _save_image(context.metadata, "seam_refined.png", localized)
            return localized
        except GenerationCancelled:
            raise
        except Exception as exc:
            message = "SDXL seam refine failed after the base generation completed."
            if _is_oom(exc):
                message = "SDXL seam refine ran out of CUDA memory after the base generation completed."
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                message,
                status_code=500,
                details={
                    "stage": PROCESSOR_ID,
                    "base_generation_completed": True,
                    "reason": str(exc),
                },
            ) from exc
        finally:
            if pipeline is not None:
                _unload_ip_adapter(pipeline)
                del pipeline
            gc.collect()
            _empty_cuda_cache()


def _settings_from_context(context: GenerationPostprocessorContext) -> SeamRefineSettings:
    return SeamRefineSettings(
        enabled=_bool_parameter(context, PARAM_ENABLED, DEFAULT_ENABLED),
        width=_int_parameter(context, PARAM_WIDTH, DEFAULT_WIDTH, 8, 128),
        strength=_float_parameter(context, PARAM_STRENGTH, DEFAULT_STRENGTH, 0.05, 0.6),
        ip_adapter_scale=_float_parameter(context, PARAM_IP_SCALE, DEFAULT_IP_SCALE, 0.0, 1.0),
        steps=_int_parameter(context, PARAM_STEPS, DEFAULT_STEPS, 2, 24),
        reference=_choice_parameter(
            context,
            PARAM_REFERENCE,
            DEFAULT_REFERENCE,
            {REFERENCE_NEAR_EDGE, REFERENCE_VISIBLE_SOURCE},
        ),
    )


def _sdxl_process_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return (_ceil_to_multiple(width, 8), _ceil_to_multiple(height, 8))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _effective_inference_steps(steps: int, strength: float) -> int:
    if strength <= 0:
        return steps
    return max(steps, 1 if strength >= 1 else math.ceil(1 / strength))


def _pad_to_size(image: Image.Image, size: tuple[int, int], *, edge: bool) -> Image.Image:
    if image.size == size:
        return image
    width, height = image.size
    target_width, target_height = size
    fill = 0 if image.mode == "L" else None
    padded = Image.new(image.mode, size, fill)
    padded.paste(image, (0, 0))
    if edge and width > 0 and height > 0:
        if target_width > width:
            right_edge = image.crop((width - 1, 0, width, height)).resize(
                (target_width - width, height)
            )
            padded.paste(right_edge, (width, 0))
        if target_height > height:
            bottom_edge = padded.crop((0, height - 1, target_width, height)).resize(
                (target_width, target_height - height)
            )
            padded.paste(bottom_edge, (0, height))
    return padded


def _build_seam_mask(
    original: Image.Image,
    generation_mask: Image.Image,
    size: tuple[int, int],
    width: int,
) -> Image.Image:
    mask = _binary_mask(generation_mask, size)
    visible_known = _visible_known_mask(original, mask, size)
    if visible_known.getbbox() is None:
        return Image.new("L", size, 0)

    generated_near_known = ImageChops.multiply(mask, _expand_mask(visible_known, width))
    known_near_generated = ImageChops.multiply(
        visible_known,
        _expand_mask(mask, max(4, width // 2)),
    )
    seam = ImageChops.lighter(generated_near_known, known_near_generated)
    blur_radius = max(1.0, width / 6)
    return seam.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _reference_image(
    original: Image.Image,
    generation_mask: Image.Image,
    size: tuple[int, int],
    width: int,
    reference: str,
) -> Image.Image:
    original_rgb = original.convert("RGB").resize(size)
    mask = _binary_mask(generation_mask, size)
    visible_known = _visible_known_mask(original, mask, size)
    if reference == REFERENCE_VISIBLE_SOURCE:
        return _crop_by_mask(original_rgb, visible_known)

    near_generated = ImageChops.multiply(visible_known, _expand_mask(mask, max(8, width * 2)))
    return _crop_by_mask(original_rgb, near_generated, fallback_mask=visible_known)


def _visible_known_mask(
    original: Image.Image,
    generation_mask: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    alpha = original.convert("RGBA").resize(size).getchannel("A").point(
        lambda value: 255 if value >= constants.WHITE_MASK_THRESHOLD else 0
    )
    known = generation_mask.point(
        lambda value: 0 if value >= constants.WHITE_MASK_THRESHOLD else 255
    )
    visible_known = ImageChops.multiply(alpha, known)
    return visible_known if visible_known.getbbox() is not None else known


def _binary_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.convert("L").resize(size, Image.Resampling.NEAREST).point(
        lambda value: 255 if value >= constants.WHITE_MASK_THRESHOLD else 0
    )


def _expand_mask(mask: Image.Image, radius: int) -> Image.Image:
    kernel = max(3, radius * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    return mask.filter(ImageFilter.MaxFilter(kernel))


def _crop_by_mask(
    image: Image.Image,
    mask: Image.Image,
    fallback_mask: Image.Image | None = None,
) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None and fallback_mask is not None:
        bbox = fallback_mask.getbbox()
    if bbox is None:
        return image.copy()
    crop = image.crop(_pad_bbox(bbox, image.size, 16))
    if crop.width < 8 or crop.height < 8:
        return image.copy()
    return crop


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _build_pipeline(torch: Any, adapter: Any) -> Any:
    from diffusers import StableDiffusionXLInpaintPipeline

    source = _local_sdxl_inpaint_snapshot() or constants.MODEL_SDXL_INPAINT
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        source,
        torch_dtype=dtype,
        add_watermarker=False,
    )
    _disable_safety_checker(pipeline)
    return pipeline


def _prepare_pipeline_runtime(pipeline: Any, torch: Any, adapter: Any) -> None:
    try:
        pipeline.vae.enable_slicing()
    except Exception:
        pass
    if torch.cuda.is_available():
        pipeline.enable_model_cpu_offload(gpu_id=_gpu_id_from_device(_resolve_device(adapter)))
    else:
        pipeline.to("cpu")


def _local_sdxl_inpaint_snapshot() -> str | None:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1"
        / "snapshots"
    )
    if not cache_root.exists():
        return None
    snapshots = sorted(
        (path for path in cache_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(snapshots[0]) if snapshots else None


def _disable_safety_checker(pipeline: Any) -> None:
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "requires_safety_checker"):
        pipeline.requires_safety_checker = False


def _torch_generator(
    torch: Any,
    context: GenerationPostprocessorContext,
    device: str,
) -> Any:
    generator_device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device)
    sample_index = _sample_index(context.metadata)
    if not context.parameters.random_seed and context.parameters.seed is not None:
        generator.manual_seed(context.parameters.seed + sample_index)
    elif context.parameters.random_seed:
        generator.manual_seed(random.randint(0, 2**31 - 1))
    return generator


def _resolve_device(adapter: Any) -> str:
    device = getattr(adapter, "device", "cuda:0")
    if not isinstance(device, str) or not device:
        return "cuda:0"
    if device == "cuda":
        return "cuda:0"
    return device


def _gpu_id_from_device(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    parts = device.split(":", 1)
    if len(parts) == 1:
        return 0
    try:
        return max(0, int(parts[1]))
    except ValueError:
        return 0


def _prompt(context: GenerationPostprocessorContext) -> str:
    prompt = context.parameters.prompt.strip()
    if not prompt:
        return DEFAULT_PROMPT
    return f"{prompt}, seamless transition, matching color, matching lighting, matching brushwork"


def _parameter(context: GenerationPostprocessorContext, key: str, default: Any) -> Any:
    return context.parameter(key, default)


def _bool_parameter(context: GenerationPostprocessorContext, key: str, default: bool) -> bool:
    value = _parameter(context, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _float_parameter(
    context: GenerationPostprocessorContext,
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
    context: GenerationPostprocessorContext,
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
    context: GenerationPostprocessorContext,
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    value = str(_parameter(context, key, default))
    return value if value in allowed else default


def _sample_index(metadata: dict[str, Any]) -> int:
    try:
        return int(metadata.get("sample_index", 0))
    except (TypeError, ValueError):
        return 0


def _seam_refine_report(
    settings: SeamRefineSettings,
    context: GenerationPostprocessorContext,
    seam_mask: Image.Image,
    reference: Image.Image,
) -> dict[str, Any]:
    return {
        "enabled": settings.enabled,
        "sample_index": _sample_index(context.metadata),
        "settings": {
            PARAM_ENABLED: settings.enabled,
            PARAM_WIDTH: settings.width,
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
        "input_size": list(context.generated.size),
        "mask_size": list(seam_mask.size),
        "reference_size": list(reference.size),
    }


def _artifact_dir(metadata: dict[str, Any]) -> Path | None:
    artifact_dir = metadata.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_filename(metadata: dict[str, Any], filename: str) -> str:
    return f"sample_{_sample_index(metadata):02d}_{filename}"


def _save_image(metadata: dict[str, Any], filename: str, image: Image.Image) -> None:
    artifact_dir = _artifact_dir(metadata)
    if artifact_dir is not None:
        image.save(artifact_dir / _artifact_filename(metadata, filename))


def _save_json(metadata: dict[str, Any], filename: str, payload: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(metadata)
    if artifact_dir is not None:
        (artifact_dir / _artifact_filename(metadata, filename)).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def _unload_ip_adapter(pipeline: Any) -> None:
    unload = getattr(pipeline, "unload_ip_adapter", None)
    if callable(unload):
        unload()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def register(context: Any) -> None:
    context.register_generation_postprocessor(SdxlSeamRefinePostprocessor())
