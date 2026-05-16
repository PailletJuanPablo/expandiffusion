"""Diffusers-backed inpaint adapters."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from .. import constants
from ..errors import AppError, GenerationCancelled
from ..image_utils import expand_mask_to_block_grid, feather_diffusion_mask
from ..runtime import resolve_device_and_dtype
from ..schemas import (
    AdapterCapabilities,
    ControlOption,
    ControlSchema,
    ModelLoadRequest,
    ModelSourceSchema,
)
from .base import CancelCheck, GenerationContext, ModelAdapter

HUB_DOWNLOAD_PROGRESS_START = 0.05
HUB_DOWNLOAD_PROGRESS_END = 0.72

LoadProgressDetails = dict[str, int | str | None]
LoadProgressCallback = Callable[[float, str, LoadProgressDetails | None], None]
FileDownloadProgressCallback = Callable[[int, int | None], None]


class DiffusersInpaintAdapter(ModelAdapter):
    """Base class for Diffusers inpaint pipelines."""

    pipeline_class_name: str
    minimum_process_size = 64

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Any | None = None
        self.device = constants.DEFAULT_DEVICE
        self.dtype = constants.DEFAULT_DTYPE

    def load(
        self,
        config: ModelLoadRequest,
        progress: LoadProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> None:
        """Load a Diffusers pipeline from Hub id, local path, or single file."""
        try:
            import torch
            import torchvision  # noqa: F401
            from diffusers import (
                ChromaInpaintPipeline,
                FluxFillPipeline,
                StableDiffusionControlNetInpaintPipeline,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionInpaintPipeline,
                StableDiffusionXLControlNetInpaintPipeline,
                StableDiffusionXLControlNetUnionInpaintPipeline,
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionXLInpaintPipeline,
                ZImageInpaintPipeline,
            )
        except ImportError as exc:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "Diffusers, PyTorch and torchvision must be installed before loading a model.",
                status_code=500,
            ) from exc
        self._validate_transformers_version()

        pipeline_classes = {
            "ChromaInpaintPipeline": ChromaInpaintPipeline,
            "FluxFillPipeline": FluxFillPipeline,
            "StableDiffusionControlNetInpaintPipeline": StableDiffusionControlNetInpaintPipeline,
            "StableDiffusionImg2ImgPipeline": StableDiffusionImg2ImgPipeline,
            "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
            "StableDiffusionXLControlNetInpaintPipeline": (
                StableDiffusionXLControlNetInpaintPipeline
            ),
            "StableDiffusionXLControlNetUnionInpaintPipeline": (
                StableDiffusionXLControlNetUnionInpaintPipeline
            ),
            "StableDiffusionXLImg2ImgPipeline": StableDiffusionXLImg2ImgPipeline,
            "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
            "ZImageInpaintPipeline": ZImageInpaintPipeline,
        }
        pipeline_class = pipeline_classes[self.pipeline_class_name]
        source = (
            config.single_file_path
            or config.local_path
            or config.model_id
            or self.default_model_id
        )
        if source is None:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "A model id, local path, or single file path is required.",
                status_code=422,
            )

        pipeline = None
        try:
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.02, "Resolving device and dtype.")
            device, dtype, dtype_name = resolve_device_and_dtype(
                torch,
                config.device,
                self._requested_dtype(config),
            )
            _raise_if_load_cancelled(is_cancelled)
            load_kwargs = _diffusers_load_kwargs(dtype)
            load_kwargs = self._pipeline_load_kwargs(
                config,
                load_kwargs,
                dtype,
                progress,
                is_cancelled,
            )
            if config.single_file_path:
                if not self.capabilities.from_single_file:
                    raise AppError(
                        constants.ERROR_UNSUPPORTED_OPERATION,
                        "This adapter does not support single-file loading.",
                        status_code=422,
                    )
                _report_load_progress(progress, 0.1, "Loading single-file checkpoint.")
                _raise_if_load_cancelled(is_cancelled)
                load_kwargs = self._single_file_load_kwargs(
                    Path(source),
                    load_kwargs,
                    dtype,
                    progress,
                    is_cancelled,
                )
                pipeline = pipeline_class.from_single_file(source, **load_kwargs)
            else:
                load_source = source
                if config.model_id:
                    if is_cancelled is None:
                        load_source = _download_hub_snapshot(
                            source,
                            load_kwargs.get("token"),
                            progress,
                        )
                    else:
                        load_source = _download_hub_snapshot(
                            source,
                            load_kwargs.get("token"),
                            progress,
                            is_cancelled,
                        )
                _raise_if_load_cancelled(is_cancelled)
                _report_load_progress(progress, 0.74, "Loading Diffusers pipeline.")
                pipeline = pipeline_class.from_pretrained(load_source, **load_kwargs)
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.9, "Moving pipeline to device.")
            pipeline = self._move_pipeline_to_device(pipeline, device)
            _raise_if_load_cancelled(is_cancelled)
            if device.startswith("cuda"):
                pipeline.enable_attention_slicing()
            self._configure_safety_checker(pipeline, config.safety_checker)
            self.pipeline = pipeline
            self.device = device
            self.dtype = dtype_name
            self.loaded_config = config
            _report_load_progress(progress, 0.96, "Loading extensions.")
            _raise_if_load_cancelled(is_cancelled)
            self._load_extensions(config)
        except AppError:
            if pipeline is not None and _is_load_cancelled(is_cancelled):
                _release_pipeline(pipeline)
            raise
        except Exception as exc:
            if pipeline is not None:
                _release_pipeline(pipeline)
            message, details = _model_load_error(source, exc)
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                message,
                status_code=500,
                details=details,
            ) from exc

    def unload(self) -> None:
        """Release the current Diffusers pipeline."""
        self.pipeline = None
        self.loaded_config = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        """Run inpaint generation with progress and cancellation callbacks."""
        if self.pipeline is None:
            raise AppError(
                constants.ERROR_MODEL_NOT_LOADED,
                "Load a model before starting generation.",
                status_code=409,
            )
        try:
            import torch
        except ImportError as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "PyTorch is required for Diffusers generation.",
                status_code=500,
            ) from exc

        parameters = context.parameters
        source, mask, process_size, output_size = self._prepare_pipeline_inputs(
            context.source,
            context.mask,
            parameters.mask_blur,
        )
        generator_seed = random.randint(0, 2**31 - 1) if parameters.random_seed else parameters.seed
        _save_adapter_image(context.metadata, "adapter_source_input.png", source)
        _save_adapter_image(context.metadata, "adapter_mask_input.png", mask)
        _save_adapter_json(
            context.metadata,
            "adapter_inputs.json",
            {
                "process_size": list(process_size),
                "output_size": list(output_size),
                "strength": parameters.strength,
                "steps": parameters.steps,
                "guidance_scale": parameters.guidance_scale,
                "sample_count": parameters.sample_count,
                "scheduler": parameters.scheduler,
                "seed": generator_seed,
                "random_seed": parameters.random_seed,
                "fill_mode": parameters.fill_mode,
                "mask_blur": parameters.mask_blur,
                "inpaint_area": parameters.inpaint_area,
            },
        )
        generator = torch.Generator(device=self.device)
        if generator_seed is not None:
            generator.manual_seed(generator_seed)

        if (
            parameters.scheduler != constants.SCHEDULER_AUTO
            and parameters.scheduler in self.capabilities.schedulers
        ):
            self._set_scheduler(parameters.scheduler)

        total_steps = max(1, parameters.steps)

        def callback_on_step_end(
            _pipeline: Any,
            step: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            if context.is_cancelled():
                raise GenerationCancelled()
            context.progress((step + 1) / total_steps, "Diffusion step")
            return callback_kwargs

        try:
            pipeline_kwargs = {
                "prompt": parameters.prompt,
                "image": source,
                "mask_image": mask,
                "width": process_size[0],
                "height": process_size[1],
                "strength": parameters.strength,
                "num_inference_steps": parameters.steps,
                "guidance_scale": parameters.guidance_scale,
                "num_images_per_prompt": parameters.sample_count,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if self.supports_negative_prompt:
                pipeline_kwargs["negative_prompt"] = parameters.negative_prompt or None
            if self.max_sequence_length is not None:
                pipeline_kwargs["max_sequence_length"] = self.max_sequence_length
            if (
                self.supports_inpaint_crop
                and parameters.inpaint_area == constants.INPAINT_AREA_ONLY_MASKED
            ):
                pipeline_kwargs["padding_mask_crop"] = parameters.mask_crop_padding
            pipeline_kwargs = self._generation_pipeline_kwargs(
                pipeline_kwargs,
                context,
                process_size,
            )
            output = self.pipeline(**pipeline_kwargs)
            for sample_index, image in enumerate(output.images):
                _save_adapter_image(
                    context.metadata,
                    f"adapter_pipeline_output_{sample_index:02d}.png",
                    image,
                )
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "Diffusers generation failed.",
                status_code=500,
                details={"reason": str(exc)},
            ) from exc
        context.progress(constants.PROGRESS_FINISHED, "Generation complete")
        return [self._restore_output_size(image, output_size) for image in output.images]

    def _prepare_pipeline_inputs(
        self,
        source: Image.Image,
        mask: Image.Image,
        mask_blur: int,
    ) -> tuple[Image.Image, Image.Image, tuple[int, int], tuple[int, int]]:
        output_size = source.size
        scale = max(1.0, self.minimum_process_size / min(output_size))
        process_size = (
            _ceil_to_multiple(max(64, math.ceil(output_size[0] * scale)), 8),
            _ceil_to_multiple(max(64, math.ceil(output_size[1] * scale)), 8),
        )
        if process_size == output_size:
            return (
                source,
                feather_diffusion_mask(expand_mask_to_block_grid(mask), mask_blur),
                process_size,
                output_size,
            )
        return (
            source.resize(process_size, Image.Resampling.LANCZOS),
            feather_diffusion_mask(
                expand_mask_to_block_grid(mask.resize(process_size, Image.Resampling.NEAREST)),
                mask_blur,
            ),
            process_size,
            output_size,
        )

    def _pipeline_load_kwargs(
        self,
        config: ModelLoadRequest,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        """Return Diffusers pipeline load kwargs for this adapter."""
        return load_kwargs

    def _generation_pipeline_kwargs(
        self,
        pipeline_kwargs: dict[str, Any],
        context: GenerationContext,
        process_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Return Diffusers generation kwargs for this adapter."""
        return pipeline_kwargs

    def _restore_output_size(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        if image.size == output_size:
            return image
        return image.resize(output_size, Image.Resampling.LANCZOS)

    def _configure_safety_checker(self, pipeline: Any, enabled: bool) -> None:
        if not hasattr(pipeline, "safety_checker"):
            return
        if not enabled:
            pipeline.safety_checker = None

    def _move_pipeline_to_device(self, pipeline: Any, device: str) -> Any:
        return pipeline.to(device)

    def _validate_transformers_version(self) -> None:
        try:
            transformers_version = version("transformers")
        except PackageNotFoundError as exc:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "Transformers must be installed before loading a Diffusers model.",
                status_code=500,
            ) from exc
        major = int(transformers_version.split(".", maxsplit=1)[0])
        if major >= 5:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "Diffusers single-file checkpoint loading requires transformers < 5.",
                status_code=500,
                details={"installed_transformers": transformers_version},
            )

    def _load_extensions(self, config: ModelLoadRequest) -> None:
        if self.pipeline is None:
            return
        for lora in config.loras:
            self.pipeline.load_lora_weights(lora.path)
            self.pipeline.set_adapters(Path(lora.path).stem, adapter_weights=lora.scale)
        for inversion in config.textual_inversions:
            kwargs: dict[str, str] = {}
            if inversion.token:
                kwargs["token"] = inversion.token
            self.pipeline.load_textual_inversion(inversion.path, **kwargs)

    def _set_scheduler(self, scheduler_id: str) -> None:
        if self.pipeline is None:
            return
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            LMSDiscreteScheduler,
        )

        scheduler_classes = {
            constants.SCHEDULER_DPM_SOLVER: DPMSolverMultistepScheduler,
            constants.SCHEDULER_EULER: EulerDiscreteScheduler,
            constants.SCHEDULER_DDIM: DDIMScheduler,
            constants.SCHEDULER_LMS: LMSDiscreteScheduler,
        }
        scheduler_class = scheduler_classes.get(scheduler_id)
        if scheduler_class is None:
            return
        self.pipeline.scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)

    def _single_file_load_kwargs(
        self,
        source: Path,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        """Return extra Diffusers kwargs for single-file checkpoints."""
        return load_kwargs

    def _requested_dtype(self, config: ModelLoadRequest) -> str:
        """Return the dtype request used by runtime resolution."""
        return config.dtype


class DiffusersImg2ImgAdapter(DiffusersInpaintAdapter):
    """Base class for Diffusers img2img pipelines used as standard checkpoints."""

    supports_inpaint_crop = False
    default_strength = 0.75
    default_result_mode = constants.RESULT_MODE_FEATHER_KNOWN

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        """Run img2img generation on the prefilled selection."""
        if self.pipeline is None:
            raise AppError(
                constants.ERROR_MODEL_NOT_LOADED,
                "Load a model before starting generation.",
                status_code=409,
            )
        try:
            import torch
        except ImportError as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "PyTorch is required for Diffusers generation.",
                status_code=500,
            ) from exc

        parameters = context.parameters
        source, _mask, _process_size, output_size = self._prepare_pipeline_inputs(
            context.source,
            context.mask,
            parameters.mask_blur,
        )
        generator_seed = random.randint(0, 2**31 - 1) if parameters.random_seed else parameters.seed
        generator = torch.Generator(device=self.device)
        if generator_seed is not None:
            generator.manual_seed(generator_seed)

        if (
            parameters.scheduler != constants.SCHEDULER_AUTO
            and parameters.scheduler in self.capabilities.schedulers
        ):
            self._set_scheduler(parameters.scheduler)

        total_steps = max(1, parameters.steps)

        def callback_on_step_end(
            _pipeline: Any,
            step: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            if context.is_cancelled():
                raise GenerationCancelled()
            context.progress((step + 1) / total_steps, "Diffusion step")
            return callback_kwargs

        try:
            pipeline_kwargs = {
                "prompt": parameters.prompt,
                "image": source,
                "strength": parameters.strength,
                "num_inference_steps": parameters.steps,
                "guidance_scale": parameters.guidance_scale,
                "num_images_per_prompt": parameters.sample_count,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if self.supports_negative_prompt:
                pipeline_kwargs["negative_prompt"] = parameters.negative_prompt or None
            output = self.pipeline(**pipeline_kwargs)
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "Diffusers generation failed.",
                status_code=500,
                details={"reason": str(exc)},
            ) from exc
        context.progress(constants.PROGRESS_FINISHED, "Generation complete")
        return [self._restore_output_size(image, output_size) for image in output.images]


class Sd15Img2ImgAdapter(DiffusersImg2ImgAdapter):
    """Stable Diffusion 1.x standard checkpoint adapter."""

    id = constants.ADAPTER_SD15_IMG2IMG
    label = "Stable Diffusion 1.x Standard"
    family = constants.FAMILY_SD15
    description = "Img2img outpaint adapter for standard SD 1.x checkpoints."
    default_model_id = constants.MODEL_SD15
    pipeline_class_name = "StableDiffusionImg2ImgPipeline"
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=True,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )


class Sd15InpaintAdapter(DiffusersInpaintAdapter):
    """Stable Diffusion 1.5 inpaint adapter."""

    id = constants.ADAPTER_SD15_INPAINT
    label = "Stable Diffusion 1.5 Inpaint"
    family = constants.FAMILY_SD15
    description = "Native Diffusers inpaint and outpaint adapter for SD 1.5."
    default_model_id = constants.MODEL_SD15_INPAINT
    pipeline_class_name = "StableDiffusionInpaintPipeline"
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=True,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )


class Sd15ControlNetInpaintAdapter(DiffusersInpaintAdapter):
    """Stable Diffusion 1.5 inpaint adapter with ControlNet color conditioning."""

    id = constants.ADAPTER_SD15_CONTROLNET_INPAINT
    label = "Stable Diffusion 1.5 Tile ControlNet"
    family = constants.FAMILY_SD15
    description = "SD 1.5 inpaint and outpaint adapter with Tile ControlNet color sketch guidance."
    default_model_id = constants.MODEL_SD15_INPAINT
    pipeline_class_name = "StableDiffusionControlNetInpaintPipeline"
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=True,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=True,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )
    controlnet_default_model_id = constants.MODEL_SD15_CONTROLNET_TILE
    controlnet_options = [
        (constants.MODEL_SD15_CONTROLNET_TILE, "Tile / color sketch"),
        (constants.MODEL_SD15_CONTROLNET_SCRIBBLE, "Scribble / line guide"),
    ]

    def load_controls(self) -> list[ControlSchema]:
        """Return load controls including the ControlNet model selector."""
        controls = super().load_controls()
        controls.append(
            ControlSchema(
                id="controlnet_model_id",
                label="ControlNet model",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_RUNTIME,
                default_value=self.controlnet_default_model_id,
                options=[
                    ControlOption(id=model_id, label=label)
                    for model_id, label in self.controlnet_options
                ],
            )
        )
        return controls

    def generation_controls(self) -> list[ControlSchema]:
        """Return generation controls including ControlNet guidance strength."""
        controls = super().generation_controls()
        controls.extend(
            [
                ControlSchema(
                    id="controlnet_conditioning_scale",
                    label="ControlNet strength",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_BASIC,
                    default_value=constants.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                ),
                ControlSchema(
                    id="control_guidance_start",
                    label="Control start",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=constants.DEFAULT_CONTROL_GUIDANCE_START,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                ControlSchema(
                    id="control_guidance_end",
                    label="Control end",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=constants.DEFAULT_CONTROL_GUIDANCE_END,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ]
        )
        return controls

    def generation_defaults(self) -> dict[str, Any]:
        """Return default generation parameters including ControlNet defaults."""
        defaults = super().generation_defaults()
        defaults.update(
            {
                "conditioning_type": constants.CONDITIONING_TYPE_COLOR,
                "controlnet_conditioning_scale": constants.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
                "control_guidance_start": constants.DEFAULT_CONTROL_GUIDANCE_START,
                "control_guidance_end": constants.DEFAULT_CONTROL_GUIDANCE_END,
            }
        )
        return defaults

    def _pipeline_load_kwargs(
        self,
        config: ModelLoadRequest,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        """Load the ControlNet model and inject it into the inpaint pipeline."""
        try:
            from diffusers import ControlNetModel
        except ImportError as exc:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "Diffusers ControlNet support is required before loading this adapter.",
                status_code=500,
            ) from exc
        _raise_if_load_cancelled(is_cancelled)
        _report_load_progress(progress, 0.12, "Loading ControlNet model.")
        source = (
            config.controlnet_local_path
            or config.controlnet_model_id
            or self.controlnet_default_model_id
        )
        controlnet = ControlNetModel.from_pretrained(source, **load_kwargs)
        return {**load_kwargs, "controlnet": controlnet}

    def _generation_pipeline_kwargs(
        self,
        pipeline_kwargs: dict[str, Any],
        context: GenerationContext,
        process_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Add ControlNet conditioning image and guidance controls."""
        parameters = context.parameters
        control_image = context.conditioning_image or context.source
        return {
            **pipeline_kwargs,
            "control_image": control_image.convert("RGB").resize(
                process_size,
                Image.Resampling.LANCZOS,
            ),
            "controlnet_conditioning_scale": (
                parameters.controlnet_conditioning_scale
                if context.conditioning_image is not None
                else 0.0
            ),
            "control_guidance_start": parameters.control_guidance_start,
            "control_guidance_end": parameters.control_guidance_end,
        }


class SdxlControlNetInpaintAdapter(Sd15ControlNetInpaintAdapter):
    """Stable Diffusion XL inpaint adapter with ControlNet color conditioning."""

    id = constants.ADAPTER_SDXL_CONTROLNET_INPAINT
    label = "Stable Diffusion XL Tile ControlNet"
    family = constants.FAMILY_SDXL
    description = "SDXL inpaint and outpaint adapter with Tile ControlNet color sketch guidance."
    default_model_id = constants.MODEL_SDXL_INPAINT
    pipeline_class_name = "StableDiffusionXLControlNetInpaintPipeline"
    minimum_process_size = constants.SDXL_MIN_PROCESS_SIZE
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=True,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=False,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )
    controlnet_default_model_id = constants.MODEL_SDXL_CONTROLNET_TILE
    controlnet_options = [
        (constants.MODEL_SDXL_CONTROLNET_TILE, "Tile / color sketch"),
        (constants.MODEL_SDXL_CONTROLNET_SCRIBBLE, "Scribble / line guide"),
    ]


class SdxlControlNetUnionInpaintAdapter(DiffusersInpaintAdapter):
    """SDXL inpaint adapter using ControlNet Union repaint conditioning."""

    id = constants.ADAPTER_SDXL_CONTROLNET_UNION_INPAINT
    label = "SDXL ControlNet Union Inpaint"
    family = constants.FAMILY_SDXL
    description = "SDXL inpaint adapter using Xinsir ControlNet Union repaint mode."
    default_model_id = constants.MODEL_SDXL_FILL_CONTROLNET_UNION
    pipeline_class_name = "StableDiffusionXLControlNetUnionInpaintPipeline"
    minimum_process_size = constants.SDXL_MIN_PROCESS_SIZE
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=False,
        img2img=True,
        txt2img=False,
        lora=False,
        controlnet=True,
        ip_adapter=False,
        textual_inversion=False,
        safety_checker=False,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=False,
    )

    def load(
        self,
        config: ModelLoadRequest,
        progress: LoadProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> None:
        direct_config = config
        if config.model_id and not config.local_path and not config.single_file_path:
            direct_config = config.model_copy(
                update={"model_id": None, "local_path": config.model_id}
            )
        super().load(direct_config, progress, is_cancelled)
        if self.loaded_config is not None and direct_config is not config:
            self.loaded_config = self.loaded_config.model_copy(
                update={"model_id": config.model_id, "local_path": None}
            )

    def load_controls(self) -> list[ControlSchema]:
        controls = super().load_controls()
        controls.append(
            ControlSchema(
                id="controlnet_model_id",
                label="ControlNet Union model",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_RUNTIME,
                default_value=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET,
                options=[
                    ControlOption(
                        id=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET,
                        label="xinsir/controlnet-union-sdxl-1.0 promax",
                    )
                ],
            )
        )
        return controls

    def generation_controls(self) -> list[ControlSchema]:
        controls = [
            control
            for control in super().generation_controls()
            if control.id not in {"outpaint_max_width", "outpaint_max_height"}
        ]
        controls.extend(
            [
                ControlSchema(
                    id="controlnet_conditioning_scale",
                    label="ControlNet strength",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_BASIC,
                    default_value=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                ),
                ControlSchema(
                    id="control_guidance_start",
                    label="Control start",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=constants.DEFAULT_CONTROL_GUIDANCE_START,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                ControlSchema(
                    id="control_guidance_end",
                    label="Control end",
                    kind=constants.CONTROL_SLIDER,
                    section=constants.CONTROL_SECTION_ADVANCED,
                    default_value=constants.DEFAULT_CONTROL_GUIDANCE_END,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ]
        )
        return controls

    def generation_defaults(self) -> dict[str, Any]:
        defaults = super().generation_defaults()
        defaults.update(
            {
                "controlnet_conditioning_scale": 1.0,
                "control_guidance_start": constants.DEFAULT_CONTROL_GUIDANCE_START,
                "control_guidance_end": constants.DEFAULT_CONTROL_GUIDANCE_END,
            }
        )
        return defaults

    def _pipeline_load_kwargs(
        self,
        config: ModelLoadRequest,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        try:
            from diffusers import ControlNetUnionModel
            from diffusers.models.model_loading_utils import load_state_dict
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "Diffusers ControlNet Union support is required before loading this adapter.",
                status_code=500,
            ) from exc
        _raise_if_load_cancelled(is_cancelled)
        _report_load_progress(progress, 0.12, "Loading ControlNet Union promax.")
        source = (
            config.controlnet_local_path
            or config.controlnet_model_id
            or constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET
        )
        if Path(source).exists():
            controlnet_dir = Path(source)
            config_file = (
                controlnet_dir / constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_CONFIG
            )
            model_file = (
                controlnet_dir / constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_WEIGHTS
            )
        else:
            config_file = Path(
                hf_hub_download(
                    source,
                    filename=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_CONFIG,
                )
            )
            model_file = Path(
                hf_hub_download(
                    source,
                    filename=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_WEIGHTS,
                )
            )
        controlnet_config = ControlNetUnionModel.load_config(str(config_file))
        controlnet = ControlNetUnionModel.from_config(controlnet_config)
        state_dict = load_state_dict(str(model_file))
        controlnet, _, _, _, _, _ = ControlNetUnionModel._load_pretrained_model(
            controlnet,
            state_dict,
            [str(model_file)],
            str(source),
            list(state_dict.keys()),
            dtype=dtype,
        )
        del state_dict
        return {**load_kwargs, "controlnet": controlnet}

    def _generation_pipeline_kwargs(
        self,
        pipeline_kwargs: dict[str, Any],
        context: GenerationContext,
        process_size: tuple[int, int],
    ) -> dict[str, Any]:
        parameters = context.parameters
        return {
            **pipeline_kwargs,
            "control_image": _repaint_control_image(
                pipeline_kwargs["image"],
                pipeline_kwargs["mask_image"],
            ).resize(process_size, Image.Resampling.LANCZOS),
            "control_mode": 7,
            "controlnet_conditioning_scale": parameters.controlnet_conditioning_scale,
            "control_guidance_start": parameters.control_guidance_start,
            "control_guidance_end": parameters.control_guidance_end,
        }


class SdxlFillControlNetUnionAdapter(ModelAdapter):
    """HF Space SDXL fill outpaint adapter."""

    id = constants.ADAPTER_SDXL_FILL_CONTROLNET_UNION
    label = "SDXL Fill ControlNet Union"
    family = constants.FAMILY_SDXL
    description = "Replicates the fffiloni Diffusers image outpaint Space pipeline."
    default_model_id = constants.MODEL_SDXL_FILL_CONTROLNET_UNION
    supports_negative_prompt = False
    returns_full_output = True
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=False,
        controlnet=True,
        ip_adapter=False,
        textual_inversion=False,
        safety_checker=False,
        schedulers=[constants.SCHEDULER_AUTO],
        from_single_file=False,
    )

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Any | None = None
        self.device = constants.DEFAULT_DEVICE
        self.dtype = constants.DEFAULT_DTYPE

    def load_controls(self) -> list[ControlSchema]:
        controls = super().load_controls()
        controls.append(
            ControlSchema(
                id="controlnet_model_id",
                label="ControlNet Union model",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_RUNTIME,
                default_value=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET,
                options=[
                    ControlOption(
                        id=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET,
                        label="xinsir/controlnet-union-sdxl-1.0 promax",
                    )
                ],
            )
        )
        return controls

    def generation_controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id="prompt",
                label="Prompt",
                kind=constants.CONTROL_TEXTAREA,
                section=constants.CONTROL_SECTION_BASIC,
                rows=4,
            ),
            ControlSchema(
                id="steps",
                label="Steps",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=8,
                min=4,
                max=50,
                step=1,
            ),
            ControlSchema(
                id="sample_count",
                label="Samples",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1,
                min=1,
                max=4,
                step=1,
            ),
            ControlSchema(
                id="guidance_scale",
                label="Guidance",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.5,
                min=0,
                max=5,
                step=0.1,
            ),
            ControlSchema(
                id="controlnet_conditioning_scale",
                label="ControlNet strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0,
                max=2,
                step=0.05,
            ),
            ControlSchema(
                id="outpaint_direction",
                label="Source alignment",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_BASIC,
                default_value="right",
                options=[
                    ControlOption(id="right", label="generate right"),
                    ControlOption(id="left", label="generate left"),
                    ControlOption(id="down", label="generate down"),
                    ControlOption(id="up", label="generate up"),
                    ControlOption(id="around", label="generate around"),
                ],
            ),
            ControlSchema(
                id="hf_space_resize_option",
                label="Resize input image",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_BASIC,
                default_value="Full",
                options=[
                    ControlOption(id="Full", label="Full"),
                    ControlOption(id="50%", label="50%"),
                    ControlOption(id="33%", label="33%"),
                    ControlOption(id="25%", label="25%"),
                    ControlOption(id="Custom", label="Custom"),
                ],
            ),
            ControlSchema(
                id="hf_space_custom_resize_percentage",
                label="Custom resize (%)",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=50,
                min=1,
                max=100,
                step=1,
            ),
            ControlSchema(
                id="hf_space_overlap_percentage",
                label="Mask overlap (%)",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=10,
                min=1,
                max=50,
                step=1,
            ),
            ControlSchema(
                id="hf_space_overlap_left",
                label="Overlap left",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=True,
            ),
            ControlSchema(
                id="hf_space_overlap_right",
                label="Overlap right",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=True,
            ),
            ControlSchema(
                id="hf_space_overlap_top",
                label="Overlap top",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=True,
            ),
            ControlSchema(
                id="hf_space_overlap_bottom",
                label="Overlap bottom",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=True,
            ),
            ControlSchema(
                id="random_seed",
                label="Random seed",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=True,
            ),
        ]

    def generation_defaults(self) -> dict[str, Any]:
        defaults = super().generation_defaults()
        defaults.update(
            {
                "steps": 8,
                "guidance_scale": 1.5,
                "strength": 1.0,
                "sample_count": 1,
                "scheduler": constants.SCHEDULER_AUTO,
                "fill_mode": constants.FILL_TRANSPARENT,
                "mask_blur": 0,
                "result_mode": constants.RESULT_MODE_PRESERVE_KNOWN,
                "outpaint_strategy": constants.OUTPAINT_STRATEGY_HF_SPACE_FILL,
                "outpaint_direction": "right",
                "hf_space_overlap_percentage": 10,
                "hf_space_overlap_left": True,
                "hf_space_overlap_right": True,
                "hf_space_overlap_top": True,
                "hf_space_overlap_bottom": True,
                "hf_space_resize_option": "Full",
                "hf_space_custom_resize_percentage": 50,
                "controlnet_conditioning_scale": 1.0,
                "correction_pipeline": [],
            }
        )
        return defaults

    def load(
        self,
        config: ModelLoadRequest,
        progress: LoadProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> None:
        try:
            import torch
            from diffusers import AutoencoderKL, TCDScheduler
            from diffusers.models.model_loading_utils import load_state_dict
            from huggingface_hub import hf_hub_download

            from .hf_space_outpaint.controlnet_union import ControlNetModel_Union
            from .hf_space_outpaint.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
        except ImportError as exc:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                (
                    "Diffusers, PyTorch and Hugging Face Hub are required for "
                    "SDXL Fill ControlNet Union."
                ),
                status_code=500,
            ) from exc

        controlnet_source = (
            config.controlnet_local_path
            or config.controlnet_model_id
            or constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET
        )
        source = config.local_path or config.model_id or self.default_model_id
        pipeline = None
        try:
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.02, "Resolving device and dtype.")
            device, dtype, dtype_name = resolve_device_and_dtype(torch, config.device, config.dtype)
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.1, "Loading ControlNet Union promax.")
            if Path(controlnet_source).exists():
                controlnet_dir = Path(controlnet_source)
                config_file = (
                    controlnet_dir
                    / constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_CONFIG
                )
                model_file = (
                    controlnet_dir
                    / constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_WEIGHTS
                )
            else:
                config_file = Path(
                    hf_hub_download(
                        controlnet_source,
                        filename=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_CONFIG,
                    )
                )
                model_file = Path(
                    hf_hub_download(
                        controlnet_source,
                        filename=constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET_WEIGHTS,
                    )
                )
            controlnet_config = ControlNetModel_Union.load_config(str(config_file))
            controlnet = ControlNetModel_Union.from_config(controlnet_config)
            state_dict = load_state_dict(str(model_file))
            controlnet, _, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
                controlnet,
                state_dict,
                [str(model_file)],
                str(controlnet_source),
                list(state_dict.keys()),
                dtype=dtype,
            )
            del state_dict
            controlnet.to(device=device, dtype=dtype)
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.35, "Loading SDXL fp16 VAE.")
            vae = AutoencoderKL.from_pretrained(
                constants.MODEL_SDXL_FILL_CONTROLNET_UNION_VAE,
                torch_dtype=dtype,
            ).to(device)
            _raise_if_load_cancelled(is_cancelled)
            _report_load_progress(progress, 0.55, "Loading RealVisXL Lightning fill pipeline.")
            pipeline = StableDiffusionXLFillPipeline.from_pretrained(
                source,
                torch_dtype=dtype,
                vae=vae,
                controlnet=controlnet,
                variant="fp16" if dtype_name == "float16" else None,
            ).to(device)
            pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
            self.pipeline = pipeline
            self.device = device
            self.dtype = dtype_name
            self.loaded_config = config.model_copy(
                update={
                    "model_id": source if not Path(str(source)).exists() else config.model_id,
                    "local_path": str(source) if Path(str(source)).exists() else config.local_path,
                    "controlnet_model_id": str(controlnet_source),
                    "device": device,
                    "dtype": dtype_name,
                }
            )
            _report_load_progress(progress, 1.0, "Model loaded.")
        except Exception:
            if pipeline is not None:
                del pipeline
            self.unload()
            raise

    def unload(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
        self.pipeline = None
        self.loaded_config = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        if self.pipeline is None:
            raise AppError(
                constants.ERROR_MODEL_NOT_LOADED,
                "Load a model before starting generation.",
                status_code=409,
            )
        parameters = context.parameters
        repaint_mode = context.metadata.get("generation_mode") == constants.GENERATION_MODE_INPAINT
        source = (
            _repaint_control_image(context.source, context.mask)
            if repaint_mode
            else context.source.convert("RGB")
        )
        extra_control_images: dict[int, Image.Image] = {}
        control_modes = [7] if repaint_mode else [6]
        if repaint_mode and context.conditioning_image is not None:
            scribble = _scribble_control_image(
                context.conditioning_image,
                context.source,
                context.mask,
            )
            if scribble is not None:
                extra_control_images[2] = scribble
                control_modes.append(2)
                _save_adapter_image(context.metadata, "adapter_scribble_input.png", scribble)
        _save_adapter_image(context.metadata, "adapter_controlnet_input.png", source)
        _save_adapter_image(context.metadata, "adapter_mask_input.png", context.mask.convert("L"))
        _save_adapter_json(
            context.metadata,
            "adapter_inputs.json",
            {
                "process_size": list(source.size),
                "output_size": list(source.size),
                "steps": parameters.steps,
                "guidance_scale": parameters.guidance_scale,
                "sample_count": parameters.sample_count,
                "controlnet_conditioning_scale": parameters.controlnet_conditioning_scale,
                "scheduler": constants.SCHEDULER_AUTO,
                "fill_mode": parameters.fill_mode,
                "result_mode": parameters.result_mode,
                "control_mode": 7 if repaint_mode else 6,
                "control_modes": control_modes,
                "uses_controlnet_scribble": 2 in control_modes,
                "hf_space_overlap_percentage": parameters.model_extra.get(
                    "hf_space_overlap_percentage",
                    10,
                )
                if parameters.model_extra
                else 10,
            },
        )
        final_prompt = f"{parameters.prompt} , high quality, 4k"
        try:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipeline.encode_prompt(final_prompt, self.device, True)
            final_images: list[Image.Image] = []
            sample_count = max(1, parameters.sample_count)
            total_outputs = max(1, sample_count * (parameters.steps + 1))
            completed_outputs = 0
            for sample_index in range(sample_count):
                final_image = None
                for output_index, image in enumerate(
                    self.pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        image=source,
                        num_inference_steps=parameters.steps,
                        guidance_scale=parameters.guidance_scale,
                        controlnet_conditioning_scale=parameters.controlnet_conditioning_scale,
                        control_mode=7 if repaint_mode else 6,
                        extra_control_images=extra_control_images or None,
                    )
                ):
                    if context.is_cancelled():
                        raise GenerationCancelled()
                    _save_adapter_image(
                        context.metadata,
                        f"adapter_pipeline_sample_{sample_index:02d}_output_{output_index:02d}.png",
                        image,
                    )
                    final_image = image
                    completed_outputs += 1
                    context.progress(
                        min(0.98, completed_outputs / total_outputs),
                        "Diffusion step",
                    )
                if final_image is None:
                    raise RuntimeError("Pipeline produced no image.")
                final_images.append(final_image.convert("RGB"))
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise AppError(
                constants.ERROR_GENERATION_FAILED,
                "Diffusers generation failed.",
                status_code=500,
                details={"reason": str(exc)},
            ) from exc
        context.progress(constants.PROGRESS_FINISHED, "Generation complete")
        return final_images


class Sd2InpaintAdapter(DiffusersInpaintAdapter):
    """Stable Diffusion 2 inpaint adapter."""

    id = "sd2-inpaint"
    label = "Stable Diffusion 2 Inpaint"
    family = constants.FAMILY_SD2
    description = "Native Diffusers inpaint and outpaint adapter for SD 2."
    default_model_id = constants.MODEL_SD2_INPAINT
    pipeline_class_name = "StableDiffusionInpaintPipeline"
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=True,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )


class SdxlImg2ImgAdapter(DiffusersImg2ImgAdapter):
    """Stable Diffusion XL standard checkpoint adapter."""

    id = constants.ADAPTER_SDXL_IMG2IMG
    label = "Stable Diffusion XL Standard"
    family = constants.FAMILY_SDXL
    description = "Img2img outpaint adapter for standard SDXL checkpoints."
    default_model_id = constants.MODEL_SDXL
    pipeline_class_name = "StableDiffusionXLImg2ImgPipeline"
    capabilities = AdapterCapabilities(
        inpaint=False,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=False,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )


class SdxlInpaintAdapter(DiffusersInpaintAdapter):
    """Stable Diffusion XL inpaint adapter."""

    id = constants.ADAPTER_SDXL_INPAINT
    label = "Stable Diffusion XL Inpaint"
    family = constants.FAMILY_SDXL
    description = "Native Diffusers inpaint and outpaint adapter for SDXL."
    default_model_id = constants.MODEL_SDXL_INPAINT
    pipeline_class_name = "StableDiffusionXLInpaintPipeline"
    minimum_process_size = constants.SDXL_MIN_PROCESS_SIZE
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=True,
        safety_checker=False,
        schedulers=constants.ALL_SCHEDULERS,
        from_single_file=True,
    )


class FluxFillAdapter(DiffusersInpaintAdapter):
    """FLUX.1 Fill inpaint and outpaint adapter."""

    id = constants.ADAPTER_FLUX_FILL
    label = "FLUX.1 Fill"
    family = constants.FAMILY_FLUX
    description = "Native Diffusers fill adapter for FLUX inpaint and outpaint."
    default_model_id = constants.MODEL_FLUX_FILL
    pipeline_class_name = "FluxFillPipeline"
    supports_negative_prompt = False
    supports_inpaint_crop = False
    default_steps = constants.DEFAULT_FLUX_STEPS
    default_guidance_scale = constants.DEFAULT_FLUX_GUIDANCE
    max_guidance_scale = constants.DEFAULT_FLUX_GUIDANCE
    default_scheduler = constants.SCHEDULER_AUTO
    max_sequence_length = constants.FLUX_MAX_SEQUENCE_LENGTH
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=False,
        safety_checker=False,
        schedulers=[constants.SCHEDULER_AUTO],
        from_single_file=False,
    )

    def _move_pipeline_to_device(self, pipeline: Any, device: str) -> Any:
        if device.startswith("cuda"):
            pipeline.enable_model_cpu_offload(gpu_id=_cuda_device_index(device))
            return pipeline
        return pipeline


class FluxFillFp8Adapter(FluxFillAdapter):
    """Local FLUX.1 Fill profile with an FP8 transformer override."""

    id = constants.ADAPTER_FLUX_FILL_FP8
    label = "FLUX.1 Fill FP8"
    description = "Local FLUX fill adapter using the existing Diffusers folder and FP8 transformer."
    default_model_id = None
    transformer_single_file_path = constants.LOCAL_FLUX_FILL_FP8_TRANSFORMER

    def load(
        self,
        config: ModelLoadRequest,
        progress: LoadProgressCallback | None = None,
        is_cancelled: CancelCheck | None = None,
    ) -> None:
        if config.model_id or config.single_file_path or config.model_url:
            super().load(config, progress, is_cancelled)
            return
        pipeline_dir = _resolve_flux_fill_local_pipeline_dir(
            Path(config.local_path)
            if config.local_path
            else constants.LOCAL_FLUX_FILL_DIFFUSERS_DIR
        )
        config = config.model_copy(update={"local_path": str(pipeline_dir)})
        super().load(config, progress, is_cancelled)

    def model_sources(self) -> list[ModelSourceSchema]:
        return [
            ModelSourceSchema(
                id=constants.MODEL_SOURCE_LOCAL_FOLDER,
                label="Existing local FLUX.1 Fill folder",
                request_field=constants.MODEL_SOURCE_FIELD_LOCAL_PATH,
                placeholder=str(_default_flux_fill_local_pipeline_dir()),
                default_value=str(_default_flux_fill_local_pipeline_dir()),
            )
        ]

    def _pipeline_load_kwargs(
        self,
        config: ModelLoadRequest,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        if not config.local_path:
            raise AppError(
                constants.ERROR_MODEL_LOAD_FAILED,
                "FLUX FP8 profile requires a local Diffusers folder.",
                status_code=422,
            )
        pipeline_dir = Path(config.local_path)
        transformer_path = Path(self.transformer_single_file_path)
        _report_load_progress(progress, 0.12, "Loading FLUX FP8 transformer.")
        _raise_if_load_cancelled(is_cancelled)
        transformer = _load_flux_transformer_from_single_file(
            transformer_path,
            pipeline_dir / "transformer",
            load_kwargs,
        )
        return {**load_kwargs, "local_files_only": True, "transformer": transformer}


class ChromaInpaintAdapter(DiffusersInpaintAdapter):
    """Chroma inpaint and outpaint adapter."""

    id = constants.ADAPTER_CHROMA_INPAINT
    label = "Chroma Inpaint"
    family = constants.FAMILY_CHROMA
    description = "Native Diffusers inpaint and outpaint adapter for Chroma."
    default_model_id = constants.MODEL_CHROMA_INPAINT
    pipeline_class_name = "ChromaInpaintPipeline"
    default_steps = constants.DEFAULT_CHROMA_STEPS
    default_guidance_scale = constants.DEFAULT_CHROMA_GUIDANCE
    default_strength = constants.DEFAULT_CHROMA_STRENGTH
    default_scheduler = constants.SCHEDULER_AUTO
    max_sequence_length = constants.CHROMA_MAX_SEQUENCE_LENGTH
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=False,
        safety_checker=False,
        schedulers=[constants.SCHEDULER_AUTO],
        from_single_file=True,
    )


class ZImageInpaintAdapter(DiffusersInpaintAdapter):
    """Z-Image inpaint and outpaint adapter."""

    id = constants.ADAPTER_ZIMAGE_INPAINT
    label = "Z-Image Inpaint"
    family = constants.FAMILY_ZIMAGE
    description = "Native Diffusers inpaint and outpaint adapter for Z-Image."
    default_model_id = constants.MODEL_ZIMAGE_INPAINT
    pipeline_class_name = "ZImageInpaintPipeline"
    supports_inpaint_crop = False
    default_steps = constants.DEFAULT_ZIMAGE_STEPS
    default_guidance_scale = constants.DEFAULT_ZIMAGE_GUIDANCE
    max_guidance_scale = 10.0
    default_scheduler = constants.SCHEDULER_AUTO
    max_sequence_length = constants.ZIMAGE_MAX_SEQUENCE_LENGTH
    capabilities = AdapterCapabilities(
        inpaint=True,
        outpaint=True,
        img2img=True,
        txt2img=False,
        lora=True,
        controlnet=False,
        ip_adapter=False,
        textual_inversion=False,
        safety_checker=False,
        schedulers=[constants.SCHEDULER_AUTO],
        from_single_file=True,
    )

    def _single_file_load_kwargs(
        self,
        source: Path,
        load_kwargs: dict[str, Any],
        dtype: Any,
        progress: LoadProgressCallback | None,
        is_cancelled: CancelCheck | None,
    ) -> dict[str, Any]:
        component_paths = _resolve_zimage_forge_components(source)
        if component_paths is None:
            return load_kwargs

        _report_load_progress(progress, 0.12, "Loading Z-Image text encoder.")
        _raise_if_load_cancelled(is_cancelled)
        text_encoder = _load_zimage_text_encoder(
            component_paths["text_encoder"],
            component_paths["text_encoder_config"],
            dtype,
        )
        _report_load_progress(progress, 0.18, "Loading Z-Image tokenizer.")
        _raise_if_load_cancelled(is_cancelled)
        tokenizer = _load_zimage_tokenizer(component_paths["tokenizer"])
        _report_load_progress(progress, 0.24, "Loading Z-Image VAE.")
        _raise_if_load_cancelled(is_cancelled)
        vae = _load_zimage_vae(
            component_paths["vae"],
            component_paths["vae_config"],
            dtype,
        )
        return {
            **load_kwargs,
            "config": str(component_paths["pipeline_config"]),
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "vae": vae,
        }

    def _requested_dtype(self, config: ModelLoadRequest) -> str:
        requested = config.dtype.strip().lower()
        if requested in {"", constants.DEFAULT_DTYPE}:
            return "bfloat16"
        return config.dtype


def _resolve_zimage_forge_components(source: Path) -> dict[str, Path] | None:
    forge_root = _find_forge_root(source)
    if forge_root is None:
        return None

    pipeline_config = forge_root / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo"
    if not pipeline_config.exists():
        return None

    modules = _forge_additional_modules(forge_root)
    models_root = forge_root / "models"
    qwen_candidates = [
        *modules,
        models_root / "text_encoder" / "qwen_3_4b.safetensors",
    ]
    vae_candidates = [
        *modules,
        models_root / "VAE" / "ae.safetensors",
    ]
    text_encoder = _first_existing_path(
        qwen_candidates,
        lambda path: "qwen" in path.name.lower() and path.suffix.lower() == ".safetensors",
    )
    vae = _first_existing_path(
        vae_candidates,
        lambda path: path.parent.name.lower() == "vae" and path.suffix.lower() == ".safetensors",
    )
    required_paths = {
        "text_encoder": text_encoder,
        "vae": vae,
        "text_encoder_config": pipeline_config / "text_encoder",
        "tokenizer": pipeline_config / "tokenizer",
        "vae_config": pipeline_config / "vae",
    }
    missing = [name for name, path in required_paths.items() if path is None or not path.exists()]
    if missing:
        raise AppError(
            constants.ERROR_MODEL_LOAD_FAILED,
            "Cannot resolve local Forge Z-Image components for this checkpoint.",
            status_code=422,
            details={
                "missing_components": ", ".join(missing),
                "forge_root": str(forge_root),
            },
        )

    return {
        "pipeline_config": pipeline_config,
        "text_encoder": text_encoder,
        "vae": vae,
        "text_encoder_config": pipeline_config / "text_encoder",
        "tokenizer": pipeline_config / "tokenizer",
        "vae_config": pipeline_config / "vae",
    }


def _find_forge_root(source: Path) -> Path | None:
    for parent in source.resolve().parents:
        if (parent / "models" / "Stable-diffusion").exists() and (parent / "config.json").exists():
            return parent
    return None


def _forge_additional_modules(forge_root: Path) -> list[Path]:
    config_path = forge_root / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    paths: list[Path] = []
    for key in (
        "forge_additional_modules_sd",
        "forge_additional_modules_qwen",
        "forge_additional_modules",
    ):
        values = raw_config.get(key)
        if not isinstance(values, list):
            continue
        paths.extend(Path(value) for value in values if isinstance(value, str))
    return paths


def _first_existing_path(paths: list[Path], predicate: Callable[[Path], bool]) -> Path | None:
    for path in paths:
        if path.exists() and predicate(path):
            return path
    return None


def _load_zimage_text_encoder(path: Path, config_dir: Path, dtype: Any) -> Any:
    from transformers import Qwen3ForCausalLM

    model_dir = _materialize_single_file_transformers_model(path, config_dir, "zimage-qwen3")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()
    return text_encoder


def _materialize_single_file_transformers_model(
    path: Path,
    config_dir: Path,
    cache_key: str,
) -> Path:
    target_dir = constants.DEFAULT_DATA_DIR / "component-cache" / cache_key / path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "generation_config.json"):
        source = config_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
    target_weights = target_dir / "model.safetensors"
    if target_weights.exists():
        try:
            if target_weights.samefile(path):
                return target_dir
        except OSError:
            pass
        target_weights.unlink()
    try:
        os.link(path, target_weights)
    except OSError as exc:
        raise AppError(
            constants.ERROR_MODEL_LOAD_FAILED,
            (
                "Cannot prepare local Z-Image text encoder weights without copying the "
                "safetensors file."
            ),
            status_code=500,
            details={"source": str(path), "target": str(target_weights), "reason": str(exc)},
        ) from exc
    return target_dir


def _load_zimage_tokenizer(config_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(config_dir), local_files_only=True)


def _load_zimage_vae(path: Path, config_dir: Path, dtype: Any) -> Any:
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_single_file(str(path), config=str(config_dir), torch_dtype=dtype)


def _load_flux_transformer_from_single_file(
    path: Path,
    config_dir: Path,
    load_kwargs: dict[str, Any],
) -> Any:
    if not path.exists():
        raise AppError(
            constants.ERROR_MODEL_LOAD_FAILED,
            f"FLUX FP8 transformer file was not found at '{path}'.",
            status_code=422,
            details={"path": str(path)},
        )
    if not config_dir.exists():
        raise AppError(
            constants.ERROR_MODEL_LOAD_FAILED,
            f"FLUX transformer config folder was not found at '{config_dir}'.",
            status_code=422,
            details={"path": str(config_dir)},
        )
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_single_file(
        str(path),
        config=str(config_dir),
        local_files_only=True,
        **load_kwargs,
    )


def _default_flux_fill_local_pipeline_dir() -> Path:
    return _resolve_flux_fill_local_pipeline_dir(constants.LOCAL_FLUX_FILL_DIFFUSERS_DIR)


def _resolve_flux_fill_local_pipeline_dir(path: Path) -> Path:
    if _is_complete_flux_fill_pipeline_dir(path):
        return path
    snapshot = _complete_flux_fill_huggingface_snapshot()
    return snapshot if snapshot is not None else path


def _complete_flux_fill_huggingface_snapshot() -> Path | None:
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--black-forest-labs--FLUX.1-Fill-dev"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir() and _is_complete_flux_fill_pipeline_dir(snapshot):
            return snapshot
    return None


def _is_complete_flux_fill_pipeline_dir(path: Path) -> bool:
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "text_encoder_2/model.safetensors.index.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "tokenizer_2/spiece.model",
        "tokenizer_2/tokenizer.json",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    return all((path / item).exists() for item in required)


def _diffusers_load_kwargs(dtype: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    token = _hugging_face_token()
    if token:
        kwargs["token"] = token
    return kwargs


def _download_hub_snapshot(
    repo_id: str,
    token: str | None,
    progress: LoadProgressCallback | None,
    is_cancelled: CancelCheck | None = None,
) -> str:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise AppError(
            constants.ERROR_MODEL_LOAD_FAILED,
            "huggingface_hub is required to download Hugging Face models with progress.",
            status_code=500,
        ) from exc

    _raise_if_load_cancelled(is_cancelled)
    _report_load_progress(progress, 0.05, f"Resolving Hugging Face files for {repo_id}.")
    info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    _raise_if_load_cancelled(is_cancelled)
    revision = getattr(info, "sha", None)
    files = _hub_snapshot_files(getattr(info, "siblings", []))
    total_files = len(files)
    if not files:
        _raise_if_load_cancelled(is_cancelled)
        snapshot_path = snapshot_download(repo_id, token=token, revision=revision)
        _raise_if_load_cancelled(is_cancelled)
        _report_load_progress(progress, 0.72, "Hugging Face download complete.")
        return snapshot_path

    known_total_bytes = all(file_size is not None for _filename, file_size in files)
    bytes_total = (
        sum(file_size or 0 for _filename, file_size in files) if known_total_bytes else None
    )
    completed_bytes = 0
    for index, (filename, expected_file_bytes) in enumerate(files, start=1):
        _raise_if_load_cancelled(is_cancelled)
        file_progress, last_file_bytes = _hub_file_progress_reporter(
            progress,
            filename,
            index,
            total_files,
            expected_file_bytes,
            completed_bytes,
            bytes_total,
            is_cancelled,
        )
        file_progress(0, expected_file_bytes)
        with _hub_file_progress_context(file_progress):
            local_file = hf_hub_download(
                repo_id,
                filename=filename,
                token=token,
                revision=revision,
            )
        _raise_if_load_cancelled(is_cancelled)
        finished_file_bytes = (
            expected_file_bytes or _path_size(Path(local_file)) or last_file_bytes()
        )
        completed_bytes += finished_file_bytes
        _report_load_progress(
            progress,
            _hub_download_progress_value(
                index,
                total_files,
                finished_file_bytes,
                expected_file_bytes,
                completed_bytes - finished_file_bytes,
                bytes_total,
            ),
            _hub_download_message(
                filename,
                index,
                total_files,
                finished_file_bytes,
                expected_file_bytes,
            ),
            {
                "files_done": index,
                "files_total": total_files,
                "file_name": filename,
                "file_bytes_done": finished_file_bytes,
                "file_bytes_total": expected_file_bytes,
                "bytes_done": completed_bytes if bytes_total is not None else None,
                "bytes_total": bytes_total,
            },
        )

    _raise_if_load_cancelled(is_cancelled)
    snapshot_path = snapshot_download(
        repo_id,
        token=token,
        revision=revision,
        local_files_only=True,
    )
    _raise_if_load_cancelled(is_cancelled)
    _report_load_progress(progress, 0.72, "Hugging Face download complete.")
    return snapshot_path


def _report_load_progress(
    progress: LoadProgressCallback | None,
    value: float,
    message: str,
    details: LoadProgressDetails | None = None,
) -> None:
    if progress is not None:
        progress(value, message, details)


def _hub_snapshot_files(siblings: list[Any]) -> list[tuple[str, int | None]]:
    files = []
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not filename or filename.endswith("/"):
            continue
        files.append((filename, _hub_file_size(sibling)))
    return files


def _hub_file_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    return size if isinstance(size, int) and size >= 0 else None


def _hub_download_progress_value(
    file_index: int,
    total_files: int,
    file_bytes_done: int,
    file_bytes_total: int | None,
    completed_bytes: int,
    bytes_total: int | None,
) -> float:
    if bytes_total:
        ratio = min((completed_bytes + file_bytes_done) / bytes_total, 1.0)
    else:
        file_ratio = min(file_bytes_done / file_bytes_total, 1.0) if file_bytes_total else 0.0
        ratio = min(((file_index - 1) + file_ratio) / max(total_files, 1), 1.0)
    return HUB_DOWNLOAD_PROGRESS_START + ratio * (
        HUB_DOWNLOAD_PROGRESS_END - HUB_DOWNLOAD_PROGRESS_START
    )


def _hub_download_message(
    filename: str,
    file_index: int,
    total_files: int,
    file_bytes_done: int,
    file_bytes_total: int | None,
) -> str:
    file_status = f"file {file_index}/{total_files}"
    if file_bytes_total:
        return (
            f"Downloading {filename}: {_format_bytes(file_bytes_done)} / "
            f"{_format_bytes(file_bytes_total)} ({file_status})."
        )
    return f"Downloading {filename}: {_format_bytes(file_bytes_done)} ({file_status})."


def _hub_file_progress_reporter(
    progress: LoadProgressCallback | None,
    filename: str,
    file_index: int,
    total_files: int,
    expected_file_bytes: int | None,
    completed_bytes: int,
    bytes_total: int | None,
    is_cancelled: CancelCheck | None = None,
) -> tuple[FileDownloadProgressCallback, Callable[[], int]]:
    last_file_bytes = 0

    def file_progress(file_bytes_done: int, file_bytes_total: int | None) -> None:
        nonlocal last_file_bytes
        _raise_if_load_cancelled(is_cancelled)
        last_file_bytes = max(last_file_bytes, file_bytes_done)
        current_file_total = file_bytes_total or expected_file_bytes
        current_bytes_done = completed_bytes + file_bytes_done
        _report_load_progress(
            progress,
            _hub_download_progress_value(
                file_index,
                total_files,
                file_bytes_done,
                current_file_total,
                completed_bytes,
                bytes_total,
            ),
            _hub_download_message(
                filename,
                file_index,
                total_files,
                file_bytes_done,
                current_file_total,
            ),
            {
                "files_done": file_index - 1,
                "files_total": total_files,
                "file_name": filename,
                "file_bytes_done": file_bytes_done,
                "file_bytes_total": current_file_total,
                "bytes_done": current_bytes_done if bytes_total is not None else None,
                "bytes_total": bytes_total,
            },
        )
        _raise_if_load_cancelled(is_cancelled)

    def get_last_file_bytes() -> int:
        return last_file_bytes

    return file_progress, get_last_file_bytes


@contextmanager
def _hub_file_progress_context(progress: FileDownloadProgressCallback):
    try:
        from huggingface_hub import file_download
    except ImportError:
        yield
        return

    original_context = getattr(file_download, "_get_progress_bar_context", None)
    if original_context is None:
        raise RuntimeError(
            "Installed huggingface_hub does not expose per-file download progress hooks."
        )

    def progress_bar_context(
        *,
        desc: str,
        log_level: int,
        total: int | None = None,
        initial: int = 0,
        unit: str = "B",
        unit_scale: bool = True,
        name: str | None = None,
        _tqdm_bar: Any | None = None,
    ) -> _DownloadProgressBar:
        return _DownloadProgressBar(progress, total, initial)

    file_download._get_progress_bar_context = progress_bar_context
    try:
        yield
    finally:
        file_download._get_progress_bar_context = original_context


class _DownloadProgressBar:
    def __init__(
        self,
        progress: FileDownloadProgressCallback,
        total: int | None,
        initial: int,
    ) -> None:
        self._progress = progress
        self.total = total if isinstance(total, int) and total >= 0 else None
        self.n = max(0, int(initial or 0))
        self._report()

    def __enter__(self) -> _DownloadProgressBar:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def update(self, value: int = 1) -> None:
        self.n += int(value or 0)
        self._report()

    def close(self) -> None:
        self._report()

    def _report(self) -> None:
        self._progress(self.n, self.total)


def _path_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{int(amount)} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _hugging_face_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _is_load_cancelled(is_cancelled: CancelCheck | None) -> bool:
    return is_cancelled is not None and is_cancelled()


def _raise_if_load_cancelled(is_cancelled: CancelCheck | None) -> None:
    if _is_load_cancelled(is_cancelled):
        raise AppError(
            constants.ERROR_MODEL_LOAD_CANCELLED,
            "Model load cancelled.",
            status_code=409,
        )


def _repaint_control_image(source: Image.Image, mask: Image.Image) -> Image.Image:
    source_rgb = source.convert("RGB")
    mask_binary = mask.convert("L").point(
        lambda pixel: 255 if pixel >= constants.WHITE_MASK_THRESHOLD else 0
    )
    black = Image.new("RGB", source_rgb.size, (0, 0, 0))
    return Image.composite(black, source_rgb, mask_binary)


def _scribble_control_image(
    conditioning: Image.Image,
    source: Image.Image,
    mask: Image.Image,
) -> Image.Image | None:
    source_rgb = source.convert("RGB")
    conditioning_rgb = conditioning.convert("RGB").resize(source_rgb.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(conditioning_rgb, source_rgb).convert("L")
    stroke_mask = diff.point(lambda pixel: 255 if pixel > 12 else 0)
    generation_mask = mask.convert("L").point(
        lambda pixel: 255 if pixel >= constants.WHITE_MASK_THRESHOLD else 0
    )
    stroke_mask = ImageChops.multiply(stroke_mask, generation_mask)
    if stroke_mask.getbbox() is None:
        return None
    scribble = Image.new("RGB", source_rgb.size, (0, 0, 0))
    scribble.paste((255, 255, 255), (0, 0), stroke_mask)
    return scribble


def _release_pipeline(pipeline: Any) -> None:
    del pipeline
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _model_load_error(source: str, exc: Exception) -> tuple[str, dict[str, str | bool]]:
    reason = str(exc)
    if _is_missing_single_file_component_error(reason):
        return (
            (
                f"Cannot load '{source}' as a standalone checkpoint. The checkpoint is missing "
                "required pipeline components such as text encoders, tokenizers, or VAE. Use the "
                "matching adapter with local components, or use a complete local Diffusers folder."
            ),
            {
                "reason": reason,
                "requires_full_diffusers_folder": True,
            },
        )
    if _is_gated_hub_error(reason):
        return (
            (
                f"Cannot access gated Hugging Face model '{source}'. Accept access to the "
                "model on Hugging Face, set HF_TOKEN in .env, then restart the API."
            ),
            {
                "reason": reason,
                "requires_hugging_face_token": True,
            },
        )
    return (
        f"Failed to load model '{source}'.",
        {"reason": reason},
    )


def _is_missing_single_file_component_error(reason: str) -> bool:
    normalized = reason.lower()
    return (
        "weights for this component appear to be missing in the checkpoint" in normalized
        and "from_single_file" in normalized
    )


def _is_gated_hub_error(reason: str) -> bool:
    normalized = reason.lower()
    return (
        "cannot access gated repo" in normalized
        or "access to model" in normalized and "restricted" in normalized
        or "401 client error" in normalized and "huggingface.co" in normalized
    )


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", maxsplit=1)[1])
    except ValueError:
        return 0


def _adapter_artifact_dir(metadata: dict[str, Any]) -> Path | None:
    artifact_dir = metadata.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_adapter_image(metadata: dict[str, Any], filename: str, image: Image.Image) -> None:
    artifact_dir = _adapter_artifact_dir(metadata)
    if artifact_dir is None:
        return
    image.save(artifact_dir / filename)


def _save_adapter_json(metadata: dict[str, Any], filename: str, payload: dict[str, Any]) -> None:
    artifact_dir = _adapter_artifact_dir(metadata)
    if artifact_dir is None:
        return
    (artifact_dir / filename).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
