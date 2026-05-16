"""Model adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .. import constants
from ..schemas import (
    AdapterCapabilities,
    AdapterInfo,
    ControlOption,
    ControlSchema,
    GenerationParameters,
    ModelLoadRequest,
    ModelSourceSchema,
)

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


@dataclass(slots=True)
class GenerationContext:
    """Runtime inputs passed into adapters."""

    source: Image.Image
    mask: Image.Image
    parameters: GenerationParameters
    progress: ProgressCallback
    is_cancelled: CancelCheck
    conditioning_image: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Base interface implemented by all generation adapters."""

    id: str
    label: str
    family: str
    description: str
    default_model_id: str | None
    capabilities: AdapterCapabilities
    supports_negative_prompt = True
    supports_inpaint_crop = True
    default_steps = constants.DEFAULT_STEPS
    default_guidance_scale = constants.DEFAULT_GUIDANCE
    max_guidance_scale = 20.0
    default_strength = constants.DEFAULT_STRENGTH
    default_scheduler = constants.SCHEDULER_DPM_SOLVER
    default_result_mode = constants.RESULT_MODE_GENERATED_SELECTION
    max_sequence_length: int | None = None

    def __init__(self) -> None:
        self.loaded_config: ModelLoadRequest | None = None

    @property
    def loaded(self) -> bool:
        """Whether the adapter currently has a pipeline loaded."""
        return self.loaded_config is not None

    def info(self, plugin_id: str | None = None) -> AdapterInfo:
        """Return serializable adapter metadata."""
        return AdapterInfo(
            id=self.id,
            label=self.label,
            family=self.family,
            description=self.description,
            default_model_id=self.default_model_id,
            capabilities=self.capabilities,
            loaded=self.loaded,
            plugin_id=plugin_id,
            model_sources=self.model_sources(),
            load_controls=self.load_controls(),
            generation_controls=self.generation_controls(),
            generation_defaults=self.generation_defaults(),
        )

    def model_sources(self) -> list[ModelSourceSchema]:
        """Return supported model source controls for this adapter."""
        sources = [
            ModelSourceSchema(
                id=constants.MODEL_SOURCE_HUB,
                label="Hugging Face model id",
                request_field=constants.MODEL_SOURCE_FIELD_MODEL_ID,
                default_value=self.default_model_id,
            ),
            ModelSourceSchema(
                id=constants.MODEL_SOURCE_LOCAL_FOLDER,
                label="Local Diffusers folder",
                request_field=constants.MODEL_SOURCE_FIELD_LOCAL_PATH,
                placeholder=r"E:\models\stable-diffusion-inpaint",
            ),
        ]
        if self.capabilities.from_single_file:
            sources.extend(
                [
                    ModelSourceSchema(
                        id=constants.MODEL_SOURCE_SINGLE_FILE,
                        label="Local checkpoint file",
                        request_field=constants.MODEL_SOURCE_FIELD_SINGLE_FILE_PATH,
                        placeholder=r"E:\models\model.safetensors",
                    ),
                    ModelSourceSchema(
                        id=constants.MODEL_SOURCE_DIRECT_URL,
                        label="Checkpoint URL",
                        request_field=constants.MODEL_SOURCE_FIELD_MODEL_URL,
                        placeholder="https://civitai.com/models/...?... or https://.../model.safetensors",
                    ),
                ]
            )
        return sources

    def load_controls(self) -> list[ControlSchema]:
        """Return model-load controls for this adapter."""
        controls = [
            ControlSchema(
                id="device",
                label="Device",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_RUNTIME,
                default_value=constants.DEFAULT_DEVICE,
            ),
            ControlSchema(
                id="dtype",
                label="Precision",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_RUNTIME,
                default_value=constants.DEFAULT_DTYPE,
                options=[
                    ControlOption(id="auto", label="Best for device"),
                    ControlOption(id="float16", label="float16"),
                    ControlOption(id="bfloat16", label="bfloat16"),
                    ControlOption(id="float32", label="float32"),
                ],
            ),
        ]
        if self.capabilities.safety_checker:
            controls.append(
                ControlSchema(
                    id="safety_checker",
                    label="Safety checker",
                    kind=constants.CONTROL_SWITCH,
                    section=constants.CONTROL_SECTION_RUNTIME,
                    default_value=True,
                )
            )
        if self.capabilities.lora:
            controls.append(
                ControlSchema(
                    id="loras",
                    label="LoRA paths",
                    kind=constants.CONTROL_TEXTAREA,
                    section=constants.CONTROL_SECTION_EXTENSIONS,
                    rows=3,
                    placeholder="path/to/lora.safetensors | 0.75",
                )
            )
        if self.capabilities.textual_inversion:
            controls.append(
                ControlSchema(
                    id="textual_inversions",
                    label="Textual inversions",
                    kind=constants.CONTROL_TEXTAREA,
                    section=constants.CONTROL_SECTION_EXTENSIONS,
                    rows=3,
                    placeholder="path/to/embedding.bin | token",
                )
            )
        return controls

    def generation_controls(self) -> list[ControlSchema]:
        """Return generation controls for this adapter."""
        controls = [
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
                default_value=self.default_steps,
                min=2,
                max=80,
                step=1,
            ),
            ControlSchema(
                id="sample_count",
                label="Samples",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=constants.DEFAULT_SAMPLE_COUNT,
                min=1,
                max=4,
                step=1,
            ),
            ControlSchema(
                id="guidance_scale",
                label="Guidance",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=self.default_guidance_scale,
                min=0 if self.default_guidance_scale == 0 else 1,
                max=self.max_guidance_scale,
                step=0.1,
            ),
            ControlSchema(
                id="strength",
                label="Strength",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=self.default_strength,
                min=0.05,
                max=1,
                step=0.01,
            ),
            ControlSchema(
                id="scheduler",
                label="Scheduler",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=self.default_scheduler,
                options=[
                    ControlOption(id=item, label=_scheduler_label(item))
                    for item in self.capabilities.schedulers
                ],
            ),
            ControlSchema(
                id="fill_mode",
                label="Fill preprocessor",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.FILL_OPENCV_NS,
                options=[
                    ControlOption(id=constants.FILL_PATCHMATCH, label="patchmatch"),
                    ControlOption(id=constants.FILL_EDGE_EXTEND, label="edge_pad"),
                    ControlOption(id=constants.FILL_OPENCV_NS, label="cv2_ns"),
                    ControlOption(id=constants.FILL_OPENCV_TELEA, label="cv2_telea"),
                    ControlOption(id=constants.FILL_PERLIN_NOISE, label="perlin"),
                    ControlOption(id=constants.FILL_GAUSSIAN_NOISE, label="gaussian"),
                    ControlOption(id=constants.FILL_TRANSPARENT, label="transparent"),
                ],
            ),
            ControlSchema(
                id="inpaint_area",
                label="Inpaint area",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.INPAINT_AREA_WHOLE_SELECTION,
                options=[
                    ControlOption(
                        id=constants.INPAINT_AREA_WHOLE_SELECTION,
                        label="whole selection",
                    ),
                    ControlOption(id=constants.INPAINT_AREA_ONLY_MASKED, label="only masked crop"),
                ],
            ),
            ControlSchema(
                id="mask_crop_padding",
                label="Mask padding",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.DEFAULT_MASK_CROP_PADDING,
                min=0,
                max=512,
                step=1,
            ),
            ControlSchema(
                id="mask_blur",
                label="Mask blur",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.DEFAULT_MASK_BLUR,
                min=0,
                max=64,
                step=1,
            ),
            ControlSchema(
                id="outpaint_max_width",
                label="Outpaint max width",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.DEFAULT_OUTPAINT_MAX_WIDTH,
                min=512,
                max=4096,
                step=64,
            ),
            ControlSchema(
                id="outpaint_max_height",
                label="Outpaint max height",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=constants.DEFAULT_OUTPAINT_MAX_HEIGHT,
                min=512,
                max=4096,
                step=64,
            ),
            ControlSchema(
                id="result_mode",
                label="Result mode",
                kind=constants.CONTROL_SELECT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=self.default_result_mode,
                options=[
                    ControlOption(
                        id=constants.RESULT_MODE_GENERATED_SELECTION,
                        label="generated selection",
                    ),
                    ControlOption(id=constants.RESULT_MODE_PRESERVE_KNOWN, label="preserve known"),
                    ControlOption(id=constants.RESULT_MODE_FEATHER_KNOWN, label="feather known"),
                    ControlOption(
                        id=constants.RESULT_MODE_RESTORE_ORIGINAL_SOFT,
                        label="restore original soft",
                    ),
                ],
            ),
            ControlSchema(
                id="random_seed",
                label="Random seed",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=True,
            ),
            ControlSchema(
                id="seed",
                label="Seed",
                kind=constants.CONTROL_NUMBER,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=0,
                min=0,
                max=2147483647,
                step=1,
            ),
            ControlSchema(
                id="img2img",
                label="Img2img",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=False,
            ),
        ]
        if self.supports_negative_prompt:
            controls.insert(
                1,
                ControlSchema(
                    id="negative_prompt",
                    label="Negative prompt",
                    kind=constants.CONTROL_TEXTAREA,
                    section=constants.CONTROL_SECTION_BASIC,
                    rows=2,
                ),
            )
        if not self.supports_inpaint_crop:
            controls = [
                control
                for control in controls
                if control.id not in {"inpaint_area", "mask_crop_padding"}
            ]
        return controls

    def generation_defaults(self) -> dict[str, Any]:
        """Return default generation parameters for this adapter."""
        return {
            "prompt": "",
            "negative_prompt": "",
            "width": constants.DEFAULT_WIDTH,
            "height": constants.DEFAULT_HEIGHT,
            "steps": self.default_steps,
            "guidance_scale": self.default_guidance_scale,
            "strength": self.default_strength,
            "seed": None,
            "random_seed": True,
            "sample_count": constants.DEFAULT_SAMPLE_COUNT,
            "scheduler": self.default_scheduler,
            "safety_checker": True,
            "img2img": False,
            "fill_mode": constants.FILL_OPENCV_NS,
            "correction_pipeline": [],
            "inpaint_area": constants.INPAINT_AREA_WHOLE_SELECTION,
            "mask_crop_padding": constants.DEFAULT_MASK_CROP_PADDING,
            "mask_blur": constants.DEFAULT_MASK_BLUR,
            "outpaint_max_width": constants.DEFAULT_OUTPAINT_MAX_WIDTH,
            "outpaint_max_height": constants.DEFAULT_OUTPAINT_MAX_HEIGHT,
            "result_mode": self.default_result_mode,
            "loras": [],
            "textual_inversions": [],
        }

    @abstractmethod
    def load(self, config: ModelLoadRequest) -> None:
        """Load adapter resources."""

    @abstractmethod
    def unload(self) -> None:
        """Release adapter resources."""

    @abstractmethod
    def generate(self, context: GenerationContext) -> list[Image.Image]:
        """Generate images for the provided context."""


def _scheduler_label(scheduler_id: str) -> str:
    labels = {
        constants.SCHEDULER_AUTO: "Auto",
        constants.SCHEDULER_DPM_SOLVER: "DPM++ 2M",
        constants.SCHEDULER_EULER: "Euler",
        constants.SCHEDULER_DDIM: "DDIM",
        constants.SCHEDULER_LMS: "LMS",
    }
    return labels.get(scheduler_id, scheduler_id)
