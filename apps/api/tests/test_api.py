"""Backend API tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from expandiffusion import constants
from expandiffusion.adapters.base import GenerationContext, ModelAdapter
from expandiffusion.adapters.diffusers_inpaint import (
    ChromaInpaintAdapter,
    FluxFillAdapter,
    FluxFillFp8Adapter,
    Sd15Img2ImgAdapter,
    Sd15InpaintAdapter,
    SdxlFillControlNetUnionAdapter,
    SdxlImg2ImgAdapter,
    SdxlInpaintAdapter,
    ZImageInpaintAdapter,
)
from expandiffusion.adapters.registry import AdapterRegistry, create_default_registry
from expandiffusion.errors import AppError
from expandiffusion.image_utils import (
    compose_generation_result,
    decode_data_url,
    encode_png_data_url,
    expand_mask_to_block_grid,
    generation_mask_from_alpha,
    match_generated_lighting_to_preserved_region,
    measure_seam_discontinuity,
    prepare_source_image,
)
from expandiffusion.jobs import JobStore
from expandiffusion.main import app, models, registry
from expandiffusion.model_storage import DOWNLOAD_CHUNK_SIZE, ModelStorage
from expandiffusion.persistence import PersistenceStore
from expandiffusion.plugin_actions import PluginActionRegistry, PluginToolRegistry
from expandiffusion.plugins import PluginManager, load_local_plugins
from expandiffusion.postprocessors import (
    GenerationPostprocessor,
    GenerationPostprocessorContext,
    GenerationPostprocessorRegistry,
)
from expandiffusion.schemas import (
    AdapterCapabilities,
    GenerationParameters,
    ModelInfo,
    ModelLoadRequest,
    OutpaintRequest,
    PluginActionResult,
    PluginActionRunRequest,
)
from expandiffusion.services import (
    GenerationService,
    ModelService,
    _compose_adapter_result,
    _prepare_hf_space_fill_request,
)

client = TestClient(app)


def _data_url(color: tuple[int, int, int, int]) -> str:
    image = Image.new("RGBA", (96, 96), color)
    return encode_png_data_url(image)


def _solid_data_url(size: tuple[int, int], color: tuple[int, int, int, int]) -> str:
    image = Image.new("RGBA", size, color)
    return encode_png_data_url(image)


def _mask_data_url(size: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    return encode_png_data_url(mask)


def _half_transparent_image(width: int, height: int, split_x: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for x in range(split_x):
        for y in range(height):
            image.putpixel((x, y), (32, 32, 36, 255))
    return image


def _half_generation_mask(width: int, height: int, split_x: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    for x in range(split_x, width):
        for y in range(height):
            mask.putpixel((x, y), 255)
    return mask


def _wait_for_job(job_id: str, timeout_seconds: float = 120.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {
            constants.JOB_SUCCEEDED,
            constants.JOB_FAILED,
            constants.JOB_CANCELLED,
        }:
            return job
        time.sleep(0.25)
    pytest.fail(f"Job {job_id} did not finish within {timeout_seconds} seconds.")


def _synthetic_seam_images() -> tuple[Image.Image, Image.Image, Image.Image]:
    original = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    generated = Image.new("RGB", (64, 32), (40, 50, 60))
    mask = Image.new("L", (64, 32), 0)
    for x in range(32):
        for y in range(32):
            original.putpixel((x, y), (180, 190, 200, 255))
            generated.putpixel((x, y), (180, 190, 200))
    for x in range(32, 64):
        for y in range(32):
            mask.putpixel((x, y), 255)
    return original, generated, mask


def _loaded_postprocessor(processor_id: str) -> GenerationPostprocessor:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    return next(item for item in postprocessors.processors() if item.id == processor_id)


class _ArithmeticCorrection(GenerationPostprocessor):
    category = constants.POSTPROCESSOR_CATEGORY_CORRECTION

    def __init__(self, processor_id: str, operation: str) -> None:
        self.id = processor_id
        self.label = processor_id
        self.operation = operation

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        red, green, blue = context.generated.getpixel((0, 0))
        if self.operation == "double":
            red *= 2
        if self.operation == "add":
            red += 5
        return Image.new("RGB", context.generated.size, (red, green, blue))


class _FullOutputTestAdapter(ModelAdapter):
    id = "full-output-test"
    label = "Full Output Test"
    family = constants.FAMILY_SDXL
    description = "Full output test adapter."
    default_model_id = None
    returns_full_output = True
    capabilities = AdapterCapabilities(outpaint=True)

    def load(self, config: ModelLoadRequest) -> None:
        self.loaded_config = config

    def unload(self) -> None:
        self.loaded_config = None

    def generate(self, context: GenerationContext) -> list[Image.Image]:
        return [Image.new("RGB", context.source.size, (200, 0, 0))]


class _ResultRefineProbe(GenerationPostprocessor):
    id = "result-refine-probe"
    label = "Result Refine Probe"
    category = "result_refine"

    def __init__(self) -> None:
        self.seen_pixels: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        self.seen_metadata: dict[str, object] = {}

    def process(self, context: GenerationPostprocessorContext) -> Image.Image:
        self.seen_pixels.append(
            (context.generated.getpixel((0, 0)), context.generated.getpixel((2, 0)))
        )
        self.seen_metadata = context.metadata
        output = context.generated.copy()
        output.putpixel((0, 0), (1, 2, 3))
        return output


def _write_plugin(tmp_path, plugin_id: str, adapter_id: str, hook_name: str) -> None:
    plugin_dir = tmp_path / plugin_id
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        (
            "{\n"
            f'  "id": "{plugin_id}",\n'
            f'  "label": "{plugin_id}",\n'
            '  "version": "0.1.0"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        _plugin_source(adapter_id=adapter_id, hook_name=hook_name),
        encoding="utf-8",
    )


def _plugin_source(adapter_id: str, hook_name: str) -> str:
    return f'''
from expandiffusion.adapters.base import ModelAdapter
from expandiffusion.schemas import AdapterCapabilities


class SampleAdapter(ModelAdapter):
    id = "{adapter_id}"
    label = "Sample Adapter"
    family = "sample"
    description = "Sample plugin adapter."
    default_model_id = None
    capabilities = AdapterCapabilities(inpaint=True, outpaint=True)

    def load(self, config):
        self.loaded_config = config

    def unload(self):
        self.loaded_config = None

    def generate(self, context):
        return [context.source]


def {hook_name}(context):
    context.register_model_adapter(SampleAdapter())
'''


def _write_postprocessor_plugin(tmp_path, plugin_id: str, processor_id: str) -> None:
    plugin_dir = tmp_path / plugin_id
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        (
            "{\n"
            f'  "id": "{plugin_id}",\n'
            f'  "label": "{plugin_id}",\n'
            '  "version": "0.1.0"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        f'''
from expandiffusion import constants
from expandiffusion.postprocessors import GenerationPostprocessor
from expandiffusion.schemas import ControlSchema


class SamplePostprocessor(GenerationPostprocessor):
    id = "{processor_id}"
    label = "Sample Postprocessor"

    def generation_controls(self):
        return [
            ControlSchema(
                id="sample_postprocessor_enabled",
                label="Sample postprocessor",
                kind=constants.CONTROL_SWITCH,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value=False,
            )
        ]

    def generation_defaults(self):
        return {{"sample_postprocessor_enabled": False}}

    def process(self, context):
        return context.generated


def register(context):
    context.register_generation_postprocessor(SamplePostprocessor())
''',
        encoding="utf-8",
    )


def _write_action_plugin(tmp_path, plugin_id: str, action_id: str) -> None:
    plugin_dir = tmp_path / plugin_id
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        (
            "{\n"
            f'  "id": "{plugin_id}",\n'
            f'  "label": "{plugin_id}",\n'
            '  "version": "0.1.0"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        f'''
from expandiffusion import constants
from expandiffusion.plugin_actions import PluginAction, PluginTool
from expandiffusion.schemas import ControlSchema, PluginActionResult


class SampleAction(PluginAction):
    id = "{action_id}"
    label = "Sample Action"

    def controls(self):
        return [
            ControlSchema(
                id="sample_action_suffix",
                label="Suffix",
                kind=constants.CONTROL_TEXT,
                section=constants.CONTROL_SECTION_ADVANCED,
                default_value="default",
            )
        ]

    def run(self, context):
        suffix = context.control("sample_action_suffix", "default")
        return PluginActionResult(
            action_id=self.id,
            text=f"{{context.image.width}}x{{context.image.height}} {{suffix}}",
        )


def register(context):
    action = SampleAction()
    context.register_action(action)
    context.register_tool(
        PluginTool(
            id="{action_id}",
            label="Sample Tool",
            action_id=action.id,
            icon="text-search",
            icon_color="#4f46e5",
            accent_color="#4f46e5",
            result_label="Sample result",
            controls=action.controls(),
            default_values=action.defaults(),
        )
    )
''',
        encoding="utf-8",
    )


def test_default_adapters_are_registered() -> None:
    response = client.get("/api/adapters")
    assert response.status_code == 200
    adapter_ids = {adapter["id"] for adapter in response.json()}
    assert adapter_ids == {
        constants.ADAPTER_SD15_IMG2IMG,
        "sd15-inpaint",
        "sd15-controlnet-inpaint",
        "sd2-inpaint",
        constants.ADAPTER_SDXL_IMG2IMG,
        "sdxl-inpaint",
        constants.ADAPTER_SDXL_CONTROLNET_INPAINT,
        constants.ADAPTER_SDXL_FILL_CONTROLNET_UNION,
        "sdxl-fill-ip-refine",
        constants.ADAPTER_FLUX_FILL,
        constants.ADAPTER_FLUX_FILL_FP8,
        constants.ADAPTER_CHROMA_INPAINT,
        constants.ADAPTER_ZIMAGE_INPAINT,
    }
    assert constants.ADAPTER_SD15_IMG2IMG in adapter_ids
    assert "sd15-inpaint" in adapter_ids
    assert "sd15-controlnet-inpaint" in adapter_ids
    assert "sd2-inpaint" in adapter_ids
    assert constants.ADAPTER_SDXL_IMG2IMG in adapter_ids
    assert "sdxl-inpaint" in adapter_ids
    assert constants.ADAPTER_SDXL_CONTROLNET_INPAINT in adapter_ids
    assert constants.ADAPTER_SDXL_FILL_CONTROLNET_UNION in adapter_ids
    assert constants.ADAPTER_FLUX_FILL in adapter_ids
    assert constants.ADAPTER_FLUX_FILL_FP8 in adapter_ids
    assert constants.ADAPTER_CHROMA_INPAINT in adapter_ids
    assert constants.ADAPTER_ZIMAGE_INPAINT in adapter_ids


def test_adapter_schema_exposes_dynamic_frontend_controls() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(item for item in response.json() if item["id"] == constants.ADAPTER_SD15_INPAINT)
    control_ids = {control["id"] for control in adapter["generation_controls"]}
    assert adapter["plugin_id"] is None
    assert {source["id"] for source in adapter["model_sources"]} == {
        constants.MODEL_SOURCE_HUB,
        constants.MODEL_SOURCE_LOCAL_FOLDER,
        constants.MODEL_SOURCE_SINGLE_FILE,
        constants.MODEL_SOURCE_DIRECT_URL,
    }
    assert any(control["id"] == "scheduler" for control in adapter["generation_controls"])
    assert "outpaint_max_width" in control_ids
    assert "outpaint_max_height" in control_ids
    assert adapter["generation_defaults"]["scheduler"] == constants.SCHEDULER_DPM_SOLVER
    assert (
        adapter["generation_defaults"]["outpaint_max_width"]
        == constants.DEFAULT_OUTPAINT_MAX_WIDTH
    )
    assert (
        adapter["generation_defaults"]["outpaint_max_height"]
        == constants.DEFAULT_OUTPAINT_MAX_HEIGHT
    )


def test_sd15_controlnet_schema_exposes_color_conditioning() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(item for item in response.json() if item["id"] == "sd15-controlnet-inpaint")
    control_ids = {control["id"] for control in adapter["generation_controls"]}
    load_controls = {control["id"]: control for control in adapter["load_controls"]}

    assert adapter["capabilities"]["controlnet"] is True
    assert adapter["capabilities"]["outpaint"] is True
    assert adapter["capabilities"]["inpaint"] is True
    assert adapter["label"] == "Stable Diffusion 1.5 Tile ControlNet"
    assert adapter["generation_defaults"]["conditioning_type"] == "color"
    assert adapter["generation_defaults"]["controlnet_conditioning_scale"] == 0.75
    assert (
        load_controls["controlnet_model_id"]["default_value"]
        == constants.MODEL_SD15_CONTROLNET_TILE
    )
    assert {option["id"] for option in load_controls["controlnet_model_id"]["options"]} == {
        constants.MODEL_SD15_CONTROLNET_TILE,
        constants.MODEL_SD15_CONTROLNET_SCRIBBLE,
    }
    assert "controlnet_conditioning_scale" in control_ids
    assert "control_guidance_start" in control_ids
    assert "control_guidance_end" in control_ids


def test_sdxl_controlnet_schema_exposes_color_conditioning() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(item for item in response.json() if item["id"] == "sdxl-controlnet-inpaint")
    control_ids = {control["id"] for control in adapter["generation_controls"]}
    load_controls = {control["id"]: control for control in adapter["load_controls"]}

    assert adapter["capabilities"]["controlnet"] is True
    assert adapter["capabilities"]["outpaint"] is True
    assert adapter["capabilities"]["inpaint"] is True
    assert adapter["label"] == "Stable Diffusion XL Tile ControlNet"
    assert adapter["generation_defaults"]["conditioning_type"] == "color"
    assert adapter["generation_defaults"]["controlnet_conditioning_scale"] == 0.75
    assert (
        load_controls["controlnet_model_id"]["default_value"]
        == constants.MODEL_SDXL_CONTROLNET_TILE
    )
    assert {option["id"] for option in load_controls["controlnet_model_id"]["options"]} == {
        constants.MODEL_SDXL_CONTROLNET_TILE,
        constants.MODEL_SDXL_CONTROLNET_SCRIBBLE,
    }
    assert "controlnet_conditioning_scale" in control_ids
    assert "control_guidance_start" in control_ids
    assert "control_guidance_end" in control_ids


def test_standard_stable_diffusion_adapters_expose_checkpoint_sources() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapters = response.json()
    sd15 = next(item for item in adapters if item["id"] == constants.ADAPTER_SD15_IMG2IMG)
    sdxl = next(item for item in adapters if item["id"] == constants.ADAPTER_SDXL_IMG2IMG)

    for adapter in [sd15, sdxl]:
        control_ids = {control["id"] for control in adapter["generation_controls"]}
        assert adapter["capabilities"]["outpaint"] is True
        assert adapter["capabilities"]["img2img"] is True
        assert adapter["capabilities"]["inpaint"] is False
        assert adapter["capabilities"]["from_single_file"] is True
        assert {source["id"] for source in adapter["model_sources"]} == {
            constants.MODEL_SOURCE_HUB,
            constants.MODEL_SOURCE_LOCAL_FOLDER,
            constants.MODEL_SOURCE_SINGLE_FILE,
            constants.MODEL_SOURCE_DIRECT_URL,
        }
        assert "inpaint_area" not in control_ids
        assert "mask_crop_padding" not in control_ids
        assert adapter["generation_defaults"]["strength"] == 0.75
        assert adapter["generation_defaults"]["result_mode"] == constants.RESULT_MODE_FEATHER_KNOWN

    assert sd15["default_model_id"] == constants.MODEL_SD15
    assert sdxl["default_model_id"] == constants.MODEL_SDXL


def test_flux_and_chroma_schemas_expose_runtime_specific_controls() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapters = response.json()
    flux = next(item for item in adapters if item["id"] == constants.ADAPTER_FLUX_FILL)
    chroma = next(item for item in adapters if item["id"] == constants.ADAPTER_CHROMA_INPAINT)
    flux_control_ids = {control["id"] for control in flux["generation_controls"]}
    chroma_control_ids = {control["id"] for control in chroma["generation_controls"]}

    assert flux["default_model_id"] == constants.MODEL_FLUX_FILL
    assert flux["capabilities"]["schedulers"] == [constants.SCHEDULER_AUTO]
    assert flux["generation_defaults"]["scheduler"] == constants.SCHEDULER_AUTO
    assert flux["generation_defaults"]["steps"] == constants.DEFAULT_FLUX_STEPS
    assert flux["generation_defaults"]["guidance_scale"] == constants.DEFAULT_FLUX_GUIDANCE
    assert flux["capabilities"]["from_single_file"] is False
    assert {source["id"] for source in flux["model_sources"]} == {
        constants.MODEL_SOURCE_HUB,
        constants.MODEL_SOURCE_LOCAL_FOLDER,
    }
    assert "negative_prompt" not in flux_control_ids
    assert "mask_crop_padding" not in flux_control_ids

    assert chroma["default_model_id"] == constants.MODEL_CHROMA_INPAINT
    assert chroma["capabilities"]["schedulers"] == [constants.SCHEDULER_AUTO]
    assert chroma["generation_defaults"]["scheduler"] == constants.SCHEDULER_AUTO
    assert chroma["generation_defaults"]["strength"] == constants.DEFAULT_CHROMA_STRENGTH
    assert "negative_prompt" in chroma_control_ids
    assert "mask_crop_padding" in chroma_control_ids


def test_sdxl_fill_controlnet_union_schema_matches_hf_space_defaults() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(
        item
        for item in response.json()
        if item["id"] == constants.ADAPTER_SDXL_FILL_CONTROLNET_UNION
    )
    control_ids = {control["id"] for control in adapter["generation_controls"]}
    load_control = next(
        control for control in adapter["load_controls"] if control["id"] == "controlnet_model_id"
    )

    assert adapter["default_model_id"] == constants.MODEL_SDXL_FILL_CONTROLNET_UNION
    assert adapter["capabilities"]["controlnet"] is True
    assert adapter["capabilities"]["outpaint"] is True
    assert adapter["capabilities"]["inpaint"] is True
    assert (
        adapter["generation_defaults"]["outpaint_strategy"]
        == constants.OUTPAINT_STRATEGY_HF_SPACE_FILL
    )
    assert adapter["generation_defaults"]["steps"] == 8
    assert adapter["generation_defaults"]["guidance_scale"] == 1.5
    assert adapter["generation_defaults"]["fill_mode"] == constants.FILL_TRANSPARENT
    assert adapter["generation_defaults"]["controlnet_conditioning_scale"] == 1.0
    assert adapter["generation_defaults"]["sample_count"] == 1
    assert adapter["generation_defaults"]["hf_space_overlap_left"] is True
    assert adapter["generation_defaults"]["hf_space_overlap_right"] is True
    assert adapter["generation_defaults"]["hf_space_overlap_top"] is True
    assert adapter["generation_defaults"]["hf_space_overlap_bottom"] is True
    assert load_control["default_value"] == constants.MODEL_SDXL_FILL_CONTROLNET_UNION_CONTROLNET
    assert "negative_prompt" not in control_ids
    assert "hf_space_overlap_percentage" in control_ids
    assert "hf_space_overlap_left" in control_ids
    assert "hf_space_overlap_right" in control_ids
    assert "hf_space_overlap_top" in control_ids
    assert "hf_space_overlap_bottom" in control_ids
    assert "sample_count" in control_ids
    assert "outpaint_direction" in control_ids
    controls = {control["id"]: control for control in adapter["generation_controls"]}
    assert controls["steps"]["max"] == 50
    assert controls["sample_count"]["max"] == 4


def test_sdxl_fill_controlnet_union_adapter_passes_repaint_conditioning(tmp_path) -> None:
    class DummyPipeline:
        def encode_prompt(self, *_args):
            return "prompt", "negative", "pooled", "negative-pooled"

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            yield Image.new("RGB", (1024, 1024), (30, 40, 50))

    pipeline = DummyPipeline()
    adapter = SdxlFillControlNetUnionAdapter()
    adapter.pipeline = pipeline
    adapter.device = "cpu"
    mask = Image.new("L", (1024, 1024), 0)
    mask.paste(255, (256, 256, 768, 768))
    context = GenerationContext(
        source=Image.new("RGB", (1024, 1024), (100, 120, 140)),
        mask=mask,
        parameters=GenerationParameters(
            prompt="replace object",
            steps=2,
            guidance_scale=7.0,
            strength=1.0,
            controlnet_conditioning_scale=0.9,
            random_seed=False,
            seed=123,
            scheduler=constants.SCHEDULER_AUTO,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
        metadata={
            "artifact_dir": str(tmp_path),
            "generation_mode": constants.GENERATION_MODE_INPAINT,
        },
    )

    adapter.generate(context)

    assert pipeline.kwargs["image"].size == (1024, 1024)
    assert pipeline.kwargs["control_mode"] == 7
    assert pipeline.kwargs["controlnet_conditioning_scale"] == 0.9
    assert pipeline.kwargs["image"].getpixel((300, 300)) == (0, 0, 0)
    assert pipeline.kwargs["image"].getpixel((100, 100)) == (100, 120, 140)


def test_sdxl_fill_controlnet_union_adapter_uses_last_space_output(tmp_path) -> None:
    class DummyPipeline:
        def encode_prompt(self, prompt, device, do_classifier_free_guidance):
            assert prompt == "extend scene"
            assert device == "cpu"
            assert do_classifier_free_guidance is True
            return "prompt", "negative", "pooled", "negative-pooled"

        def __call__(self, **kwargs):
            assert kwargs["prompt_embeds"] == "prompt"
            assert kwargs["image"].size == (96, 96)
            assert kwargs["num_inference_steps"] == 2
            assert kwargs["guidance_scale"] == 1.5
            assert kwargs["controlnet_conditioning_scale"] == 1.0
            assert kwargs["control_mode"] == 6
            yield Image.new("RGB", (96, 96), (10, 20, 30))
            yield Image.new("RGB", (96, 96), (70, 80, 90))

    adapter = SdxlFillControlNetUnionAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (96, 96), (0, 0, 0)),
        mask=Image.new("L", (96, 96), 255),
        parameters=GenerationParameters(
            prompt="extend scene",
            steps=2,
            guidance_scale=1.5,
            controlnet_conditioning_scale=1.0,
            fill_mode=constants.FILL_TRANSPARENT,
            result_mode=constants.RESULT_MODE_GENERATED_SELECTION,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
        metadata={"artifact_dir": str(tmp_path)},
    )

    images = adapter.generate(context)

    assert images[0].getpixel((0, 0)) == (70, 80, 90)
    adapter_inputs = json.loads((tmp_path / "adapter_inputs.json").read_text())
    assert adapter_inputs["process_size"] == [96, 96]
    assert adapter_inputs["steps"] == 2
    assert adapter_inputs["controlnet_conditioning_scale"] == 1.0
    assert adapter_inputs["control_mode"] == 6
    assert adapter_inputs["control_modes"] == [6]


def test_sdxl_fill_controlnet_union_adapter_generates_multiple_samples(tmp_path) -> None:
    class DummyPipeline:
        def __init__(self) -> None:
            self.calls = 0

        def encode_prompt(self, *_args):
            return "prompt", "negative", "pooled", "negative-pooled"

        def __call__(self, **_kwargs):
            self.calls += 1
            red = 40 * self.calls
            yield Image.new("RGB", (96, 96), (red, 20, 30))

    pipeline = DummyPipeline()
    adapter = SdxlFillControlNetUnionAdapter()
    adapter.pipeline = pipeline
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (96, 96), (0, 0, 0)),
        mask=Image.new("L", (96, 96), 255),
        parameters=GenerationParameters(
            prompt="extend scene",
            steps=2,
            guidance_scale=1.5,
            controlnet_conditioning_scale=1.0,
            sample_count=2,
            fill_mode=constants.FILL_TRANSPARENT,
            result_mode=constants.RESULT_MODE_PRESERVE_KNOWN,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
        metadata={"artifact_dir": str(tmp_path)},
    )

    images = adapter.generate(context)

    assert len(images) == 2
    assert images[0].getpixel((0, 0)) == (40, 20, 30)
    assert images[1].getpixel((0, 0)) == (80, 20, 30)
    adapter_inputs = json.loads((tmp_path / "adapter_inputs.json").read_text())
    assert adapter_inputs["sample_count"] == 2


def test_hf_space_fill_upscales_source_to_match_space_alignment() -> None:
    class FullOutputAdapter:
        returns_full_output = True

    image = Image.new("RGBA", (972, 636), (40, 50, 60, 255))
    parameters = GenerationParameters(
        width=2008,
        height=1320,
        outpaint_strategy=constants.OUTPAINT_STRATEGY_HF_SPACE_FILL,
        outpaint_direction="around",
        hf_space_resize_option="Full",
        hf_space_overlap_percentage=4,
    )

    prepared = _prepare_hf_space_fill_request(
        FullOutputAdapter(),
        image,
        parameters,
        constants.GENERATION_MODE_OUTPAINT,
    )

    assert prepared is not None
    _, _, _, metadata = prepared
    assert metadata["source_size"] == [2008, 1313]
    assert metadata["margin"] == [0, 3]


def test_hf_space_fill_can_exclude_specific_overlap_sides() -> None:
    class FullOutputAdapter:
        returns_full_output = True

    image = Image.new("RGBA", (100, 80), (40, 50, 60, 255))
    parameters = GenerationParameters(
        width=200,
        height=160,
        outpaint_strategy=constants.OUTPAINT_STRATEGY_HF_SPACE_FILL,
        outpaint_direction="around",
        hf_space_resize_option="Full",
        hf_space_overlap_percentage=10,
        hf_space_overlap_left=False,
        hf_space_overlap_right=True,
        hf_space_overlap_top=False,
        hf_space_overlap_bottom=True,
    )

    prepared = _prepare_hf_space_fill_request(
        FullOutputAdapter(),
        image,
        parameters,
        constants.GENERATION_MODE_OUTPAINT,
    )

    assert prepared is not None
    _, _, _, metadata = prepared
    assert metadata["overlap_sides"] == {
        "left": False,
        "right": True,
        "top": False,
        "bottom": True,
    }
    assert metadata["preserved_rect"] == [2, 2, 180, 144]


def test_hf_space_fill_uses_prepared_control_image_without_edge_pad() -> None:
    service = GenerationService.__new__(GenerationService)
    adapter = _FullOutputTestAdapter()
    image = Image.new("RGB", (6, 4), (0, 0, 0))
    image.paste((120, 130, 140), (1, 1, 4, 3))
    mask = Image.new("L", (6, 4), 255)
    mask.paste(0, (1, 1, 4, 3))
    parameters = GenerationParameters(
        outpaint_strategy=constants.OUTPAINT_STRATEGY_HF_SPACE_FILL,
        fill_mode=constants.FILL_EDGE_EXTEND,
    )

    source = service._prepare_source_for_generation(
        adapter,
        image,
        mask,
        parameters,
        constants.GENERATION_MODE_OUTPAINT,
    )

    assert source.getpixel((0, 0)) == (0, 0, 0)
    assert source.getpixel((2, 1)) == (120, 130, 140)


def test_free_full_output_outpaint_blacks_generation_mask_before_adapter() -> None:
    service = GenerationService.__new__(GenerationService)
    adapter = _FullOutputTestAdapter()
    image = Image.new("RGBA", (4, 1), (20, 30, 40, 255))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((1, 0), 255)
    mask.putpixel((2, 0), 255)
    mask.putpixel((3, 0), 255)
    parameters = GenerationParameters(
        outpaint_strategy="local_context",
        fill_mode=constants.FILL_TRANSPARENT,
    )

    source = service._prepare_source_for_generation(
        adapter,
        image,
        mask,
        parameters,
        constants.GENERATION_MODE_OUTPAINT,
    )

    assert source.getpixel((0, 0)) == (20, 30, 40)
    assert source.getpixel((1, 0)) == (0, 0, 0)
    assert source.getpixel((2, 0)) == (0, 0, 0)
    assert source.getpixel((3, 0)) == (0, 0, 0)


def test_outpaint_composition_restores_known_pixels_before_returning_result() -> None:
    source = Image.new("RGBA", (4, 1), (10, 20, 30, 255))
    generated = Image.new("RGB", (4, 1), (200, 0, 0))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((2, 0), 255)
    mask.putpixel((3, 0), 255)
    parameters = GenerationParameters(result_mode=constants.RESULT_MODE_GENERATED_SELECTION)

    result = _compose_adapter_result(
        constants.GENERATION_MODE_OUTPAINT,
        parameters,
        source,
        generated,
        mask,
    )

    assert result.getpixel((0, 0)) == (10, 20, 30)
    assert result.getpixel((1, 0)) == (10, 20, 30)
    assert result.getpixel((2, 0)) == (200, 0, 0)
    assert result.getpixel((3, 0)) == (200, 0, 0)


def test_hf_space_full_output_composition_keeps_generated_overlap() -> None:
    class FullOutputAdapter:
        returns_full_output = True

    source = Image.new("RGBA", (4, 1), (10, 20, 30, 255))
    generated = Image.new("RGB", (4, 1), (200, 0, 0))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((2, 0), 255)
    mask.putpixel((3, 0), 255)
    parameters = GenerationParameters(
        result_mode=constants.RESULT_MODE_PRESERVE_KNOWN,
        outpaint_strategy=constants.OUTPAINT_STRATEGY_HF_SPACE_FILL,
    )

    result = _compose_adapter_result(
        constants.GENERATION_MODE_OUTPAINT,
        parameters,
        source,
        generated,
        mask,
        FullOutputAdapter(),
    )

    assert result.getpixel((0, 0)) == (200, 0, 0)
    assert result.getpixel((1, 0)) == (200, 0, 0)
    assert result.getpixel((2, 0)) == (200, 0, 0)
    assert result.getpixel((3, 0)) == (200, 0, 0)


def test_flux_fp8_schema_uses_local_diffusers_profile() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(
        item for item in response.json() if item["id"] == constants.ADAPTER_FLUX_FILL_FP8
    )
    source = adapter["model_sources"][0]

    assert adapter["label"] == "FLUX.1 Fill FP8"
    assert adapter["default_model_id"] is None
    assert adapter["capabilities"]["from_single_file"] is False
    assert [item["id"] for item in adapter["model_sources"]] == [
        constants.MODEL_SOURCE_LOCAL_FOLDER
    ]
    assert source["request_field"] == constants.MODEL_SOURCE_FIELD_LOCAL_PATH
    assert source["default_value"] == str(constants.LOCAL_FLUX_FILL_DIFFUSERS_DIR)


def test_zimage_schema_exposes_forge_single_file_loading() -> None:
    response = client.get("/api/adapters")

    assert response.status_code == 200
    adapter = next(
        item for item in response.json() if item["id"] == constants.ADAPTER_ZIMAGE_INPAINT
    )
    control_ids = {control["id"] for control in adapter["generation_controls"]}

    assert adapter["default_model_id"] == constants.MODEL_ZIMAGE_INPAINT
    assert adapter["capabilities"]["schedulers"] == [constants.SCHEDULER_AUTO]
    assert adapter["capabilities"]["from_single_file"] is True
    assert adapter["generation_defaults"]["steps"] == constants.DEFAULT_ZIMAGE_STEPS
    assert adapter["generation_defaults"]["guidance_scale"] == constants.DEFAULT_ZIMAGE_GUIDANCE
    assert {source["id"] for source in adapter["model_sources"]} == {
        constants.MODEL_SOURCE_HUB,
        constants.MODEL_SOURCE_LOCAL_FOLDER,
        constants.MODEL_SOURCE_SINGLE_FILE,
        constants.MODEL_SOURCE_DIRECT_URL,
    }
    assert "negative_prompt" in control_ids
    assert "mask_crop_padding" not in control_ids


def test_flux_and_chroma_load_use_real_diffusers_pipeline_classes(monkeypatch) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    loaded_sources = []

    class FakePipeline:
        def __init__(self, source: str) -> None:
            self.source = source

        def to(self, device: str):
            self.device = device
            return self

        def enable_attention_slicing(self) -> None:
            self.attention_slicing = True

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cpu", torch.float32, "float32"

    def fake_validate_transformers_version(_adapter) -> None:
        return None

    def fake_flux_from_pretrained(source, **kwargs):
        loaded_sources.append(("flux", source, kwargs["torch_dtype"]))
        return FakePipeline(source)

    def fake_chroma_from_pretrained(source, **kwargs):
        loaded_sources.append(("chroma", source, kwargs["torch_dtype"]))
        return FakePipeline(source)

    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        fake_validate_transformers_version,
    )
    monkeypatch.setattr(
        diffusers_inpaint,
        "_download_hub_snapshot",
        lambda source, _token, _progress: source,
    )
    monkeypatch.setattr(
        diffusers.FluxFillPipeline,
        "from_pretrained",
        staticmethod(fake_flux_from_pretrained),
    )
    monkeypatch.setattr(
        diffusers.ChromaInpaintPipeline,
        "from_pretrained",
        staticmethod(fake_chroma_from_pretrained),
    )

    flux = FluxFillAdapter()
    chroma = ChromaInpaintAdapter()

    flux.load(ModelLoadRequest(adapter_id=flux.id, model_id=constants.MODEL_FLUX_FILL))
    chroma.load(ModelLoadRequest(adapter_id=chroma.id, model_id=constants.MODEL_CHROMA_INPAINT))

    assert loaded_sources == [
        ("flux", constants.MODEL_FLUX_FILL, torch.float32),
        ("chroma", constants.MODEL_CHROMA_INPAINT, torch.float32),
    ]
    assert flux.loaded is True
    assert chroma.loaded is True


def test_flux_cuda_load_uses_model_cpu_offload(monkeypatch) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    captured = {}

    class FakePipeline:
        def to(self, _device: str):
            raise AssertionError("Flux should not move the full pipeline to CUDA.")

        def enable_model_cpu_offload(self, gpu_id: int) -> None:
            captured["gpu_id"] = gpu_id

        def enable_attention_slicing(self) -> None:
            captured["attention_slicing"] = True

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cuda:0", torch.float16, "float16"

    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(
        diffusers_inpaint,
        "_download_hub_snapshot",
        lambda source, _token, _progress: source,
    )
    monkeypatch.setattr(
        diffusers.FluxFillPipeline,
        "from_pretrained",
        staticmethod(lambda _source, **_kwargs: FakePipeline()),
    )

    adapter = FluxFillAdapter()
    adapter.load(ModelLoadRequest(adapter_id=adapter.id, model_id=constants.MODEL_FLUX_FILL))

    assert captured == {"gpu_id": 0, "attention_slicing": True}
    assert adapter.device == "cuda:0"
    assert adapter.loaded is True


def test_flux_fp8_load_replaces_transformer_from_local_file(monkeypatch, tmp_path) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    local_pipeline = tmp_path / "FLUX.1-Fill-dev"
    transformer_config = local_pipeline / "transformer"
    transformer_file = tmp_path / "flux-fill-fp8.safetensors"
    transformer_config.mkdir(parents=True)
    transformer_file.write_bytes(b"fp8")
    captured = {}

    class FakePipeline:
        def __init__(self, transformer) -> None:
            self.transformer = transformer

        def to(self, device: str):
            captured["device"] = device
            return self

        def enable_attention_slicing(self) -> None:
            captured["attention_slicing"] = True

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cpu", torch.float32, "float32"

    def fake_transformer_from_single_file(source, **kwargs):
        captured["transformer_source"] = source
        captured["transformer_kwargs"] = kwargs
        return "fp8-transformer"

    def fake_pipeline_from_pretrained(source, **kwargs):
        captured["pipeline_source"] = source
        captured["pipeline_kwargs"] = kwargs
        return FakePipeline(kwargs["transformer"])

    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(
        diffusers.FluxTransformer2DModel,
        "from_single_file",
        staticmethod(fake_transformer_from_single_file),
    )
    monkeypatch.setattr(
        diffusers.FluxFillPipeline,
        "from_pretrained",
        staticmethod(fake_pipeline_from_pretrained),
    )

    adapter = FluxFillFp8Adapter()
    adapter.transformer_single_file_path = transformer_file
    adapter.load(ModelLoadRequest(adapter_id=adapter.id, local_path=str(local_pipeline)))

    assert captured["transformer_source"] == str(transformer_file)
    assert captured["transformer_kwargs"]["config"] == str(transformer_config)
    assert captured["transformer_kwargs"]["torch_dtype"] == torch.float32
    assert captured["pipeline_source"] == str(local_pipeline)
    assert captured["pipeline_kwargs"]["transformer"] == "fp8-transformer"
    assert adapter.pipeline.transformer == "fp8-transformer"
    assert adapter.loaded is True


def test_standard_stable_diffusion_load_uses_img2img_pipeline_classes(monkeypatch) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    loaded_sources = []

    class FakePipeline:
        def __init__(self, source: str) -> None:
            self.source = source

        def to(self, device: str):
            self.device = device
            return self

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cpu", torch.float32, "float32"

    def fake_sd15_from_single_file(source, **kwargs):
        loaded_sources.append(("sd15", source, kwargs["torch_dtype"]))
        return FakePipeline(source)

    def fake_sdxl_from_single_file(source, **kwargs):
        loaded_sources.append(("sdxl", source, kwargs["torch_dtype"]))
        return FakePipeline(source)

    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(
        diffusers.StableDiffusionImg2ImgPipeline,
        "from_single_file",
        staticmethod(fake_sd15_from_single_file),
    )
    monkeypatch.setattr(
        diffusers.StableDiffusionXLImg2ImgPipeline,
        "from_single_file",
        staticmethod(fake_sdxl_from_single_file),
    )

    sd15 = Sd15Img2ImgAdapter()
    sdxl = SdxlImg2ImgAdapter()

    sd15.load(
        ModelLoadRequest(
            adapter_id=sd15.id,
            single_file_path=r"E:\models\cyberrealistic.safetensors",
        )
    )
    sdxl.load(ModelLoadRequest(adapter_id=sdxl.id, single_file_path=r"E:\models\sdxl.safetensors"))

    assert loaded_sources == [
        ("sd15", r"E:\models\cyberrealistic.safetensors", torch.float32),
        ("sdxl", r"E:\models\sdxl.safetensors", torch.float32),
    ]
    assert sd15.loaded is True
    assert sdxl.loaded is True


def test_zimage_single_file_load_uses_forge_components(monkeypatch, tmp_path) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    checkpoint = tmp_path / "models" / "Stable-diffusion" / "cyberrealisticZImage.safetensors"
    qwen = tmp_path / "models" / "text_encoder" / "qwen_3_4b.safetensors"
    vae = tmp_path / "models" / "VAE" / "ae.safetensors"
    pipeline_config = tmp_path / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo"
    for path in [
        checkpoint,
        qwen,
        vae,
        pipeline_config / "text_encoder" / "config.json",
        pipeline_config / "tokenizer" / "tokenizer.json",
        pipeline_config / "vae" / "config.json",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        (
            "{"
            f'"forge_additional_modules_sd":["{vae.as_posix()}","{qwen.as_posix()}"]'
            "}"
        ),
        encoding="utf-8",
    )

    captured = {}

    class FakePipeline:
        def to(self, device: str):
            self.device = device
            return self

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        captured["requested_dtype"] = _dtype
        return "cpu", torch.float32, "float32"

    def fake_from_single_file(source, **kwargs):
        captured["source"] = source
        captured["kwargs"] = kwargs
        return FakePipeline()

    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(diffusers_inpaint, "_load_zimage_text_encoder", lambda *_args: "qwen")
    monkeypatch.setattr(diffusers_inpaint, "_load_zimage_tokenizer", lambda *_args: "tokenizer")
    monkeypatch.setattr(diffusers_inpaint, "_load_zimage_vae", lambda *_args: "vae")
    monkeypatch.setattr(
        diffusers.ZImageInpaintPipeline,
        "from_single_file",
        staticmethod(fake_from_single_file),
    )

    adapter = ZImageInpaintAdapter()
    adapter.load(ModelLoadRequest(adapter_id=adapter.id, single_file_path=str(checkpoint)))

    assert captured["source"] == str(checkpoint)
    assert captured["kwargs"]["config"] == str(pipeline_config)
    assert captured["kwargs"]["text_encoder"] == "qwen"
    assert captured["kwargs"]["tokenizer"] == "tokenizer"
    assert captured["kwargs"]["vae"] == "vae"
    assert captured["kwargs"]["torch_dtype"] == torch.float32
    assert captured["requested_dtype"] == "bfloat16"
    assert adapter.loaded is True


def test_diffusers_load_passes_hugging_face_token_from_env(monkeypatch) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    captured_kwargs = {}
    download_tokens = []

    class FakePipeline:
        def to(self, device: str):
            self.device = device
            return self

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cpu", torch.float32, "float32"

    def fake_from_pretrained(_source, **kwargs):
        captured_kwargs.update(kwargs)
        return FakePipeline()

    def fake_download_hub_snapshot(source, token, _progress):
        download_tokens.append(token)
        return source

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(
        diffusers_inpaint,
        "_download_hub_snapshot",
        fake_download_hub_snapshot,
    )
    monkeypatch.setattr(
        diffusers.FluxFillPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    adapter = FluxFillAdapter()
    adapter.load(ModelLoadRequest(adapter_id=adapter.id, model_id=constants.MODEL_FLUX_FILL))

    assert captured_kwargs["torch_dtype"] == torch.float32
    assert captured_kwargs["token"] == "test-token"
    assert download_tokens == ["test-token"]


def test_hugging_face_download_context_reports_byte_progress() -> None:
    from huggingface_hub import file_download

    from expandiffusion.adapters import diffusers_inpaint

    events = []

    with diffusers_inpaint._hub_file_progress_context(
        lambda done, total: events.append((done, total))
    ):
        with file_download._get_progress_bar_context(
            desc="weights.safetensors",
            log_level=20,
            total=100,
            initial=10,
        ) as progress_bar:
            progress_bar.update(25)
            progress_bar.update(65)

    assert events[0] == (10, 100)
    assert (35, 100) in events
    assert events[-1] == (100, 100)


def test_diffusers_gated_model_error_is_actionable(monkeypatch) -> None:
    import diffusers
    import torch

    from expandiffusion.adapters import diffusers_inpaint

    def fake_resolve_device_and_dtype(_torch, _device, _dtype):
        return "cpu", torch.float32, "float32"

    def fake_from_pretrained(_source, **_kwargs):
        raise RuntimeError(
            "401 Client Error. Cannot access gated repo. Access to model "
            "black-forest-labs/FLUX.1-Fill-dev is restricted. Please log in."
        )

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        diffusers_inpaint,
        "resolve_device_and_dtype",
        fake_resolve_device_and_dtype,
    )
    monkeypatch.setattr(
        diffusers_inpaint.DiffusersInpaintAdapter,
        "_validate_transformers_version",
        lambda _adapter: None,
    )
    monkeypatch.setattr(
        diffusers_inpaint,
        "_download_hub_snapshot",
        lambda source, _token, _progress: source,
    )
    monkeypatch.setattr(
        diffusers.FluxFillPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    adapter = FluxFillAdapter()
    with pytest.raises(AppError) as error:
        adapter.load(ModelLoadRequest(adapter_id=adapter.id, model_id=constants.MODEL_FLUX_FILL))

    assert "Cannot access gated Hugging Face model" in error.value.message
    assert "set HF_TOKEN in .env" in error.value.message
    assert error.value.details["requires_hugging_face_token"] is True


def test_flux_single_file_checkpoint_is_rejected() -> None:
    adapter = FluxFillAdapter()

    with pytest.raises(AppError) as error:
        adapter.load(
            ModelLoadRequest(
                adapter_id=adapter.id,
                single_file_path=r"E:\models\flux-fill-fp8.safetensors",
            )
        )

    assert error.value.code == constants.ERROR_UNSUPPORTED_OPERATION
    assert "does not support single-file loading" in error.value.message


def test_plugins_endpoint_returns_plugin_load_status() -> None:
    response = client.get("/api/plugins")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_valid_plugin_registers_adapter_with_owner(tmp_path) -> None:
    _write_plugin(
        tmp_path,
        plugin_id="sample-plugin",
        adapter_id="sample-adapter",
        hook_name="register",
    )
    registry = AdapterRegistry()

    plugins = load_local_plugins(registry, tmp_path)

    assert len(plugins) == 1
    assert plugins[0].loaded is True
    assert plugins[0].adapter_ids == ["sample-adapter"]
    adapter = registry.list()[0]
    assert adapter.id == "sample-adapter"
    assert adapter.plugin_id == "sample-plugin"


def test_legacy_plugin_hook_still_registers_adapter(tmp_path) -> None:
    _write_plugin(
        tmp_path,
        plugin_id="legacy-plugin",
        adapter_id="legacy-adapter",
        hook_name="register_model_adapters",
    )
    registry = AdapterRegistry()

    plugins = load_local_plugins(registry, tmp_path)

    assert plugins[0].loaded is True
    assert registry.get("legacy-adapter").id == "legacy-adapter"


def test_invalid_plugin_manifest_is_reported_without_registration(tmp_path) -> None:
    plugin_dir = tmp_path / "broken-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text("{}", encoding="utf-8")
    (plugin_dir / "plugin.py").write_text("def register(context):\n    pass\n", encoding="utf-8")
    registry = AdapterRegistry()

    plugins = load_local_plugins(registry, tmp_path)

    assert plugins[0].loaded is False
    assert plugins[0].error_code == constants.ERROR_PLUGIN_LOAD_FAILED
    assert registry.list() == []


def test_duplicate_plugin_adapter_id_is_reported_without_replacement(tmp_path) -> None:
    _write_plugin(
        tmp_path,
        plugin_id="duplicate-plugin",
        adapter_id=constants.ADAPTER_SD15_INPAINT,
        hook_name="register",
    )
    registry = create_default_registry()
    original_adapter = registry.get(constants.ADAPTER_SD15_INPAINT)

    plugins = load_local_plugins(registry, tmp_path)

    assert plugins[0].loaded is False
    assert plugins[0].error_code == constants.ERROR_PLUGIN_LOAD_FAILED
    assert registry.get(constants.ADAPTER_SD15_INPAINT) is original_adapter


def test_plugin_manager_can_disable_and_enable_plugin(tmp_path) -> None:
    _write_plugin(
        tmp_path,
        plugin_id="sample-plugin",
        adapter_id="sample-adapter",
        hook_name="register",
    )
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(registry, postprocessors, persistence, tmp_path)

    plugins = manager.load_all()

    assert plugins[0].loaded is True
    assert registry.get("sample-adapter").id == "sample-adapter"

    disabled = manager.disable("sample-plugin")

    assert disabled.enabled is False
    assert "sample-adapter" not in registry.adapter_ids()
    assert persistence.is_plugin_enabled("sample-plugin") is False

    enabled = manager.enable("sample-plugin")

    assert enabled.enabled is True
    assert enabled.loaded is True
    assert registry.get("sample-adapter").id == "sample-adapter"


def test_plugin_manager_skips_persistently_disabled_plugins(tmp_path) -> None:
    _write_plugin(
        tmp_path,
        plugin_id="sample-plugin",
        adapter_id="sample-adapter",
        hook_name="register",
    )
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    persistence.set_plugin_enabled("sample-plugin", False)
    manager = PluginManager(registry, postprocessors, persistence, tmp_path)

    plugins = manager.load_all()

    assert plugins[0].enabled is False
    assert plugins[0].loaded is False
    assert registry.adapter_ids() == set()


def test_postprocessor_plugin_controls_are_added_to_adapter_metadata(tmp_path) -> None:
    _write_postprocessor_plugin(
        tmp_path,
        plugin_id="sample-detailer",
        processor_id="sample-detailer-processor",
    )
    registry = create_default_registry()
    postprocessors = GenerationPostprocessorRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(registry, postprocessors, persistence, tmp_path)

    plugins = manager.load_all()
    adapters = manager.list_adapters()

    assert plugins[0].postprocessor_ids == ["sample-detailer-processor"]
    assert any(
        control.id == "sample_postprocessor_enabled" and control.plugin_id == "sample-detailer"
        for control in adapters[0].generation_controls
    )
    assert adapters[0].generation_defaults["sample_postprocessor_enabled"] is False
    assert any(
        processor.id == "sample-detailer-processor" and processor.plugin_id == "sample-detailer"
        for processor in adapters[0].postprocessors
    )


def test_action_plugin_controls_are_registered_and_executable(tmp_path) -> None:
    _write_action_plugin(
        tmp_path,
        plugin_id="sample-action-plugin",
        action_id="sample-action",
    )
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(registry, postprocessors, persistence, tmp_path, actions, tools)

    plugins = manager.load_all()
    action_infos = manager.list_actions()
    tool_infos = manager.list_tools()
    result = manager.run_action(
        "sample-action",
        PluginActionRunRequest(
            image=_data_url((10, 20, 30, 255)),
            controls={"sample_action_suffix": "ready"},
        ),
    )

    assert plugins[0].action_ids == ["sample-action"]
    assert plugins[0].tool_ids == ["sample-action"]
    assert action_infos[0].id == "sample-action"
    assert action_infos[0].plugin_id == "sample-action-plugin"
    assert action_infos[0].controls[0].plugin_id == "sample-action-plugin"
    assert action_infos[0].default_values["sample_action_suffix"] == "default"
    assert tool_infos[0].id == "sample-action"
    assert tool_infos[0].action_id == "sample-action"
    assert tool_infos[0].plugin_id == "sample-action-plugin"
    assert tool_infos[0].controls[0].plugin_id == "sample-action-plugin"
    assert result.text == "96x96 ready"


def test_plugin_manager_unregisters_actions_on_disable(tmp_path) -> None:
    _write_action_plugin(
        tmp_path,
        plugin_id="sample-action-plugin",
        action_id="sample-action",
    )
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(registry, postprocessors, persistence, tmp_path, actions, tools)
    manager.load_all()

    disabled = manager.disable("sample-action-plugin")

    assert disabled.enabled is False
    assert disabled.action_ids == []
    assert disabled.tool_ids == []
    assert manager.list_actions() == []
    assert manager.list_tools() == []


def test_included_auto_detailer_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)

    auto_detailer = next(plugin for plugin in plugins if plugin.id == "auto-detailer")
    assert auto_detailer.loaded is True
    assert auto_detailer.postprocessor_ids == ["auto-detailer"]


def test_included_gfpgan_face_restore_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)

    face_restore = next(plugin for plugin in plugins if plugin.id == "gfpgan-face-restore")
    assert face_restore.loaded is True
    assert face_restore.postprocessor_ids == ["gfpgan-face-restore"]


def test_included_art_face_repair_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)

    face_repair = next(plugin for plugin in plugins if plugin.id == "art-face-repair")
    module = sys.modules["expandiffusion_plugin_art_face_repair"]
    assert face_repair.loaded is True
    assert face_repair.postprocessor_ids == ["art-face-repair"]
    assert "painted face" in module.DETECTION_QUERIES
    assert "profile face" in module.DETECTION_QUERIES
    assert "side profile face" in module.DETECTION_QUERIES


def test_included_image_to_text_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()

    plugins = load_local_plugins(
        registry,
        constants.DEFAULT_PLUGIN_DIR,
        postprocessors,
        actions,
        tools,
    )

    image_to_text = next(plugin for plugin in plugins if plugin.id == "image-to-text")
    assert image_to_text.loaded is True
    assert image_to_text.action_ids == ["image-to-text"]
    assert image_to_text.tool_ids == ["image-to-text"]
    assert any(action.id == "image-to-text" for action in actions.action_infos())
    assert any(
        tool.id == "image-to-text"
        and tool.action_id == "image-to-text"
        and tool.icon == "captions"
        and tool.icon_color == "#4f46e5"
        and tool.accent_color == "#4f46e5"
        and tool.result_label == "Image description"
        and tool.controls == []
        for tool in tools.tool_infos()
    )


def test_plugin_action_result_can_return_mask() -> None:
    result = PluginActionResult(
        action_id="sample-action",
        mask=_mask_data_url((8, 8), (2, 2, 6, 6)),
    )

    assert decode_data_url(result.mask).convert("L").getpixel((3, 3)) == 255


def test_included_object_selector_plugin_loads(tmp_path) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(
        registry,
        postprocessors,
        persistence,
        constants.DEFAULT_PLUGIN_DIR,
        actions,
        tools,
    )

    plugins = manager.load_all()

    object_selector = next(plugin for plugin in plugins if plugin.id == "object-selector")
    tool = next(item for item in tools.tool_infos() if item.id == "object-selector")
    assert object_selector.loaded is True
    assert object_selector.action_ids == ["object-selector"]
    assert object_selector.tool_ids == ["object-selector"]
    assert tool.action_id == "object-selector"
    assert tool.icon == "wand-sparkles"
    assert tool.target == constants.PLUGIN_TOOL_TARGET_CANVAS


def test_object_selector_prompt_click_returns_visible_mask(tmp_path, monkeypatch) -> None:
    manager = _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]

    monkeypatch.setattr(
        module,
        "_detect_prompt_boxes",
        lambda _image, prompt: [
            {"box": (0, 1, 3, 6), "label": prompt, "confidence": 0.4},
            {"box": (4, 1, 8, 6), "label": prompt, "confidence": 0.9},
        ],
    )

    records = {}

    def fake_segment_mask(image, box, points):
        records["box"] = box
        records["points"] = points
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box)
        return mask, 0.82

    monkeypatch.setattr(module, "_segment_mask", fake_segment_mask)

    result = manager.run_action(
        "object-selector",
        PluginActionRunRequest(
            image=_solid_data_url((8, 8), (10, 20, 30, 255)),
            controls={"object_selector_prompt": "cup"},
            target={
                "kind": "canvas",
                "bounds": {"x": 10, "y": 20, "width": 8, "height": 8},
                "scale": 1,
                "point": {"x": 6, "y": 4},
                "visible_mask": _mask_data_url((8, 8), (5, 0, 8, 8)),
            },
        ),
    )

    mask = decode_data_url(result.mask).convert("L")
    assert records == {"box": (4, 1, 8, 6), "points": [(6, 4)]}
    assert mask.size == (8, 8)
    assert mask.getpixel((4, 3)) == 0
    assert mask.getpixel((6, 3)) == 255
    assert result.data["boxes"][0]["label"] == "cup"
    assert result.data["source"] == "prompt_and_click"


def test_object_selector_prompt_only_selects_all_detected_boxes(tmp_path, monkeypatch) -> None:
    manager = _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]

    monkeypatch.setattr(
        module,
        "_detect_prompt_boxes",
        lambda _image, prompt: [
            {"box": (0, 1, 3, 6), "label": prompt, "confidence": 0.4},
            {"box": (4, 1, 8, 6), "label": prompt, "confidence": 0.9},
        ],
    )

    records = []

    def fake_segment_mask(image, box, points):
        records.append({"box": box, "points": points})
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box)
        return mask, 0.82

    monkeypatch.setattr(module, "_segment_mask", fake_segment_mask)

    result = manager.run_action(
        "object-selector",
        PluginActionRunRequest(
            image=_solid_data_url((8, 8), (10, 20, 30, 255)),
            controls={"object_selector_prompt": "umbrella"},
            target={
                "kind": "canvas",
                "bounds": {"x": 10, "y": 20, "width": 8, "height": 8},
                "scale": 1,
                "visible_mask": _mask_data_url((8, 8), (0, 0, 8, 8)),
            },
        ),
    )

    mask = decode_data_url(result.mask).convert("L")
    assert records == [
        {"box": (0, 1, 3, 6), "points": []},
        {"box": (4, 1, 8, 6), "points": []},
    ]
    assert mask.getpixel((1, 3)) == 255
    assert mask.getpixel((6, 3)) == 255
    assert len(result.data["selections"]) == 2
    assert result.data["selections"][0]["id"] == "object-1"
    assert result.data["selections"][1]["id"] == "object-2"
    assert result.data["source"] == "prompt"


def test_object_selector_click_only_uses_point_prompt(tmp_path, monkeypatch) -> None:
    manager = _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]
    records = {}

    def fake_segment_mask(image, box, points):
        records["box"] = box
        records["points"] = points
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 5, 5))
        return mask, 0.71

    monkeypatch.setattr(module, "_segment_mask", fake_segment_mask)

    result = manager.run_action(
        "object-selector",
        PluginActionRunRequest(
            image=_solid_data_url((8, 8), (10, 20, 30, 255)),
            target={
                "kind": "canvas",
                "bounds": {"x": 0, "y": 0, "width": 8, "height": 8},
                "scale": 1,
                "point": {"x": 3, "y": 4},
                "visible_mask": _mask_data_url((8, 8), (0, 0, 8, 8)),
            },
        ),
    )

    assert records == {"box": None, "points": [(3, 4)]}
    assert decode_data_url(result.mask).convert("L").getpixel((3, 3)) == 255
    assert result.data["source"] == "click"


def test_object_selector_uses_multiple_canvas_points(tmp_path, monkeypatch) -> None:
    manager = _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]
    records = {}

    def fake_segment_mask(image, box, points):
        records["box"] = box
        records["points"] = points
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (2, 2, 6, 6))
        return mask, 0.77

    monkeypatch.setattr(module, "_segment_mask", fake_segment_mask)

    result = manager.run_action(
        "object-selector",
        PluginActionRunRequest(
            image=_solid_data_url((8, 8), (10, 20, 30, 255)),
            target={
                "kind": "canvas",
                "bounds": {"x": 0, "y": 0, "width": 8, "height": 8},
                "scale": 1,
                "points": [{"x": 3, "y": 4}, {"x": 5, "y": 6}],
                "visible_mask": _mask_data_url((8, 8), (0, 0, 8, 8)),
            },
        ),
    )

    assert records == {"box": None, "points": [(3, 4), (5, 6)]}
    assert decode_data_url(result.mask).convert("L").getpixel((4, 4)) == 255


def test_object_selector_sam_prompt_shape(tmp_path, monkeypatch) -> None:
    _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]
    torch = pytest.importorskip("torch")
    captured = {}

    class FakeImageProcessor:
        def post_process_masks(self, pred_masks, _original_sizes, _reshaped_input_sizes):
            return [pred_masks[0]]

    class FakeProcessor:
        image_processor = FakeImageProcessor()

        def __call__(self, _image, **kwargs):
            captured.update(kwargs)
            return {
                "original_sizes": torch.tensor([[8, 8]]),
                "reshaped_input_sizes": torch.tensor([[8, 8]]),
            }

    class FakeOutputs:
        pred_masks = torch.ones((1, 1, 1, 8, 8), dtype=torch.bool)
        iou_scores = torch.tensor([[[0.83]]])

    class FakeModel:
        def __call__(self, **_inputs):
            return FakeOutputs()

    monkeypatch.setattr(module, "_load_sam", lambda: (FakeProcessor(), FakeModel()))

    mask, confidence = module._segment_mask(
        Image.new("RGB", (8, 8), (10, 20, 30)),
        (1, 2, 6, 7),
        [(3, 4), (5, 6)],
    )

    assert captured["input_boxes"] == [[[1, 2, 6, 7]]]
    assert captured["input_points"] == [[[[3, 4], [5, 6]]]]
    assert captured["input_labels"] == [[[1, 1]]]
    assert mask.getbbox() == (0, 0, 8, 8)
    assert confidence == pytest.approx(0.83)


def test_object_selector_reports_missing_prompt_detection(tmp_path, monkeypatch) -> None:
    manager = _object_selector_manager(tmp_path)
    module = sys.modules["expandiffusion_plugin_object_selector"]
    monkeypatch.setattr(module, "_detect_prompt_boxes", lambda _image, _prompt: [])

    with pytest.raises(AppError, match="did not detect"):
        manager.run_action(
            "object-selector",
            PluginActionRunRequest(
                image=_solid_data_url((8, 8), (10, 20, 30, 255)),
                controls={"object_selector_prompt": "missing object"},
                target={
                    "kind": "canvas",
                    "bounds": {"x": 0, "y": 0, "width": 8, "height": 8},
                    "scale": 1,
                    "visible_mask": _mask_data_url((8, 8), (0, 0, 8, 8)),
                },
            ),
        )


def _object_selector_manager(tmp_path) -> PluginManager:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(
        registry,
        postprocessors,
        persistence,
        constants.DEFAULT_PLUGIN_DIR,
        actions,
        tools,
    )
    manager.load_all()
    return manager


def test_included_sdxl_ip_visual_refine_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)

    visual_refine = next(plugin for plugin in plugins if plugin.id == "sdxl-ip-visual-refine")
    adapter = registry.get("sdxl-fill-ip-refine")
    assert visual_refine.loaded is True
    assert visual_refine.adapter_ids == ["sdxl-fill-ip-refine"]
    assert adapter.capabilities.ip_adapter is True
    assert adapter.capabilities.outpaint is True


def test_sdxl_ip_visual_refine_controls_do_not_touch_base_adapter() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)

    base_adapter = SdxlFillControlNetUnionAdapter()
    visual_refine_adapter = registry.get("sdxl-fill-ip-refine")
    base_controls = {control.id for control in base_adapter.generation_controls()}
    base_defaults = base_adapter.generation_defaults()
    refine_controls = {control.id for control in visual_refine_adapter.generation_controls()}
    refine_defaults = visual_refine_adapter.generation_defaults()

    assert "visual_refine_enabled" not in base_controls
    assert "visual_refine_enabled" not in base_defaults
    assert "visual_refine_enabled" in refine_controls
    assert "negative_prompt" in refine_controls
    assert "inpaint_strength" in refine_controls
    assert "inpaint_area" in refine_controls
    assert "mask_crop_padding" in refine_controls
    assert "fill_mode" in refine_controls
    assert "result_mode" in refine_controls
    assert "conditioning_type" in refine_controls
    assert "img2img" in refine_controls
    assert "control_guidance_start" in refine_controls
    assert "control_guidance_end" in refine_controls
    assert refine_defaults["visual_refine_enabled"] is False
    assert refine_defaults["visual_refine_strength"] == 0.45
    assert refine_defaults["ip_adapter_scale"] == 0.45
    assert refine_defaults["negative_prompt"] == ""
    assert refine_defaults["inpaint_strength"] == 0.65
    assert refine_defaults["inpaint_area"] == constants.INPAINT_AREA_WHOLE_SELECTION
    assert refine_defaults["mask_crop_padding"] == constants.DEFAULT_MASK_CROP_PADDING
    assert refine_defaults["fill_mode"] == constants.FILL_TRANSPARENT
    assert refine_defaults["result_mode"] == constants.RESULT_MODE_PRESERVE_KNOWN
    assert refine_defaults["conditioning_type"] == constants.CONDITIONING_TYPE_COLOR
    assert refine_defaults["img2img"] is False
    assert refine_defaults["control_guidance_start"] == constants.DEFAULT_CONTROL_GUIDANCE_START
    assert refine_defaults["control_guidance_end"] == constants.DEFAULT_CONTROL_GUIDANCE_END


def test_sdxl_ip_visual_refine_can_skip_refine(monkeypatch) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    adapter = registry.get("sdxl-fill-ip-refine")
    base_image = Image.new("RGB", (32, 16), (11, 22, 33))
    records = {}

    def fake_base_generate(_adapter, _context):
        records["steps"] = _context.parameters.steps
        records["controlnet_conditioning_scale"] = (
            _context.parameters.controlnet_conditioning_scale
        )
        records["source_size"] = _context.source.size
        records["mask_size"] = _context.mask.size
        return [base_image]

    monkeypatch.setattr(SdxlFillControlNetUnionAdapter, "generate", fake_base_generate)
    context = GenerationContext(
        source=Image.new("RGB", (32, 16), (100, 90, 80)),
        mask=_half_generation_mask(32, 16, 16),
        parameters=GenerationParameters(
            steps=6,
            controlnet_conditioning_scale=0.85,
            visual_refine_enabled=False,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert images == [base_image]
    assert records == {
        "steps": 6,
        "controlnet_conditioning_scale": 0.85,
        "source_size": (32, 16),
        "mask_size": (32, 16),
    }


def test_sdxl_ip_visual_refine_inpaint_uses_conservative_masked_branch(
    monkeypatch,
) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    adapter = registry.get("sdxl-fill-ip-refine")
    adapter.device = "cpu"
    source = Image.new("RGB", (96, 64), (20, 30, 40))
    source.putpixel((80, 32), (210, 120, 80))
    records = {}

    def fake_base_generate(_adapter, _context):
        raise AssertionError("inpaint must not use the fill/outpaint first pass")

    class FakeInpaintPipeline:
        def load_ip_adapter(self, *_args, **_kwargs):
            raise AssertionError("conservative inpaint must not load IP-Adapter")

        def __call__(self, **kwargs):
            records["image"] = kwargs["image"].copy()
            records["mask"] = kwargs["mask_image"].copy()
            records["control_image"] = kwargs["control_image"].copy()
            records["control_mode"] = kwargs["control_mode"]
            records["controlnet_conditioning_scale"] = kwargs[
                "controlnet_conditioning_scale"
            ]
            records["control_guidance_start"] = kwargs["control_guidance_start"]
            records["control_guidance_end"] = kwargs["control_guidance_end"]
            records["negative_prompt"] = kwargs["negative_prompt"]
            records["padding_mask_crop"] = kwargs.get("padding_mask_crop")
            records["strength"] = kwargs["strength"]
            records["steps"] = kwargs["num_inference_steps"]
            records["guidance_scale"] = kwargs["guidance_scale"]
            records["prompt"] = kwargs["prompt"]
            return type("Output", (), {"images": [Image.new("RGB", (96, 64), (8, 9, 10))]})()

    monkeypatch.setattr(SdxlFillControlNetUnionAdapter, "generate", fake_base_generate)
    monkeypatch.setattr(adapter, "_build_inpaint_pipeline", lambda: FakeInpaintPipeline())
    context = GenerationContext(
        source=source,
        mask=_half_generation_mask(96, 64, 48),
        parameters=GenerationParameters(
            prompt="repair object",
            steps=7,
            guidance_scale=1.5,
            strength=1.0,
            inpaint_strength=0.42,
            negative_prompt="bad lighting",
            inpaint_area=constants.INPAINT_AREA_ONLY_MASKED,
            mask_crop_padding=64,
            controlnet_conditioning_scale=0.83,
            control_guidance_start=0.1,
            control_guidance_end=0.9,
            random_seed=False,
            seed=10,
            sample_count=1,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
        metadata={"generation_mode": constants.GENERATION_MODE_INPAINT},
    )

    images = adapter.generate(context)

    assert images[0].getpixel((0, 0)) == (8, 9, 10)
    assert records["prompt"] == "repair object, same style, same lighting, same color palette"
    assert records["image"].getpixel((80, 32)) == (210, 120, 80)
    assert records["mask"].getpixel((0, 0)) == 0
    assert records["mask"].getpixel((95, 0)) == 255
    assert records["control_image"].getpixel((0, 0)) == (20, 30, 40)
    assert records["control_image"].getpixel((80, 32)) == (0, 0, 0)
    assert records["control_mode"] == 7
    assert records["controlnet_conditioning_scale"] == 0.83
    assert records["control_guidance_start"] == 0.1
    assert records["control_guidance_end"] == 0.9
    assert records["negative_prompt"] == "bad lighting"
    assert records["padding_mask_crop"] == 64
    assert records["strength"] == 0.42
    assert records["steps"] == 7
    assert records["guidance_scale"] == 1.5


def test_sdxl_ip_visual_refine_controlnet_union_shim_maps_repaint_mode() -> None:
    import torch
    from diffusers.configuration_utils import FrozenDict

    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    adapter = registry.get("sdxl-fill-ip-refine")
    plugin_module = sys.modules[adapter.__class__.__module__]
    records = {}

    class WrappedControlNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FrozenDict({"num_control_type": 8})
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(
            self,
            _sample,
            _timestep,
            *,
            encoder_hidden_states,
            controlnet_cond_list,
            conditioning_scale,
            added_cond_kwargs,
            return_dict,
            **_kwargs,
        ):
            records["encoder_hidden_states"] = encoder_hidden_states
            records["controlnet_cond_list"] = controlnet_cond_list
            records["conditioning_scale"] = (torch.ones(()) * conditioning_scale).item()
            records["control_type"] = added_cond_kwargs["control_type"]
            records["return_dict"] = return_dict
            return "down", "mid"

    wrapped = WrappedControlNet()
    shim = plugin_module._controlnet_union_for_inpaint(wrapped)
    condition = torch.ones((1, 3, 8, 8))
    control_type = torch.zeros((2, 8))
    control_type[:, 7] = 1
    result = shim(
        torch.zeros((2, 4, 8, 8)),
        torch.tensor(1),
        encoder_hidden_states=torch.zeros((2, 1, 4)),
        controlnet_cond=[condition],
        control_type=control_type,
        control_type_idx=[7],
        conditioning_scale=[0.83],
        added_cond_kwargs={"text_embeds": torch.zeros((2, 4))},
        return_dict=False,
    )

    assert result == ("down", "mid")
    assert shim.wrapped_controlnet is wrapped
    assert len(records["controlnet_cond_list"]) == 8
    assert records["controlnet_cond_list"][0].sum().item() == 0
    assert torch.equal(records["controlnet_cond_list"][7], condition)
    assert records["conditioning_scale"] == pytest.approx(0.83)
    assert records["control_type"] is control_type
    assert records["return_dict"] is False


def test_sdxl_ip_visual_refine_uses_generation_mask_and_near_edge_reference(
    monkeypatch,
) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    adapter = registry.get("sdxl-fill-ip-refine")
    adapter.device = "cpu"
    source = Image.new("RGB", (128, 64), (20, 30, 40))
    for x in range(64):
        for y in range(64):
            source.putpixel((x, y), (200, 120, 80))
    first_pass = Image.new("RGB", (128, 64), (1, 2, 3))
    records = {}

    def fake_base_generate(_adapter, _context):
        return [first_pass]

    class FakeRefinePipeline:
        def load_ip_adapter(self, repo, *, subfolder, weight_name, image_encoder_folder):
            records["ip_adapter"] = {
                "repo": repo,
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": image_encoder_folder,
            }

        def set_ip_adapter_scale(self, scale):
            records["scale"] = scale

        def unload_ip_adapter(self):
            records["unloaded"] = True

        def __call__(self, **kwargs):
            records["mask"] = kwargs["mask_image"].copy()
            records["reference"] = kwargs["ip_adapter_image"].copy()
            records["strength"] = kwargs["strength"]
            records["steps"] = kwargs["num_inference_steps"]
            records["prompt"] = kwargs["prompt"]
            return type("Output", (), {"images": [Image.new("RGB", (128, 64), (9, 9, 9))]})()

    monkeypatch.setattr(SdxlFillControlNetUnionAdapter, "generate", fake_base_generate)
    monkeypatch.setattr(adapter, "_build_refine_pipeline", lambda: FakeRefinePipeline())
    context = GenerationContext(
        source=source,
        mask=_half_generation_mask(128, 64, 64),
        parameters=GenerationParameters(
            prompt="continue the painting",
            visual_refine_enabled=True,
            visual_refine_strength=0.5,
            ip_adapter_scale=0.35,
            visual_refine_steps=7,
            visual_refine_reference="near_edge",
            random_seed=False,
            seed=10,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert images[0].getpixel((0, 0)) == (9, 9, 9)
    assert records["ip_adapter"] == {
        "repo": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "image_encoder_folder": "models/image_encoder",
    }
    assert records["scale"] == 0.35
    assert records["strength"] == 0.5
    assert records["steps"] == 7
    expected_prompt = "continue the painting, same style, same lighting, same color palette"
    assert records["prompt"] == expected_prompt
    assert records["mask"].size == (128, 64)
    assert records["mask"].getpixel((0, 0)) == 0
    assert records["mask"].getpixel((127, 0)) == 255
    assert records["reference"].width < source.width
    assert records["reference"].getpixel((0, 0)) == (200, 120, 80)
    assert records["unloaded"] is True


def test_included_image_adjustments_plugin_loads() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()

    plugins = load_local_plugins(
        registry,
        constants.DEFAULT_PLUGIN_DIR,
        postprocessors,
        actions,
        tools,
    )

    image_adjustments = next(plugin for plugin in plugins if plugin.id == "image-adjustments")
    tool = next(tool for tool in tools.tool_infos() if tool.id == "image-adjustments")
    assert image_adjustments.loaded is True
    assert image_adjustments.action_ids == ["image-adjustments"]
    assert image_adjustments.tool_ids == ["image-adjustments"]
    assert tool.action_id == "image-adjustments"
    assert tool.target == constants.PLUGIN_TOOL_TARGET_IMAGE
    assert tool.icon == "palette"
    assert tool.icon_color == "#f97316"
    assert tool.accent_color == "#f97316"
    assert tool.live_preview is True
    assert {control.id for control in tool.controls} == {
        "image_adjustments_brightness",
        "image_adjustments_contrast",
        "image_adjustments_saturation",
        "image_adjustments_vibrance",
        "image_adjustments_exposure",
        "image_adjustments_gamma",
        "image_adjustments_shadows",
        "image_adjustments_highlights",
        "image_adjustments_hue",
        "image_adjustments_warmth",
        "image_adjustments_tint",
        "image_adjustments_color_match",
        "image_adjustments_sharpness",
    }


def test_image_adjustments_action_returns_adjusted_image(tmp_path) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(
        registry,
        postprocessors,
        persistence,
        constants.DEFAULT_PLUGIN_DIR,
        actions,
        tools,
    )
    manager.load_all()

    result = manager.run_action(
        "image-adjustments",
        PluginActionRunRequest(
            image=_data_url((100, 80, 60, 255)),
            controls={
                "image_adjustments_brightness": 1.4,
                "image_adjustments_contrast": 1.0,
                "image_adjustments_saturation": 1.0,
                "image_adjustments_hue": 0,
                "image_adjustments_warmth": 0,
                "image_adjustments_sharpness": 1.0,
            },
            target={"kind": "raster"},
        ),
    )

    assert result.image is not None
    output = decode_data_url(result.image).convert("RGBA")
    assert output.size == (96, 96)
    assert output.getpixel((0, 0))[0] > 100
    assert result.data["target"] == {"kind": "raster"}


def test_image_adjustments_action_can_auto_match_color_balance(tmp_path) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(
        registry,
        postprocessors,
        persistence,
        constants.DEFAULT_PLUGIN_DIR,
        actions,
        tools,
    )
    manager.load_all()

    result = manager.run_action(
        "image-adjustments",
        PluginActionRunRequest(
            image=_data_url((210, 72, 72, 255)),
            controls={"image_adjustments_color_match": 100},
            target={"kind": "raster"},
        ),
    )

    assert result.image is not None
    red, green, blue, _alpha = decode_data_url(result.image).convert("RGBA").getpixel((0, 0))
    assert abs(red - green) <= 1
    assert abs(red - blue) <= 1


def test_image_to_text_action_runs_clip_interrogator(monkeypatch, tmp_path) -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()
    actions = PluginActionRegistry()
    tools = PluginToolRegistry()
    persistence = PersistenceStore(tmp_path / "app_state.json")
    manager = PluginManager(
        registry,
        postprocessors,
        persistence,
        constants.DEFAULT_PLUGIN_DIR,
        actions,
        tools,
    )
    manager.load_all()
    module = sys.modules["expandiffusion_plugin_image_to_text"]

    class FakeInterrogator:
        device = "cpu"

        def interrogate(self, image, min_flavors, max_flavors):
            assert image.size == (96, 96)
            assert min_flavors == 8
            assert max_flavors == 32
            return "compact red square, studio lighting, sharp focus"

    monkeypatch.setattr(
        module,
        "_load_interrogator",
        lambda: FakeInterrogator(),
    )

    result = manager.run_action(
        "image-to-text",
        PluginActionRunRequest(
            image=_data_url((180, 10, 20, 255)),
        ),
    )

    assert result.text == "compact red square, studio lighting, sharp focus"
    assert result.data["engine"] == "clip-interrogator"
    assert result.data["caption_model_name"] == "blip-large"
    assert result.data["clip_model_name"] == "ViT-L-14/openai"


def test_included_correction_plugins_load_as_corrections() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    correction_plugins = {
        "correction-mask-feather",
        "correction-border-blend",
        "correction-color-match",
        "correction-histogram-match",
        "correction-multiband-blend",
    }

    for plugin_id in correction_plugins:
        plugin = next(item for item in plugins if item.id == plugin_id)
        assert plugin.loaded is True
        assert plugin.postprocessor_ids == [plugin_id]
    for processor in postprocessors.correction_pipeline(sorted(correction_plugins)):
        assert processor.category == constants.POSTPROCESSOR_CATEGORY_CORRECTION


def test_included_seam_refine_plugin_loads_as_result_refine() -> None:
    registry = AdapterRegistry()
    postprocessors = GenerationPostprocessorRegistry()

    plugins = load_local_plugins(registry, constants.DEFAULT_PLUGIN_DIR, postprocessors)
    plugin = next(item for item in plugins if item.id == "sdxl-seam-refine")
    processor = next(item for item in postprocessors.processors() if item.id == "sdxl-seam-refine")
    controls = {control.id: control for control in processor.generation_controls()}

    assert plugin.loaded is True
    assert plugin.postprocessor_ids == ["sdxl-seam-refine"]
    assert processor.category == "result_refine"
    assert controls["seam_refine_enabled"].default_value is False
    assert processor.generation_defaults()["seam_refine_enabled"] is False


def test_seam_refine_pads_non_multiple_of_eight_size_for_sdxl(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    module = sys.modules["expandiffusion_plugin_sdxl_seam_refine"]
    processor = next(item for item in postprocessors.processors() if item.id == "sdxl-seam-refine")
    records = {}

    class FakePipeline:
        def load_ip_adapter(self, *_args, **_kwargs) -> None:
            records["loaded"] = True

        def set_ip_adapter_scale(self, scale: float) -> None:
            records["scale"] = scale

        def enable_model_cpu_offload(self, *_args, **_kwargs) -> None:
            records["offload"] = True

        def unload_ip_adapter(self) -> None:
            records["unloaded"] = True

        def __call__(self, **kwargs):
            records["width"] = kwargs["width"]
            records["height"] = kwargs["height"]
            records["image_size"] = kwargs["image"].size
            records["mask_size"] = kwargs["mask_image"].size
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (90, 80, 70))
            return type("Output", (), {"images": [image]})()

    monkeypatch.setattr(module, "_build_pipeline", lambda _torch, _adapter: FakePipeline())
    generated = Image.new("RGB", (1008, 962), (10, 20, 30))
    original = Image.new("RGBA", (1008, 962), (10, 20, 30, 255))
    mask = Image.new("L", (1008, 962), 0)
    for x in range(500, 1008):
        for y in range(962):
            mask.putpixel((x, y), 255)

    result = processor.process(
        GenerationPostprocessorContext(
            original=original,
            generated=generated,
            mask=mask,
            parameters=GenerationParameters(
                prompt="",
                seam_refine_enabled=True,
                seam_refine_width=24,
                seam_refine_strength=0.18,
                seam_refine_ip_adapter_scale=0.35,
                seam_refine_steps=6,
                seam_refine_reference="near_edge",
            ),
            adapter=Sd15InpaintAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
            metadata={},
        )
    )

    assert records["width"] == 1008
    assert records["height"] == 968
    assert records["image_size"] == (1008, 968)
    assert records["mask_size"] == (1008, 968)
    assert result.size == (1008, 962)
    assert records["unloaded"] is True


def test_seam_refine_mask_follows_irregular_mask_not_bounding_box() -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    module = sys.modules["expandiffusion_plugin_sdxl_seam_refine"]
    original = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    mask = Image.new("L", (64, 64), 0)
    for x in range(8, 16):
        for y in range(24, 40):
            mask.putpixel((x, y), 255)
    for x in range(48, 56):
        for y in range(24, 40):
            mask.putpixel((x, y), 255)

    seam_mask = module._build_seam_mask(original, mask, (64, 64), 4)

    assert seam_mask.getbbox() is not None
    assert seam_mask.getpixel((8, 32)) > 0
    assert seam_mask.getpixel((55, 32)) > 0
    assert seam_mask.getpixel((32, 32)) == 0


def test_seam_refine_raises_internal_steps_for_low_strength(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    module = sys.modules["expandiffusion_plugin_sdxl_seam_refine"]
    processor = next(item for item in postprocessors.processors() if item.id == "sdxl-seam-refine")
    records = {}

    class FakePipeline:
        def load_ip_adapter(self, *_args, **_kwargs) -> None:
            pass

        def set_ip_adapter_scale(self, _scale: float) -> None:
            pass

        def unload_ip_adapter(self) -> None:
            pass

        def enable_model_cpu_offload(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, **kwargs):
            records["steps"] = kwargs["num_inference_steps"]
            return type(
                "Output",
                (),
                {"images": [Image.new("RGB", kwargs["image"].size, (90, 80, 70))]},
            )()

    monkeypatch.setattr(module, "_build_pipeline", lambda _torch, _adapter: FakePipeline())
    original = Image.new("RGBA", (64, 62), (10, 20, 30, 255))
    generated = Image.new("RGB", (64, 62), (10, 20, 30))
    mask = Image.new("L", (64, 62), 0)
    for x in range(32, 64):
        for y in range(62):
            mask.putpixel((x, y), 255)

    result = processor.process(
        GenerationPostprocessorContext(
            original=original,
            generated=generated,
            mask=mask,
            parameters=GenerationParameters(
                seam_refine_enabled=True,
                seam_refine_strength=0.05,
                seam_refine_steps=2,
            ),
            adapter=Sd15InpaintAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
        )
    )

    assert records["steps"] == 20
    assert result.size == (64, 62)


def test_auto_detailer_skips_when_enabled_but_no_regions_are_detected() -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    processor = next(item for item in postprocessors.processors() if item.id == "auto-detailer")
    adapter = Sd15InpaintAdapter()
    generated = Image.new("RGB", (96, 96), (32, 32, 36))
    diagnostics = []

    result = processor.process(
        GenerationPostprocessorContext(
            original=Image.new("RGBA", (96, 96), (0, 0, 0, 0)),
            generated=generated,
            mask=Image.new("L", (96, 96), 255),
            parameters=GenerationParameters(
                auto_detailer_enabled=True,
                auto_detailer_targets="bodies",
            ),
            adapter=adapter,
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
            diagnostics=diagnostics,
        )
    )

    assert result.tobytes() == generated.tobytes()
    assert diagnostics[0]["status"] == "skipped"
    assert "did not detect" in diagnostics[0]["message"]


def test_gfpgan_face_restore_skips_when_no_faces_are_detected(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    processor = next(
        item for item in postprocessors.processors() if item.id == "gfpgan-face-restore"
    )
    module = sys.modules["expandiffusion_plugin_gfpgan_face_restore"]
    generated = Image.new("RGB", (96, 96), (32, 32, 36))
    diagnostics = []

    class RestorerWithoutFaces:
        def enhance(self, input_bgr, **_kwargs):
            return [], [], input_bgr

    monkeypatch.setattr(
        module.GFPGANFaceRestorePostprocessor,
        "_restorer",
        lambda _self, _settings, _context: RestorerWithoutFaces(),
    )

    result = processor.process(
        GenerationPostprocessorContext(
            original=Image.new("RGBA", (96, 96), (0, 0, 0, 0)),
            generated=generated,
            mask=Image.new("L", (96, 96), 255),
            parameters=GenerationParameters(gfpgan_face_restore_enabled=True),
            adapter=Sd15InpaintAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
            diagnostics=diagnostics,
        )
    )

    assert result.tobytes() == generated.tobytes()
    assert diagnostics[0]["status"] == "skipped"
    assert "did not detect" in diagnostics[0]["message"]


def test_auto_detailer_rejects_destructive_detail_pass(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    processor = next(item for item in postprocessors.processors() if item.id == "auto-detailer")
    module = sys.modules["expandiffusion_plugin_auto_detailer"]
    diagnostics = []

    class DestructiveAdapter:
        def generate(self, context: GenerationContext) -> list[Image.Image]:
            return [Image.new("RGB", context.source.size, (0, 255, 255))]

    monkeypatch.setattr(
        module,
        "_detect_regions",
        lambda _image, _mask, _targets, _min_size: [
            module.DetectedRegion((16, 16, 48, 48), "face")
        ],
    )

    with pytest.raises(AppError) as error:
        processor.process(
            GenerationPostprocessorContext(
                original=Image.new("RGBA", (96, 96), (0, 0, 0, 0)),
                generated=Image.new("RGB", (96, 96), (32, 32, 36)),
                mask=Image.new("L", (96, 96), 255),
                parameters=GenerationParameters(
                    auto_detailer_enabled=True,
                    auto_detailer_targets="faces",
                    auto_detailer_strength=0.95,
                    auto_detailer_padding=0,
                    auto_detailer_mask_blur=0,
                ),
                adapter=DestructiveAdapter(),
                progress=lambda _value, _message: None,
                is_cancelled=lambda: False,
                diagnostics=diagnostics,
            )
        )

    assert "rejected a destructive detail pass" in error.value.message
    assert diagnostics[0]["status"] == "failed"
    assert diagnostics[0]["region"]["box"] == [16, 16, 48, 48]


def test_art_face_repair_upscales_expanded_crop(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    processor = next(item for item in postprocessors.processors() if item.id == "art-face-repair")
    module = sys.modules["expandiffusion_plugin_art_face_repair"]
    calls = []
    diagnostics = []

    class DetailAdapter:
        def generate(self, context: GenerationContext) -> list[Image.Image]:
            calls.append(context)
            return [Image.new("RGB", context.source.size, (210, 176, 142))]

    monkeypatch.setattr(
        module,
        "_detect_art_face_regions",
        lambda _image, _mask, _min_area: [
            module.ArtFaceRegion((58, 48, 100, 96), "head")
        ],
    )

    result = processor.process(
        GenerationPostprocessorContext(
            original=Image.new("RGBA", (160, 160), (0, 0, 0, 0)),
            generated=Image.new("RGB", (160, 160), (164, 130, 96)),
            mask=Image.new("L", (160, 160), 255),
            parameters=GenerationParameters(
                art_face_repair_enabled=True,
                art_face_repair_strength=0.62,
                art_face_repair_steps=8,
                art_face_repair_mask_blur=0,
            ),
            adapter=DetailAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
            diagnostics=diagnostics,
        )
    )

    assert result.tobytes() != Image.new("RGB", (160, 160), (164, 130, 96)).tobytes()
    assert max(calls[0].source.size) == 512
    assert calls[0].parameters.art_face_repair_enabled is False
    assert diagnostics[0]["status"] == "applied"
    assert diagnostics[0]["regions"][0]["status"] == "applied"


def test_art_face_repair_rejects_dark_halo(monkeypatch) -> None:
    postprocessors = GenerationPostprocessorRegistry()
    load_local_plugins(AdapterRegistry(), constants.DEFAULT_PLUGIN_DIR, postprocessors)
    processor = next(item for item in postprocessors.processors() if item.id == "art-face-repair")
    module = sys.modules["expandiffusion_plugin_art_face_repair"]
    generated = Image.new("RGB", (160, 160), (210, 190, 160))
    diagnostics = []

    class DarkHaloAdapter:
        def generate(self, context: GenerationContext) -> list[Image.Image]:
            return [Image.new("RGB", context.source.size, (4, 4, 4))]

    monkeypatch.setattr(
        module,
        "_detect_art_face_regions",
        lambda _image, _mask, _min_area: [
            module.ArtFaceRegion((58, 48, 100, 96), "head")
        ],
    )

    result = processor.process(
        GenerationPostprocessorContext(
            original=Image.new("RGBA", (160, 160), (0, 0, 0, 0)),
            generated=generated,
            mask=Image.new("L", (160, 160), 255),
            parameters=GenerationParameters(
                art_face_repair_enabled=True,
                art_face_repair_mask_blur=0,
            ),
            adapter=DarkHaloAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
            diagnostics=diagnostics,
        )
    )

    assert result.tobytes() == generated.tobytes()
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["regions"][0]["status"] == "skipped_dark_halo"


def test_runtime_reports_real_pytorch_state() -> None:
    response = client.get("/api/runtime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["torch_version"]
    assert payload["preferred_device"]
    assert payload["preferred_dtype"] in {"float16", "float32"}
    assert "CPU" in payload["note"] or "cuda" in payload["preferred_device"]


def test_persistent_state_endpoint_returns_json() -> None:
    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == constants.PERSISTENCE_VERSION
    assert isinstance(payload["model_loads"], list)
    assert isinstance(payload["projects"], list)
    assert isinstance(payload["generations"], list)


def test_model_load_progress_endpoint_returns_current_state() -> None:
    response = client.get("/api/models/load/progress")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "progress" in payload
    assert "message" in payload
    assert "file_bytes_done" in payload
    assert "bytes_done" in payload


def test_persistence_store_records_model_and_generation(tmp_path) -> None:
    store = PersistenceStore(tmp_path / "app_state.json")
    model_request = ModelLoadRequest(
        adapter_id=constants.ADAPTER_SD15_INPAINT,
        model_id=constants.MODEL_SD15_INPAINT,
        model_url="https://example.test/model.safetensors",
        device="cuda:0",
        dtype="float16",
        safety_checker=False,
    )
    model = ModelInfo(
        adapter_id=constants.ADAPTER_SD15_INPAINT,
        adapter_label="Stable Diffusion 1.5 Inpaint",
        model_id=constants.MODEL_SD15_INPAINT,
        local_path=None,
        single_file_path=None,
        model_url=None,
        device="cuda:0",
        dtype="float16",
        loaded=True,
    )
    store.record_model_loaded(model, model_request)

    job_store = JobStore(store)
    job_store.create(
        OutpaintRequest(
            adapter_id=constants.ADAPTER_SD15_INPAINT,
            image=_data_url((32, 32, 36, 255)),
            mask=_data_url((255, 255, 255, 255)),
            parameters=GenerationParameters(
                prompt="extend the garden",
                width=96,
                height=96,
                sample_count=2,
            ),
            project_id="project-a",
            metadata={},
        )
    )

    state = store.get_state()
    assert store.path.exists()
    assert state.current_model is not None
    assert state.current_model.safety_checker is False
    assert state.current_model.model_url == "https://example.test/model.safetensors"
    assert state.model_loads[0].adapter_id == constants.ADAPTER_SD15_INPAINT
    assert state.projects[0].project_id == "project-a"
    assert state.projects[0].generation_count == 1
    assert state.generations[0].prompt == "extend the garden"
    assert state.generations[0].sample_count == 2


def test_model_storage_downloads_direct_url(tmp_path, monkeypatch) -> None:
    model_bytes = b"m" * (DOWNLOAD_CHUNK_SIZE + 7)

    class FakeResponse(BytesIO):
        headers = {
            "Content-Disposition": 'attachment; filename="downloaded.safetensors"',
            "Content-Length": str(len(model_bytes)),
        }

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self.close()

    def fake_urlopen(_url, timeout):
        assert timeout == 60
        return FakeResponse(model_bytes)

    monkeypatch.setattr("expandiffusion.model_storage.urlopen", fake_urlopen)

    storage = ModelStorage(tmp_path)
    progress_events = []
    target = storage.resolve_url(
        "https://example.test/models/model.safetensors?token=secret",
        progress=lambda done, total, filename: progress_events.append((done, total, filename)),
    )

    assert target.parent == tmp_path
    assert target.name.startswith("downloaded-")
    assert target.suffix == ".safetensors"
    assert target.read_bytes() == model_bytes
    assert progress_events[0] == (0, len(model_bytes), target.name)
    assert progress_events[1][0] == DOWNLOAD_CHUNK_SIZE
    assert progress_events[-1] == (len(model_bytes), len(model_bytes), target.name)


def test_model_storage_downloads_civitai_model_page_url(tmp_path, monkeypatch) -> None:
    model_bytes = b"model"
    opened_urls = []

    class FakeResponse(BytesIO):
        headers = {
            "Content-Disposition": 'attachment; filename="cyberrealistic_final.safetensors"',
            "Content-Length": str(len(model_bytes)),
        }

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self.close()

    def fake_urlopen(url, timeout):
        opened_urls.append(url)
        assert timeout == 60
        return FakeResponse(model_bytes)

    monkeypatch.setattr("expandiffusion.model_storage.urlopen", fake_urlopen)

    storage = ModelStorage(tmp_path)
    target = storage.resolve_url(
        "https://civitai.com/models/15003/cyberrealistic?modelVersionId=2681234"
    )

    assert opened_urls == ["https://civitai.com/api/download/models/2681234"]
    assert target.name.startswith("cyberrealistic_final-")
    assert target.suffix == ".safetensors"
    assert target.read_bytes() == model_bytes


def test_model_storage_download_can_be_cancelled(tmp_path, monkeypatch) -> None:
    model_bytes = b"m" * (DOWNLOAD_CHUNK_SIZE + 7)

    class FakeResponse(BytesIO):
        headers = {
            "Content-Disposition": 'attachment; filename="downloaded.safetensors"',
            "Content-Length": str(len(model_bytes)),
        }

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self.close()

    monkeypatch.setattr(
        "expandiffusion.model_storage.urlopen",
        lambda _url, timeout: FakeResponse(model_bytes),
    )

    storage = ModelStorage(tmp_path)
    progress_events = []

    with pytest.raises(AppError) as error:
        storage.resolve_url(
            "https://example.test/models/model.safetensors",
            progress=lambda done, total, filename: progress_events.append((done, total, filename)),
            is_cancelled=lambda: len(progress_events) > 1,
        )

    assert error.value.code == constants.ERROR_MODEL_LOAD_CANCELLED
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.safetensors"))


def test_model_load_can_be_cancelled(tmp_path) -> None:
    started = threading.Event()
    errors = []

    class CancelableAdapter(ModelAdapter):
        id = "cancelable-adapter"
        label = "Cancelable Adapter"
        family = "test"
        description = "Test adapter."
        default_model_id = None
        capabilities = AdapterCapabilities(inpaint=True, outpaint=True)

        def load(self, config, progress=None, is_cancelled=None):
            started.set()
            if progress:
                progress(0.2, "Fake loading.", None)
            while not is_cancelled():
                time.sleep(0.01)
            raise AppError(
                constants.ERROR_MODEL_LOAD_CANCELLED,
                "Model load cancelled.",
                status_code=409,
            )

        def unload(self):
            self.loaded_config = None

        def generate(self, context):
            return [context.source]

    adapter = CancelableAdapter()
    test_registry = AdapterRegistry()
    test_registry.register(adapter)
    service = ModelService(test_registry, PersistenceStore(tmp_path / "app_state.json"))

    def load_model() -> None:
        try:
            service.load(ModelLoadRequest(adapter_id=adapter.id, local_path=r"E:\models\test"))
        except AppError as exc:
            errors.append(exc)

    thread = threading.Thread(target=load_model)
    thread.start()
    assert started.wait(1)

    progress = service.cancel_load()
    thread.join(2)

    assert progress.status == "cancelling"
    assert not thread.is_alive()
    assert errors[0].code == constants.ERROR_MODEL_LOAD_CANCELLED
    assert service.load_progress().status == "cancelled"
    assert adapter.loaded is False


def test_model_unload_endpoint_unloads_adapter() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post("/api/models/unload", json={"adapter_id": adapter.id})

        assert response.status_code == 200
        assert response.json()["loaded"] is False
        assert adapter.loaded is False
        assert models.loaded_adapter_id is None
    finally:
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


def test_generation_requires_loaded_model() -> None:
    response = client.post(
        "/api/generations/outpaint",
        json={
            "adapter_id": "sd15-inpaint",
            "image": _data_url((32, 32, 36, 255)),
            "mask": _data_url((255, 255, 255, 255)),
            "parameters": {
                "prompt": "clean cyberpunk wall",
                "width": 96,
                "height": 96,
                "steps": 4,
                "sample_count": 2,
                "random_seed": False,
                "seed": 123,
            },
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "MODEL_NOT_LOADED"


def test_alpha_channel_is_the_base_generation_mask() -> None:
    image = Image.new("RGBA", (16, 8), (0, 0, 0, 0))
    for x in range(8):
        for y in range(8):
            image.putpixel((x, y), (180, 190, 200, 255))

    mask = generation_mask_from_alpha(image)

    assert mask.getpixel((2, 4)) == 0
    assert mask.getpixel((12, 4)) == 255


def test_unknown_adapter_returns_structured_error() -> None:
    response = client.post(
        "/api/generations/outpaint",
        json={
            "adapter_id": "missing",
            "image": _data_url((0, 0, 0, 255)),
            "mask": _data_url((255, 255, 255, 255)),
            "parameters": {"prompt": "test"},
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "ADAPTER_NOT_FOUND"


@pytest.mark.integration
def test_outpaint_job_lifecycle_returns_results_through_api_and_websocket() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_one_image(context):
        context.progress(0.5, "Integration diffusion step")
        image = context.source.copy().convert("RGB")
        mask = context.mask.convert("L")
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) >= 128:
                    image.putpixel((x, y), (220, 24, 48))
        return [image]

    adapter.generate = generate_one_image  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/outpaint",
            json={
                "adapter_id": adapter.id,
                "image": encode_png_data_url(_half_transparent_image(96, 96, 48)),
                "mask": _data_url((0, 0, 0, 255)),
                "parameters": {
                    "prompt": "integration outpaint",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                },
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        with client.websocket_connect(f"/api/jobs/{job_id}/events") as websocket:
            event = websocket.receive_json()
            assert event["job"]["id"] == job_id
            assert event["job"]["status"] == constants.JOB_SUCCEEDED

        result_response = client.get(f"/api/jobs/{job_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["job_id"] == job_id
        assert len(result["images"]) == 1
        assert result["images"][0].startswith("data:image/png;base64,")
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((5, 5)) == (32, 32, 36)
        assert output.getpixel((80, 80)) != (32, 32, 36)
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_outpaint_uses_rgba_alpha_even_if_request_mask_misses_empty_pixels() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_from_alpha_mask(context):
        image = context.source.copy().convert("RGB")
        mask = context.mask.convert("L")
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) >= 128:
                    image.putpixel((x, y), (210, 120, 40))
        return [image]

    adapter.generate = generate_from_alpha_mask  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    for x in range(48):
        for y in range(96):
            image.putpixel((x, y), (32, 32, 36, 255))

    try:
        response = client.post(
            "/api/generations/outpaint",
            json={
                "adapter_id": adapter.id,
                "image": encode_png_data_url(image),
                "mask": _data_url((0, 0, 0, 255)),
                "parameters": {
                    "prompt": "alpha mask outpaint",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.getpixel((80, 48)) == (210, 120, 40)
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_outpaint_preserves_context_even_if_generated_output_changes_it() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_changed_context(context):
        return [Image.new("RGB", context.source.size, (220, 24, 48))]

    adapter.generate = generate_changed_context  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    for x in range(48):
        for y in range(96):
            image.putpixel((x, y), (32, 32, 36, 255))

    try:
        response = client.post(
            "/api/generations/outpaint",
            json={
                "adapter_id": adapter.id,
                "image": encode_png_data_url(image),
                "mask": _data_url((0, 0, 0, 255)),
                "parameters": {
                    "prompt": "preserve source context",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_GENERATED_SELECTION,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.getpixel((80, 48)) == (220, 24, 48)
        assert result["metadata"]["composition_result_mode"] == constants.RESULT_MODE_PRESERVE_KNOWN
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_inpaint_uses_request_mask_on_opaque_selection() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_from_request_mask(context):
        image = context.source.copy().convert("RGB")
        mask = context.mask.convert("L")
        assert context.metadata["generation_mode"] == constants.GENERATION_MODE_INPAINT
        assert image.getpixel((80, 48)) == (32, 32, 36)
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) >= 128:
                    image.putpixel((x, y), (72, 180, 240))
        return [image]

    adapter.generate = generate_from_request_mask  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/inpaint",
            json={
                "adapter_id": adapter.id,
                "image": _data_url((32, 32, 36, 255)),
                "mask": encode_png_data_url(_half_generation_mask(96, 96, 48)),
                "parameters": {
                    "prompt": "repair selected area",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_PRESERVE_KNOWN,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.getpixel((80, 48)) == (72, 180, 240)
        assert result["metadata"]["mode"] == constants.GENERATION_MODE_INPAINT
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_inpaint_generated_selection_request_still_preserves_known_pixels() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_changed_full_selection(context):
        return [Image.new("RGB", context.source.size, (240, 72, 96))]

    adapter.generate = generate_changed_full_selection  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/inpaint",
            json={
                "adapter_id": adapter.id,
                "image": _data_url((32, 32, 36, 255)),
                "mask": encode_png_data_url(_half_generation_mask(96, 96, 48)),
                "parameters": {
                    "prompt": "repair selected area",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_GENERATED_SELECTION,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.getpixel((80, 48)) == (240, 72, 96)
        assert result["metadata"]["composition_result_mode"] == constants.RESULT_MODE_PRESERVE_KNOWN
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_controlnet_inpaint_passes_conditioning_image_to_adapter() -> None:
    adapter = registry.get(constants.ADAPTER_SD15_CONTROLNET_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_with_conditioning(context):
        assert context.metadata["generation_mode"] == constants.GENERATION_MODE_INPAINT
        assert context.conditioning_image is not None
        assert context.conditioning_image.size == context.source.size
        assert context.conditioning_image.getpixel((80, 48)) == (245, 245, 245)
        image = context.source.copy().convert("RGB")
        mask = context.mask.convert("L")
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) >= 128:
                    image.putpixel((x, y), (180, 72, 240))
        return [image]

    adapter.generate = generate_with_conditioning  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/inpaint",
            json={
                "adapter_id": adapter.id,
                "image": _data_url((32, 32, 36, 255)),
                "mask": encode_png_data_url(_half_generation_mask(96, 96, 48)),
                "conditioning": {
                    "type": constants.CONDITIONING_TYPE_SCRIBBLE,
                    "image": _data_url((245, 245, 245, 255)),
                },
                "parameters": {
                    "prompt": "guided repair selected area",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_PRESERVE_KNOWN,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.getpixel((80, 48)) == (180, 72, 240)
        assert result["metadata"]["mode"] == constants.GENERATION_MODE_INPAINT
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_native_sketch_inpaint_uses_conditioning_as_sdxl_source() -> None:
    adapter = registry.get(constants.ADAPTER_SDXL_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_from_native_sketch(context):
        assert context.metadata["generation_mode"] == constants.GENERATION_MODE_INPAINT
        assert context.conditioning_image is None
        assert context.source.getpixel((80, 48)) == (245, 180, 32)
        image = context.source.copy().convert("RGB")
        mask = context.mask.convert("L")
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) >= 128:
                    image.putpixel((x, y), (180, 120, 72))
        return [image]

    adapter.generate = generate_from_native_sketch  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/inpaint",
            json={
                "adapter_id": adapter.id,
                "image": _data_url((32, 32, 36, 255)),
                "mask": encode_png_data_url(_half_generation_mask(96, 96, 48)),
                "conditioning": {
                    "type": constants.CONDITIONING_TYPE_COLOR,
                    "image": _data_url((245, 180, 32, 255)),
                },
                "parameters": {
                    "prompt": "guided repair selected area",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_PRESERVE_KNOWN,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        assert output.getpixel((8, 48)) == (32, 32, 36)
        assert output.size == (96, 96)
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


@pytest.mark.integration
def test_sdxl_generation_auto_matches_lighting_before_composition() -> None:
    adapter = registry.get(constants.ADAPTER_SDXL_INPAINT)
    original_generate = adapter.generate
    original_config = adapter.loaded_config
    original_loaded_adapter_id = models.loaded_adapter_id

    def generate_with_mismatched_lighting(context):
        image = Image.new("RGB", context.source.size, (10, 95, 110))
        return [image]

    adapter.generate = generate_with_mismatched_lighting  # type: ignore[method-assign]
    adapter.loaded_config = ModelLoadRequest(adapter_id=adapter.id, device="cpu", dtype="float32")
    models.loaded_adapter_id = adapter.id

    try:
        response = client.post(
            "/api/generations/inpaint",
            json={
                "adapter_id": adapter.id,
                "image": _data_url((84, 70, 48, 255)),
                "mask": encode_png_data_url(_half_generation_mask(96, 96, 48)),
                "parameters": {
                    "prompt": "match local lighting",
                    "width": 96,
                    "height": 96,
                    "steps": 2,
                    "sample_count": 1,
                    "random_seed": False,
                    "seed": 123,
                    "correction_pipeline": [],
                    "result_mode": constants.RESULT_MODE_PRESERVE_KNOWN,
                },
            },
        )
        assert response.status_code == 200
        job = _wait_for_job(response.json()["job_id"])
        assert job["status"] == constants.JOB_SUCCEEDED, job
        result = client.get(f"/api/jobs/{job['id']}/result").json()
        output = decode_data_url(result["images"][0]).convert("RGB")
        generated_pixel = output.getpixel((80, 48))
        assert generated_pixel[0] > 70
        assert generated_pixel[1] < 85
        assert generated_pixel[2] < 70
    finally:
        adapter.generate = original_generate  # type: ignore[method-assign]
        adapter.loaded_config = original_config
        models.loaded_adapter_id = original_loaded_adapter_id


def test_sdxl_inpaint_uses_native_1024_process_size() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["width"] == 2256
            assert kwargs["height"] == constants.SDXL_MIN_PROCESS_SIZE
            assert kwargs["image"].size == (2256, constants.SDXL_MIN_PROCESS_SIZE)
            assert kwargs["mask_image"].size == (2256, constants.SDXL_MIN_PROCESS_SIZE)
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = SdxlInpaintAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (704, 320), (32, 32, 36)),
        mask=Image.new("L", (704, 320), 255),
        parameters=GenerationParameters(
            prompt="extend the scene",
            steps=2,
            scheduler=constants.SCHEDULER_AUTO,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert images[0].size == (704, 320)


def test_sdxl_adapter_artifact_records_real_generation_parameters(tmp_path) -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["strength"] == 1.0
            assert kwargs["num_images_per_prompt"] == 1
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = SdxlInpaintAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    adapter._set_scheduler = lambda _scheduler: None  # type: ignore[method-assign]
    context = GenerationContext(
        source=Image.new("RGB", (1536, 1024), (32, 32, 36)),
        mask=Image.new("L", (1536, 1024), 255),
        parameters=GenerationParameters(
            prompt="directional outpaint",
            steps=2,
            strength=1.0,
            scheduler=constants.SCHEDULER_DPM_SOLVER,
            fill_mode=constants.FILL_EDGE_EXTEND,
            mask_blur=0,
            inpaint_area=constants.INPAINT_AREA_WHOLE_SELECTION,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
        metadata={"artifact_dir": str(tmp_path)},
    )

    adapter.generate(context)

    adapter_inputs = json.loads((tmp_path / "adapter_inputs.json").read_text())
    assert adapter_inputs["process_size"] == [1536, 1024]
    assert adapter_inputs["output_size"] == [1536, 1024]
    assert adapter_inputs["strength"] == 1.0
    assert adapter_inputs["scheduler"] == constants.SCHEDULER_DPM_SOLVER
    assert adapter_inputs["fill_mode"] == constants.FILL_EDGE_EXTEND
    assert adapter_inputs["mask_blur"] == 0
    assert adapter_inputs["inpaint_area"] == constants.INPAINT_AREA_WHOLE_SELECTION
    assert adapter_inputs["sample_count"] == 1
    assert adapter_inputs["seed"] == 123


def test_diffusers_adapter_processes_non_multiple_of_eight_selection() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            width = kwargs["width"]
            height = kwargs["height"]
            assert width == 176
            assert height == 512
            assert kwargs["image"].size == (176, 512)
            assert kwargs["mask_image"].size == (176, 512)
            assert kwargs["strength"] == 1.0
            assert "padding_mask_crop" not in kwargs
            kwargs["callback_on_step_end"](self, 0, None, {})
            image = Image.new("RGB", (width, height), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = Sd15InpaintAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (172, 512), (32, 32, 36)),
        mask=Image.new("L", (172, 512), 255),
        parameters=GenerationParameters(
            prompt="selection crop",
            width=172,
            height=512,
            steps=2,
            scheduler=constants.SCHEDULER_AUTO,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1
    assert images[0].size == (172, 512)


def test_diffusers_adapter_can_crop_around_mask() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["padding_mask_crop"] == 64
            assert kwargs["strength"] == 0.65
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = Sd15InpaintAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        parameters=GenerationParameters(
            prompt="selection crop",
            steps=2,
            strength=0.65,
            scheduler=constants.SCHEDULER_AUTO,
            inpaint_area=constants.INPAINT_AREA_ONLY_MASKED,
            mask_crop_padding=64,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1


def test_sd15_controlnet_adapter_passes_conditioning_to_pipeline() -> None:
    from expandiffusion.adapters import diffusers_inpaint

    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["control_image"].size == (128, 128)
            assert kwargs["controlnet_conditioning_scale"] == 0.85
            assert kwargs["control_guidance_start"] == 0.2
            assert kwargs["control_guidance_end"] == 0.9
            assert kwargs["mask_image"].size == (128, 128)
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter_class = getattr(diffusers_inpaint, "Sd15ControlNetInpaintAdapter", None)
    assert adapter_class is not None
    adapter = adapter_class()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        conditioning_image=Image.new("RGB", (128, 128), (255, 255, 255)),
        parameters=GenerationParameters(
            prompt="guided outpaint",
            steps=2,
            scheduler=constants.SCHEDULER_AUTO,
            sample_count=1,
            random_seed=False,
            seed=123,
            controlnet_conditioning_scale=0.85,
            control_guidance_start=0.2,
            control_guidance_end=0.9,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1
    assert images[0].size == (128, 128)


def test_sd15_controlnet_adapter_can_run_without_active_guide() -> None:
    from expandiffusion.adapters import diffusers_inpaint

    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["control_image"].size == (128, 128)
            assert kwargs["controlnet_conditioning_scale"] == 0.0
            assert kwargs["mask_image"].size == (128, 128)
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter_class = getattr(diffusers_inpaint, "Sd15ControlNetInpaintAdapter", None)
    assert adapter_class is not None
    adapter = adapter_class()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        conditioning_image=None,
        parameters=GenerationParameters(
            prompt="standard inpaint through loaded controlnet adapter",
            steps=2,
            scheduler=constants.SCHEDULER_AUTO,
            sample_count=1,
            random_seed=False,
            seed=123,
            controlnet_conditioning_scale=0.85,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1


def test_sdxl_controlnet_adapter_passes_conditioning_to_pipeline() -> None:
    from expandiffusion.adapters import diffusers_inpaint

    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["control_image"].size == (
                constants.SDXL_MIN_PROCESS_SIZE,
                constants.SDXL_MIN_PROCESS_SIZE,
            )
            assert kwargs["controlnet_conditioning_scale"] == 0.85
            assert kwargs["control_guidance_start"] == 0.2
            assert kwargs["control_guidance_end"] == 0.9
            assert kwargs["mask_image"].size == (
                constants.SDXL_MIN_PROCESS_SIZE,
                constants.SDXL_MIN_PROCESS_SIZE,
            )
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter_class = getattr(diffusers_inpaint, "SdxlControlNetInpaintAdapter", None)
    assert adapter_class is not None
    adapter = adapter_class()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        conditioning_image=Image.new("RGB", (128, 128), (255, 255, 255)),
        parameters=GenerationParameters(
            prompt="guided sdxl inpaint",
            steps=2,
            scheduler=constants.SCHEDULER_AUTO,
            sample_count=1,
            random_seed=False,
            seed=123,
            controlnet_conditioning_scale=0.85,
            control_guidance_start=0.2,
            control_guidance_end=0.9,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1
    assert images[0].size == (128, 128)


def test_standard_stable_diffusion_adapter_calls_img2img_without_mask_kwargs() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert "mask_image" not in kwargs
            assert "padding_mask_crop" not in kwargs
            assert "width" not in kwargs
            assert "height" not in kwargs
            assert kwargs["negative_prompt"] == "low quality"
            assert kwargs["strength"] == 0.55
            assert kwargs["image"].size == (128, 128)
            kwargs["callback_on_step_end"](self, 0, None, {})
            image = Image.new("RGB", kwargs["image"].size, (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = Sd15Img2ImgAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        parameters=GenerationParameters(
            prompt="extend the scene",
            negative_prompt="low quality",
            steps=2,
            strength=0.55,
            scheduler=constants.SCHEDULER_AUTO,
            inpaint_area=constants.INPAINT_AREA_ONLY_MASKED,
            mask_crop_padding=64,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1


def test_flux_adapter_calls_fill_pipeline_with_supported_kwargs() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert "negative_prompt" not in kwargs
            assert "padding_mask_crop" not in kwargs
            assert kwargs["max_sequence_length"] == constants.FLUX_MAX_SEQUENCE_LENGTH
            assert kwargs["guidance_scale"] == constants.DEFAULT_FLUX_GUIDANCE
            assert kwargs["strength"] == 1.0
            assert kwargs["image"].size == (128, 128)
            assert kwargs["mask_image"].size == (128, 128)
            kwargs["callback_on_step_end"](self, 0, None, {})
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = FluxFillAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        parameters=GenerationParameters(
            prompt="fill the scene",
            negative_prompt="must be ignored",
            steps=2,
            guidance_scale=constants.DEFAULT_FLUX_GUIDANCE,
            scheduler=constants.SCHEDULER_DPM_SOLVER,
            inpaint_area=constants.INPAINT_AREA_ONLY_MASKED,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1


def test_chroma_adapter_calls_inpaint_pipeline_with_supported_kwargs() -> None:
    class DummyPipeline:
        def __call__(self, **kwargs):
            assert kwargs["negative_prompt"] == "low quality"
            assert kwargs["padding_mask_crop"] == 64
            assert kwargs["max_sequence_length"] == constants.CHROMA_MAX_SEQUENCE_LENGTH
            assert kwargs["strength"] == constants.DEFAULT_CHROMA_STRENGTH
            kwargs["callback_on_step_end"](self, 0, None, {})
            image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (5, 10, 15))
            return type("Output", (), {"images": [image]})()

    adapter = ChromaInpaintAdapter()
    adapter.pipeline = DummyPipeline()
    adapter.device = "cpu"
    context = GenerationContext(
        source=Image.new("RGB", (128, 128), (32, 32, 36)),
        mask=Image.new("L", (128, 128), 255),
        parameters=GenerationParameters(
            prompt="fill the scene",
            negative_prompt="low quality",
            steps=2,
            strength=constants.DEFAULT_CHROMA_STRENGTH,
            scheduler=constants.SCHEDULER_AUTO,
            inpaint_area=constants.INPAINT_AREA_ONLY_MASKED,
            mask_crop_padding=64,
            sample_count=1,
            random_seed=False,
            seed=123,
        ),
        progress=lambda _value, _message: None,
        is_cancelled=lambda: False,
    )

    images = adapter.generate(context)

    assert len(images) == 1


def test_source_preparation_extends_known_edge_into_empty_area() -> None:
    image = Image.new("RGBA", (16, 8), (0, 0, 0, 0))
    for x in range(8):
        for y in range(8):
            image.putpixel((x, y), (200, 20, 10, 255))
    mask = Image.new("L", (16, 8), 0)
    for x in range(8, 16):
        for y in range(8):
            mask.putpixel((x, y), 255)

    source = prepare_source_image(image, mask, constants.FILL_EDGE_EXTEND, image.size)

    assert source.getpixel((2, 4)) == (200, 20, 10)
    assert source.getpixel((14, 4)) == (200, 20, 10)


def test_source_preparation_clears_hidden_white_under_transparency() -> None:
    image = Image.new("RGBA", (16, 8), (255, 255, 255, 0))
    for x in range(8):
        for y in range(8):
            image.putpixel((x, y), (20, 60, 120, 255))
    mask = Image.new("L", (16, 8), 0)
    for x in range(8, 16):
        for y in range(8):
            mask.putpixel((x, y), 255)

    transparent_source = prepare_source_image(
        image,
        mask,
        constants.FILL_TRANSPARENT,
        image.size,
    )
    filled_source = prepare_source_image(image, mask, constants.FILL_OPENCV_NS, image.size)

    assert transparent_source.getpixel((14, 4)) == (0, 0, 0)
    assert filled_source.getpixel((14, 4)) != (255, 255, 255)


def test_generation_mask_expands_to_latent_grid_blocks() -> None:
    mask = Image.new("L", (16, 16), 0)
    mask.putpixel((9, 9), 255)

    expanded = expand_mask_to_block_grid(mask)

    assert expanded.getpixel((8, 8)) == 255
    assert expanded.getpixel((15, 15)) == 255
    assert expanded.getpixel((7, 7)) == 0


def test_empty_correction_pipeline_uses_full_generated_selection_like_original() -> None:
    original = Image.new("RGBA", (64, 32), (20, 30, 40, 255))
    generated = Image.new("RGB", (64, 32), (220, 230, 240))
    mask = Image.new("L", (64, 32), 0)
    for x in range(32, 64):
        for y in range(32):
            mask.putpixel((x, y), 255)

    corrected = compose_generation_result(
        original,
        generated,
        mask,
        constants.RESULT_MODE_GENERATED_SELECTION,
    )

    assert corrected.getpixel((4, 16)) == (220, 230, 240)
    assert corrected.getpixel((60, 16)) == (220, 230, 240)


def test_result_mode_can_preserve_known_pixels() -> None:
    original = Image.new("RGBA", (64, 32), (20, 30, 40, 255))
    generated = Image.new("RGB", (64, 32), (220, 230, 240))
    mask = Image.new("L", (64, 32), 0)
    for x in range(32, 64):
        for y in range(32):
            mask.putpixel((x, y), 255)

    result = compose_generation_result(
        original,
        generated,
        mask,
        constants.RESULT_MODE_PRESERVE_KNOWN,
    )

    assert result.getpixel((4, 16)) == (20, 30, 40)
    assert result.getpixel((60, 16)) == (220, 230, 240)


def test_sdxl_lighting_match_adjusts_generated_interior_without_breaking_edge() -> None:
    original = Image.new("RGBA", (128, 32), (0, 0, 0, 0))
    generated = Image.new("RGB", (128, 32), (34, 36, 34))
    mask = Image.new("L", (128, 32), 0)
    for x in range(64):
        for y in range(32):
            mask.putpixel((x, y), 255)
    for x in range(64, 82):
        for y in range(32):
            original.putpixel((x, y), (40, 44, 38, 255))
    for x in range(82, 128):
        for y in range(32):
            original.putpixel((x, y), (170, 176, 150, 255))

    corrected = match_generated_lighting_to_preserved_region(original, generated, mask)

    edge_luma = sum(corrected.getpixel((63, 16))) / 3
    interior_luma = sum(corrected.getpixel((8, 16))) / 3
    assert edge_luma < 70
    assert interior_luma > 70


def test_restore_original_soft_result_mode_keeps_generated_empty_area() -> None:
    original = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    for x in range(32, 64):
        for y in range(32):
            original.putpixel((x, y), (20, 30, 40, 255))
    generated = Image.new("RGB", (64, 32), (220, 230, 240))
    mask = Image.new("L", (64, 32), 0)
    for x in range(32):
        for y in range(32):
            mask.putpixel((x, y), 255)

    result = compose_generation_result(
        original,
        generated,
        mask,
        constants.RESULT_MODE_RESTORE_ORIGINAL_SOFT,
    )

    assert result.getpixel((4, 16)) == (220, 230, 240)
    assert result.getpixel((60, 16)) == (20, 30, 40)
    boundary_pixel = result.getpixel((32, 16))
    assert 20 < boundary_pixel[0] < 220


def test_restore_original_soft_result_mode_keeps_generated_inpaint_mask_area() -> None:
    original = Image.new("RGBA", (64, 32), (20, 30, 40, 255))
    generated = Image.new("RGB", (64, 32), (220, 230, 240))
    mask = Image.new("L", (64, 32), 0)
    for x in range(32, 64):
        for y in range(32):
            mask.putpixel((x, y), 255)

    result = compose_generation_result(
        original,
        generated,
        mask,
        constants.RESULT_MODE_RESTORE_ORIGINAL_SOFT,
    )

    assert result.getpixel((4, 16)) == (20, 30, 40)
    assert result.getpixel((60, 16)) == (220, 230, 240)


def test_result_refine_postprocessors_run_after_preserve_known_composition(tmp_path) -> None:
    registry = AdapterRegistry()
    adapter = _FullOutputTestAdapter()
    registry.register(adapter)
    postprocessors = GenerationPostprocessorRegistry()
    probe = _ResultRefineProbe()
    postprocessors.register(probe, plugin_id="test")
    persistence = PersistenceStore(tmp_path / "state.json")
    service = GenerationService(
        registry,
        ModelService(registry, persistence),
        JobStore(persistence),
        postprocessors,
    )
    image = Image.new("RGBA", (4, 1), (10, 20, 30, 255))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((2, 0), 255)
    mask.putpixel((3, 0), 255)
    job = service.jobs.create(
        OutpaintRequest(
            adapter_id=adapter.id,
            image=encode_png_data_url(image),
            mask=encode_png_data_url(mask),
            parameters=GenerationParameters(
                result_mode=constants.RESULT_MODE_PRESERVE_KNOWN,
            ),
            metadata={},
        )
    )

    asyncio.run(service.run_outpaint(job.id))
    output = decode_data_url(service.jobs.get_result(job.id).images[0]).convert("RGB")

    assert probe.seen_pixels == [((10, 20, 30), (200, 0, 0))]
    assert "artifact_dir" in probe.seen_metadata
    assert probe.seen_metadata["sample_index"] == 0
    assert output.getpixel((0, 0)) == (1, 2, 3)


@pytest.mark.parametrize(
    "processor_id",
    [
        "correction-mask-feather",
        "correction-border-blend",
        "correction-color-match",
        "correction-histogram-match",
        "correction-multiband-blend",
    ],
)
def test_correction_plugins_reduce_synthetic_seam(processor_id: str) -> None:
    original, generated, mask = _synthetic_seam_images()
    processor = _loaded_postprocessor(processor_id)

    before = measure_seam_discontinuity(original, generated, mask)
    corrected = processor.process(
        GenerationPostprocessorContext(
            original=original,
            generated=generated,
            mask=mask,
            parameters=GenerationParameters(correction_pipeline=[processor_id]),
            adapter=Sd15InpaintAdapter(),
            progress=lambda _value, _message: None,
            is_cancelled=lambda: False,
        )
    )
    after = measure_seam_discontinuity(original, corrected, mask)

    assert after["rgb_delta"] < before["rgb_delta"]


def test_correction_pipeline_resolves_processors_in_requested_order() -> None:
    postprocessors = GenerationPostprocessorRegistry()
    postprocessors.register(_ArithmeticCorrection("double", "double"), plugin_id="test")
    postprocessors.register(_ArithmeticCorrection("add", "add"), plugin_id="test")
    original = Image.new("RGBA", (1, 1), (0, 0, 0, 255))
    mask = Image.new("L", (1, 1), 255)
    generated = Image.new("RGB", (1, 1), (10, 0, 0))

    for processor in postprocessors.correction_pipeline(["double", "add"]):
        generated = processor.process(
            GenerationPostprocessorContext(
                original=original,
                generated=generated,
                mask=mask,
                parameters=GenerationParameters(correction_pipeline=["double", "add"]),
                adapter=Sd15InpaintAdapter(),
                progress=lambda _value, _message: None,
                is_cancelled=lambda: False,
            )
        )

    assert generated.getpixel((0, 0)) == (25, 0, 0)


def test_unknown_correction_pipeline_id_fails_explicitly() -> None:
    postprocessors = GenerationPostprocessorRegistry()

    with pytest.raises(AppError) as error:
        postprocessors.correction_pipeline(["missing-correction"])

    assert error.value.code == constants.ERROR_INVALID_GENERATION_PARAMETERS


def test_legacy_correction_mode_payload_is_rejected() -> None:
    response = client.post(
        "/api/generations/outpaint",
        json={
            "adapter_id": constants.ADAPTER_SD15_INPAINT,
            "image": encode_png_data_url(_half_transparent_image(96, 96, 48)),
            "mask": _data_url((0, 0, 0, 255)),
            "parameters": {
                "prompt": "legacy correction mode",
                "correction_mode": "disabled",
            },
        },
    )

    assert response.status_code == 422


def test_diffusers_adapter_handles_pipelines_without_safety_checker() -> None:
    class PipelineWithoutSafetyChecker:
        pass

    class PipelineWithSafetyChecker:
        safety_checker = object()

    adapter = Sd15InpaintAdapter()
    without_safety_checker = PipelineWithoutSafetyChecker()
    with_safety_checker = PipelineWithSafetyChecker()

    adapter._configure_safety_checker(without_safety_checker, enabled=False)
    adapter._configure_safety_checker(with_safety_checker, enabled=False)

    assert not hasattr(without_safety_checker, "safety_checker")
    assert with_safety_checker.safety_checker is None


@pytest.mark.real_sd
@pytest.mark.skipif(
    os.environ.get("EXPANDIFFUSION_RUN_REAL_SD_TESTS") != "1",
    reason="set EXPANDIFFUSION_RUN_REAL_SD_TESTS=1 to load and run the real SD pipeline",
)
def test_real_sd15_model_can_load_and_generate() -> None:
    load_response = client.post(
        "/api/models/load",
        json={
            "adapter_id": constants.ADAPTER_SD15_INPAINT,
            "model_id": constants.MODEL_SD15_INPAINT,
            "device": "auto",
            "dtype": "auto",
            "safety_checker": False,
        },
    )
    assert load_response.status_code == 200
    assert load_response.json()["loaded"] is True

    generation_response = client.post(
        "/api/generations/outpaint",
        json={
            "adapter_id": constants.ADAPTER_SD15_INPAINT,
            "image": _data_url((32, 32, 36, 255)),
            "mask": _data_url((255, 255, 255, 255)),
            "parameters": {
                "prompt": "plain studio wall",
                "width": 128,
                "height": 128,
                "steps": 2,
                "sample_count": 1,
                "random_seed": False,
                "seed": 123,
                "safety_checker": False,
            },
        },
    )
    assert generation_response.status_code == 200
    job_id = generation_response.json()["job_id"]
    job = _wait_for_job(job_id)
    assert job["status"] == constants.JOB_SUCCEEDED, job
    result_response = client.get(f"/api/jobs/{job_id}/result")
    assert result_response.status_code == 200
    assert result_response.json()["images"]
