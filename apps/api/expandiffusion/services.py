"""Application services."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from PIL import Image, ImageDraw

from . import constants
from .adapters.base import GenerationContext
from .adapters.registry import AdapterRegistry
from .errors import AppError, GenerationCancelled
from .generation_artifacts import GenerationArtifactRecorder
from .image_utils import (
    combine_generation_masks,
    compose_generation_result,
    decode_data_url,
    encode_png_data_url,
    expand_mask_to_block_grid,
    generation_mask_from_alpha,
    match_generated_lighting_to_preserved_region,
    measure_seam_discontinuity,
    normalize_mask,
    prepare_source_image,
)
from .jobs import JobRecord, JobStore
from .model_storage import ModelStorage
from .persistence import PersistenceStore
from .postprocessors import GenerationPostprocessorContext, GenerationPostprocessorRegistry
from .schemas import (
    JobCreateResponse,
    JobResult,
    ModelInfo,
    ModelLoadProgress,
    ModelLoadRequest,
    OutpaintRequest,
)

logger = logging.getLogger("uvicorn.error")


class ModelService:
    """Coordinates adapter load and unload operations."""

    def __init__(
        self,
        registry: AdapterRegistry,
        persistence: PersistenceStore,
        model_storage: ModelStorage | None = None,
    ) -> None:
        self.registry = registry
        self.persistence = persistence
        self.model_storage = model_storage or ModelStorage()
        self.loaded_adapter_id: str | None = None
        self._load_progress = ModelLoadProgress(
            status="idle",
            progress=0.0,
            message="No model load running.",
            updated_at=_utc_now(),
        )
        self._load_progress_lock = threading.Lock()
        self._load_operation_lock = threading.Lock()
        self._load_cancel_event = threading.Event()

    def list_models(self) -> list[ModelInfo]:
        """Return current loaded model state for all adapters."""
        models: list[ModelInfo] = []
        for adapter in self.registry.adapters():
            config = adapter.loaded_config
            models.append(
                ModelInfo(
                    adapter_id=adapter.id,
                    adapter_label=adapter.label,
                    model_id=config.model_id if config else adapter.default_model_id,
                    local_path=config.local_path if config else None,
                    single_file_path=config.single_file_path if config else None,
                    model_url=config.model_url if config else None,
                    device=getattr(adapter, "device", constants.DEFAULT_DEVICE),
                    dtype=getattr(adapter, "dtype", constants.DEFAULT_DTYPE),
                    loaded=adapter.loaded,
                )
            )
        return models

    def load(self, request: ModelLoadRequest) -> ModelInfo:
        """Load an adapter and unload the previously active adapter."""
        if not self._load_operation_lock.acquire(blocking=False):
            raise AppError(
                constants.ERROR_MODEL_LOAD_IN_PROGRESS,
                "A model load is already running. Cancel it before starting another load.",
                status_code=409,
            )
        source = _model_load_source(request)
        try:
            self._load_cancel_event.clear()
            self._update_load_progress(
                status="running",
                adapter_id=request.adapter_id,
                source=source,
                progress=0.01,
                message="Resolving model source.",
                files_done=None,
                files_total=None,
            )
            self._raise_if_load_cancelled()
            request = self._resolve_model_url(request)
            self._raise_if_load_cancelled()
            adapter = self.registry.get(request.adapter_id)
            if self.loaded_adapter_id and self.loaded_adapter_id != request.adapter_id:
                previous_adapter_id = self.loaded_adapter_id
                self.registry.get(previous_adapter_id).unload()
                self.loaded_adapter_id = None
                self.persistence.record_model_unloaded(previous_adapter_id)
            if adapter.loaded:
                adapter.unload()
                if self.loaded_adapter_id == request.adapter_id:
                    self.loaded_adapter_id = None
                    self.persistence.record_model_unloaded(request.adapter_id)
            self._load_adapter(adapter, request, source)
            self._raise_if_load_cancelled()
            self.loaded_adapter_id = request.adapter_id
            model = self._model_info(request.adapter_id)
            self.persistence.record_model_loaded(model, request)
            self._update_load_progress(
                status="succeeded",
                adapter_id=request.adapter_id,
                source=source,
                progress=1.0,
                message="Model loaded.",
            )
            return model
        except AppError as exc:
            if exc.code == constants.ERROR_MODEL_LOAD_CANCELLED:
                self._unload_cancelled_adapter(request.adapter_id)
                status = "cancelled"
            else:
                status = "failed"
            self._update_load_progress(
                status=status,
                adapter_id=request.adapter_id,
                source=source,
                message=exc.message,
            )
            raise
        except Exception as exc:
            self._update_load_progress(
                status="failed",
                adapter_id=request.adapter_id,
                source=source,
                message=str(exc),
            )
            raise
        finally:
            self._load_cancel_event.clear()
            self._load_operation_lock.release()

    def load_progress(self) -> ModelLoadProgress:
        """Return the latest model load progress."""
        with self._load_progress_lock:
            return self._load_progress.model_copy()

    def cancel_load(self) -> ModelLoadProgress:
        """Request cancellation for the active model load."""
        if not self._is_load_active():
            return self.load_progress()
        self._load_cancel_event.set()
        self._update_load_progress(
            status="cancelling",
            message="Cancelling model load after the current step.",
        )
        return self.load_progress()

    def _resolve_model_url(self, request: ModelLoadRequest) -> ModelLoadRequest:
        if not request.model_url:
            return request
        self._update_load_progress(
            status="running",
            adapter_id=request.adapter_id,
            source="Direct model URL",
            progress=0.03,
            message="Downloading direct model URL.",
        )
        self._raise_if_load_cancelled()

        def progress(
            bytes_done: int,
            bytes_total: int | None,
            filename: str,
        ) -> None:
            self._raise_if_load_cancelled()
            ratio = bytes_done / bytes_total if bytes_total else 0.0
            self._update_load_progress(
                status="running",
                adapter_id=request.adapter_id,
                source="Direct model URL",
                progress=0.03 + ratio * 0.67 if bytes_total else 0.05,
                message=_download_message(filename, bytes_done, bytes_total),
                file_name=filename,
                file_bytes_done=bytes_done,
                file_bytes_total=bytes_total,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
            )
            self._raise_if_load_cancelled()

        local_file = self.model_storage.resolve_url(
            request.model_url,
            progress=progress,
            is_cancelled=self._load_cancel_event.is_set,
        )
        return request.model_copy(
            update={
                "model_id": None,
                "local_path": None,
                "single_file_path": str(local_file),
            }
        )

    def _load_adapter(self, adapter: Any, request: ModelLoadRequest, source: str) -> None:
        progress_parameters = inspect.signature(adapter.load).parameters

        def progress(
            value: float,
            message: str,
            details: dict[str, int | str | None] | None = None,
        ) -> None:
            self._raise_if_load_cancelled()
            self._update_load_progress(
                status="running",
                adapter_id=request.adapter_id,
                source=source,
                progress=value,
                message=message,
                files_done=_int_detail(details, "files_done"),
                files_total=_int_detail(details, "files_total"),
                file_name=_str_detail(details, "file_name"),
                file_bytes_done=_int_detail(details, "file_bytes_done"),
                file_bytes_total=_int_detail(details, "file_bytes_total"),
                bytes_done=_int_detail(details, "bytes_done"),
                bytes_total=_int_detail(details, "bytes_total"),
            )
            self._raise_if_load_cancelled()

        kwargs: dict[str, Any] = {}
        if "progress" in progress_parameters:
            kwargs["progress"] = progress
        if "is_cancelled" in progress_parameters:
            kwargs["is_cancelled"] = self._load_cancel_event.is_set
        if kwargs:
            adapter.load(request, **kwargs)
            return
        self._raise_if_load_cancelled()
        adapter.load(request)
        self._raise_if_load_cancelled()

    def _update_load_progress(
        self,
        *,
        status: str,
        adapter_id: str | None = None,
        source: str | None = None,
        progress: float | None = None,
        message: str,
        files_done: int | None = None,
        files_total: int | None = None,
        file_name: str | None = None,
        file_bytes_done: int | None = None,
        file_bytes_total: int | None = None,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
    ) -> None:
        with self._load_progress_lock:
            previous = self._load_progress
            next_progress = previous.progress if progress is None else progress
            self._load_progress = ModelLoadProgress(
                status=status,
                adapter_id=adapter_id if adapter_id is not None else previous.adapter_id,
                source=source if source is not None else previous.source,
                progress=max(0.0, min(constants.PROGRESS_FINISHED, next_progress)),
                message=message,
                files_done=files_done,
                files_total=files_total,
                file_name=file_name,
                file_bytes_done=file_bytes_done,
                file_bytes_total=file_bytes_total,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
                updated_at=_utc_now(),
            )
            current = self._load_progress
        logger.info(
            (
                "Model load progress: status=%s adapter_id=%s source=%s progress=%s "
                "message=%s files=%s/%s file=%s file_bytes=%s/%s bytes=%s/%s"
            ),
            current.status,
            current.adapter_id,
            current.source,
            round(current.progress * 100),
            current.message,
            current.files_done if current.files_done is not None else "-",
            current.files_total if current.files_total is not None else "-",
            current.file_name or "-",
            current.file_bytes_done if current.file_bytes_done is not None else "-",
            current.file_bytes_total if current.file_bytes_total is not None else "-",
            current.bytes_done if current.bytes_done is not None else "-",
            current.bytes_total if current.bytes_total is not None else "-",
        )

    def unload(self, adapter_id: str) -> ModelInfo:
        """Unload an adapter."""
        if self._is_load_active():
            raise AppError(
                constants.ERROR_MODEL_LOAD_IN_PROGRESS,
                "Cancel the running model load before unloading.",
                status_code=409,
            )
        adapter = self.registry.get(adapter_id)
        adapter.unload()
        if self.loaded_adapter_id == adapter_id:
            self.loaded_adapter_id = None
        model = self._model_info(adapter_id)
        self.persistence.record_model_unloaded(adapter_id)
        return model

    def ensure_loaded(self, adapter_id: str) -> None:
        """Require explicit model loading before generation."""
        adapter = self.registry.get(adapter_id)
        if adapter.loaded:
            return
        raise AppError(
            constants.ERROR_MODEL_NOT_LOADED,
            "Load the selected model before running generation.",
            status_code=409,
        )

    def _model_info(self, adapter_id: str) -> ModelInfo:
        for model in self.list_models():
            if model.adapter_id == adapter_id:
                return model
        raise AppError(
            constants.ERROR_ADAPTER_NOT_FOUND,
            "Adapter is not registered.",
            status_code=404,
        )

    def _is_load_active(self) -> bool:
        with self._load_progress_lock:
            return self._load_progress.status in {"running", "cancelling"}

    def _raise_if_load_cancelled(self) -> None:
        if self._load_cancel_event.is_set():
            raise AppError(
                constants.ERROR_MODEL_LOAD_CANCELLED,
                "Model load cancelled.",
                status_code=409,
            )

    def _unload_cancelled_adapter(self, adapter_id: str) -> None:
        try:
            adapter = self.registry.get(adapter_id)
        except AppError:
            return
        if adapter.loaded:
            adapter.unload()
        if self.loaded_adapter_id == adapter_id:
            self.loaded_adapter_id = None


def _model_load_source(request: ModelLoadRequest) -> str:
    if request.model_id:
        return request.model_id
    if request.local_path:
        return request.local_path
    if request.single_file_path:
        return request.single_file_path
    if request.model_url:
        return "Direct model URL"
    return "Unknown model source"


def _composition_result_mode(mode: str, requested_result_mode: str) -> str:
    if (
        mode in {constants.GENERATION_MODE_OUTPAINT, constants.GENERATION_MODE_INPAINT}
        and requested_result_mode == constants.RESULT_MODE_GENERATED_SELECTION
    ):
        return constants.RESULT_MODE_PRESERVE_KNOWN
    return requested_result_mode


def _adapter_returns_full_output(adapter: Any) -> bool:
    return bool(getattr(adapter, "returns_full_output", False))


def _uses_hf_space_full_output(adapter: Any | None, mode: str, parameters: Any) -> bool:
    return (
        mode == constants.GENERATION_MODE_OUTPAINT
        and adapter is not None
        and _adapter_returns_full_output(adapter)
        and _extra_parameter(parameters, "outpaint_strategy")
        == constants.OUTPAINT_STRATEGY_HF_SPACE_FILL
    )


def _prepare_hf_space_fill_request(
    adapter: Any,
    image: Image.Image,
    parameters: Any,
    mode: str,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]] | None:
    if mode != constants.GENERATION_MODE_OUTPAINT:
        return None
    if not _adapter_returns_full_output(adapter):
        return None
    if (
        _extra_parameter(parameters, "outpaint_strategy")
        != constants.OUTPAINT_STRATEGY_HF_SPACE_FILL
    ):
        return None

    requested_width = max(64, int(parameters.width))
    requested_height = max(64, int(parameters.height))
    target_width = _hf_space_fill_safe_dimension(requested_width)
    target_height = _hf_space_fill_safe_dimension(requested_height)
    source = _visible_source_image(image)
    scale_factor = min(target_width / source.width, target_height / source.height)
    resized_width = max(64, int(source.width * scale_factor))
    resized_height = max(64, int(source.height * scale_factor))
    source = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    resize_percentage = _hf_space_resize_percentage(
        str(_extra_parameter(parameters, "hf_space_resize_option", "Full")),
        _extra_parameter(parameters, "hf_space_custom_resize_percentage", 50),
    )
    resize_factor = resize_percentage / 100
    resized_width = max(64, int(source.width * resize_factor))
    resized_height = max(64, int(source.height * resize_factor))
    source = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    overlap_percentage = int(_extra_parameter(parameters, "hf_space_overlap_percentage", 10))
    overlap_x = max(int(resized_width * (overlap_percentage / 100)), 1)
    overlap_y = max(int(resized_height * (overlap_percentage / 100)), 1)
    overlap_left = _bool_extra_parameter(parameters, "hf_space_overlap_left", True)
    overlap_right = _bool_extra_parameter(parameters, "hf_space_overlap_right", True)
    overlap_top = _bool_extra_parameter(parameters, "hf_space_overlap_top", True)
    overlap_bottom = _bool_extra_parameter(parameters, "hf_space_overlap_bottom", True)
    alignment = _hf_space_alignment_for_direction(
        str(_extra_parameter(parameters, "outpaint_direction", "right"))
    )
    margin_x, margin_y = _hf_space_margins(
        alignment,
        target_width,
        target_height,
        resized_width,
        resized_height,
    )

    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", (target_width, target_height), 255)
    mask_draw = ImageDraw.Draw(mask)
    white_gaps_patch = 2
    left = margin_x + (overlap_x if overlap_left else white_gaps_patch)
    right = margin_x + resized_width - (overlap_x if overlap_right else white_gaps_patch)
    top = margin_y + (overlap_y if overlap_top else white_gaps_patch)
    bottom = margin_y + resized_height - (overlap_y if overlap_bottom else white_gaps_patch)
    if alignment == "Left":
        left = margin_x + (overlap_x if overlap_left else 0)
    elif alignment == "Right":
        right = margin_x + resized_width - (overlap_x if overlap_right else 0)
    elif alignment == "Top":
        top = margin_y + (overlap_y if overlap_top else 0)
    elif alignment == "Bottom":
        bottom = margin_y + resized_height - (overlap_y if overlap_bottom else 0)
    mask_draw.rectangle([(left, top), (right, bottom)], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    return (
        background,
        cnet_image,
        mask,
        {
            "target_size": [target_width, target_height],
            "requested_target_size": [requested_width, requested_height],
            "size_multiple": constants.HF_SPACE_FILL_SIZE_MULTIPLE,
            "source_size": list(source.size),
            "alignment": alignment,
            "overlap_percentage": overlap_percentage,
            "overlap_sides": {
                "left": overlap_left,
                "right": overlap_right,
                "top": overlap_top,
                "bottom": overlap_bottom,
            },
            "resize_percentage": resize_percentage,
            "overlap": [overlap_x, overlap_y],
            "margin": [margin_x, margin_y],
            "preserved_rect": [left, top, right, bottom],
        },
    )


def _compose_adapter_result(
    mode: str,
    parameters: Any,
    image: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    adapter: Any | None = None,
) -> Image.Image:
    if _uses_hf_space_full_output(adapter, mode, parameters):
        return generated.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    return compose_generation_result(
        image,
        generated,
        mask,
        _composition_result_mode(mode, parameters.result_mode),
    )


def _hf_space_fill_safe_dimension(value: int) -> int:
    multiple = constants.HF_SPACE_FILL_SIZE_MULTIPLE
    return max(multiple, value - (value % multiple))


def _hf_space_resize_percentage(option: str, custom_value: Any) -> int:
    if option == "50%":
        return 50
    if option == "33%":
        return 33
    if option == "25%":
        return 25
    if option == "Custom":
        try:
            return max(1, min(100, int(custom_value)))
        except (TypeError, ValueError):
            return 50
    return 100


def _visible_source_image(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    alpha = image.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        raise AppError(
            "OUTPAINT_CONTEXT_REQUIRED",
            "Import visible image content before generating HF Space outpaint.",
        )
    return image.crop(bounds).convert("RGB")


def _hf_space_alignment_for_direction(direction: str) -> str:
    if direction == "around":
        return "Middle"
    if direction == "left":
        return "Right"
    if direction == "up":
        return "Bottom"
    if direction == "down":
        return "Top"
    return "Left"


def _hf_space_margins(
    alignment: str,
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    margin_x = (target_width - source_width) // 2
    margin_y = (target_height - source_height) // 2
    if alignment == "Left":
        margin_x = 0
    elif alignment == "Right":
        margin_x = target_width - source_width
    elif alignment == "Top":
        margin_y = 0
    elif alignment == "Bottom":
        margin_y = target_height - source_height
    return (
        max(0, min(margin_x, target_width - source_width)),
        max(0, min(margin_y, target_height - source_height)),
    )


def _extra_parameter(parameters: Any, key: str, default: Any = None) -> Any:
    extra = getattr(parameters, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return getattr(parameters, key, default)


def _bool_extra_parameter(parameters: Any, key: str, default: bool) -> bool:
    value = _extra_parameter(parameters, key, default)
    return value if isinstance(value, bool) else default


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _int_detail(details: dict[str, int | str | None] | None, key: str) -> int | None:
    if not details:
        return None
    value = details.get(key)
    return value if isinstance(value, int) else None


def _str_detail(details: dict[str, int | str | None] | None, key: str) -> str | None:
    if not details:
        return None
    value = details.get(key)
    return value if isinstance(value, str) else None


def _download_message(filename: str, done: int, total: int | None) -> str:
    if total:
        return f"Downloading {filename}: {_format_bytes(done)} / {_format_bytes(total)}."
    return f"Downloading {filename}: {_format_bytes(done)}."


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{amount:.1f} TB"


class GenerationService:
    """Creates and executes generation jobs."""

    def __init__(
        self,
        registry: AdapterRegistry,
        models: ModelService,
        jobs: JobStore,
        postprocessors: GenerationPostprocessorRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.models = models
        self.jobs = jobs
        self.postprocessors = postprocessors or GenerationPostprocessorRegistry()
        self.artifacts = GenerationArtifactRecorder()

    def start_outpaint(self, request: OutpaintRequest) -> JobCreateResponse:
        """Create an outpaint job."""
        return self._start_generation(
            request.model_copy(update={"mode": constants.GENERATION_MODE_OUTPAINT})
        )

    def start_inpaint(self, request: OutpaintRequest) -> JobCreateResponse:
        """Create an inpaint job."""
        return self._start_generation(
            request.model_copy(update={"mode": constants.GENERATION_MODE_INPAINT})
        )

    def _start_generation(self, request: OutpaintRequest) -> JobCreateResponse:
        self.models.ensure_loaded(request.adapter_id)
        self.postprocessors.correction_pipeline(request.parameters.correction_pipeline)
        job = self.jobs.create(request)
        return JobCreateResponse(job_id=job.id, status=job.status)

    async def run_outpaint(self, job_id: str) -> None:
        """Execute a previously created outpaint job."""
        await self._run_generation(self.jobs.get(job_id))

    async def run_inpaint(self, job_id: str) -> None:
        """Execute a previously created inpaint job."""
        await self._run_generation(self.jobs.get(job_id))

    async def _run_generation(self, job: JobRecord) -> None:
        await self.jobs.mark_running(job)
        loop = asyncio.get_running_loop()

        def progress(value: float, message: str) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.jobs.update(job, value, message))
            )

        def is_cancelled() -> bool:
            return job.cancel_requested or job.status == constants.JOB_CANCELLED

        try:
            request = job.request
            adapter = self.registry.get(request.adapter_id)
            image = decode_data_url(request.image)
            artifact_dir = self.artifacts.job_dir(job.id)
            self.artifacts.save_image(job.id, "request_image.png", image)
            raw_request_mask = decode_data_url(request.mask)
            self.artifacts.save_image(job.id, "request_mask_raw.png", raw_request_mask)
            prepared_hf_space = _prepare_hf_space_fill_request(
                adapter,
                image,
                request.parameters,
                request.mode,
            )
            if prepared_hf_space is not None:
                background, image, raw_request_mask, metadata = prepared_hf_space
                self.artifacts.save_image(job.id, "hf_space_background.png", background)
                self.artifacts.save_image(job.id, "hf_space_cnet_image.png", image)
                self.artifacts.save_image(job.id, "hf_space_mask.png", raw_request_mask)
                self.artifacts.save_json(job.id, "hf_space_inputs.json", metadata)
            conditioning_image = (
                decode_data_url(request.conditioning.image).convert("RGB").resize(image.size)
                if request.conditioning
                else None
            )
            if conditioning_image is not None:
                self.artifacts.save_image(job.id, "conditioning_input.png", conditioning_image)
            request_mask = normalize_mask(raw_request_mask, image.size)
            self.artifacts.save_image(job.id, "request_mask_normalized.png", request_mask)
            if request.mode == constants.GENERATION_MODE_INPAINT:
                mask = request_mask
            else:
                alpha_mask = generation_mask_from_alpha(image)
                self.artifacts.save_image(job.id, "outpaint_alpha_mask.png", alpha_mask)
                mask = combine_generation_masks(alpha_mask, request_mask)
            self.artifacts.save_image(job.id, "generation_mask.png", mask)
            parameters = request.parameters.model_copy(
                update={"width": image.width, "height": image.height}
            )
            self.artifacts.save_json(
                job.id,
                "request.json",
                {
                    "job_id": job.id,
                    "artifact_dir": str(artifact_dir),
                    "adapter_id": request.adapter_id,
                    "mode": request.mode,
                    "project_id": request.project_id,
                    "metadata": request.metadata,
                    "parameters": parameters.model_dump(mode="json"),
                    "image_size": list(image.size),
                    "mask_size": list(mask.size),
                    "conditioning": {
                        "type": request.conditioning.type,
                        "size": list(conditioning_image.size) if conditioning_image else None,
                    }
                    if request.conditioning
                    else None,
                },
            )
            directional_plan = request.metadata.get("directional_outpaint_plan")
            if isinstance(directional_plan, dict):
                self.artifacts.save_json(
                    job.id,
                    "directional_outpaint_plan.json",
                    directional_plan,
                )
            source = self._prepare_source_for_generation(
                adapter,
                image,
                mask,
                parameters,
                request.mode,
            )
            native_sketch_image = (
                conditioning_image
                if conditioning_image is not None and not adapter.capabilities.controlnet
                else None
            )
            if native_sketch_image is not None:
                source = native_sketch_image.convert("RGB").resize(source.size)
                self.artifacts.save_image(job.id, "native_sketch_source.png", source)
            self.artifacts.save_image(job.id, "source_model_input.png", source)
            pipeline_mask = normalize_mask(mask, source.size)
            if not _adapter_returns_full_output(adapter):
                pipeline_mask = expand_mask_to_block_grid(pipeline_mask)
            self.artifacts.save_image(job.id, "pipeline_mask.png", pipeline_mask)
            context = GenerationContext(
                source=source,
                mask=pipeline_mask,
                parameters=parameters,
                progress=progress,
                is_cancelled=is_cancelled,
                conditioning_image=None if native_sketch_image is not None else conditioning_image,
                metadata={
                    **request.metadata,
                    "artifact_dir": str(artifact_dir),
                    "generation_mode": request.mode,
                    "guide_mode": "native_sketch" if native_sketch_image is not None else None,
                },
            )
            images = await asyncio.to_thread(adapter.generate, context)
            if is_cancelled():
                raise GenerationCancelled()
            corrected_images = []
            seam_metrics = []
            postprocessor_diagnostics = []
            generation_processors = self.postprocessors.processors_except_categories(
                {
                    constants.POSTPROCESSOR_CATEGORY_CORRECTION,
                    constants.POSTPROCESSOR_CATEGORY_RESULT_REFINE,
                }
            )
            correction_processors = self.postprocessors.correction_pipeline(
                parameters.correction_pipeline
            )
            result_refine_processors = self.postprocessors.processors_for_category(
                constants.POSTPROCESSOR_CATEGORY_RESULT_REFINE
            )
            for sample_index, generated in enumerate(images):
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_adapter_generated.png",
                    generated,
                )
                for processor in generation_processors:
                    generated = await self._run_postprocessor(
                        processor,
                        image,
                        generated,
                        mask,
                        parameters,
                        adapter,
                        progress,
                        is_cancelled,
                        request.metadata,
                        job.id,
                        sample_index,
                        postprocessor_diagnostics,
                    )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_after_generation_postprocessors.png",
                    generated,
                )
                base_generated = self._match_lighting_for_adapter(adapter, image, generated, mask)
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_base_lighting_matched.png",
                    base_generated,
                )
                base_result = _compose_adapter_result(
                    request.mode,
                    parameters,
                    image,
                    base_generated,
                    mask,
                    adapter,
                )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_base_composed.png",
                    base_result,
                )
                for processor in correction_processors:
                    generated = await self._run_postprocessor(
                        processor,
                        image,
                        generated,
                        mask,
                        parameters,
                        adapter,
                        progress,
                        is_cancelled,
                        request.metadata,
                        job.id,
                        sample_index,
                        postprocessor_diagnostics,
                    )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_after_correction_postprocessors.png",
                    generated,
                )
                corrected_generated = self._match_lighting_for_adapter(
                    adapter, image, generated, mask
                )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_corrected_lighting_matched.png",
                    corrected_generated,
                )
                corrected = _compose_adapter_result(
                    request.mode,
                    parameters,
                    image,
                    corrected_generated,
                    mask,
                    adapter,
                )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_result_composed.png",
                    corrected,
                )
                for processor in result_refine_processors:
                    corrected = await self._run_postprocessor(
                        processor,
                        image,
                        corrected,
                        mask,
                        parameters,
                        adapter,
                        progress,
                        is_cancelled,
                        request.metadata,
                        job.id,
                        sample_index,
                        postprocessor_diagnostics,
                    )
                self.artifacts.save_image(
                    job.id,
                    f"sample_{sample_index:02d}_after_result_refine_postprocessors.png",
                    corrected,
                )
                corrected_images.append(corrected)
                seam_metrics.append(
                    {
                        "before": measure_seam_discontinuity(image, base_result, mask),
                        "after": measure_seam_discontinuity(image, corrected, mask),
                    }
                )
            result = JobResult(
                job_id=job.id,
                images=[encode_png_data_url(image) for image in corrected_images],
                metadata={
                    "adapter_id": request.adapter_id,
                    "mode": request.mode,
                    "width": image.width,
                    "height": image.height,
                    "correction_pipeline": parameters.correction_pipeline,
                    "inpaint_area": parameters.inpaint_area,
                    "mask_crop_padding": parameters.mask_crop_padding,
                    "mask_blur": parameters.mask_blur,
                    "result_mode": parameters.result_mode,
                    "composition_result_mode": _composition_result_mode(
                        request.mode, parameters.result_mode
                    ),
                    "effective_composition_result_mode": (
                        constants.RESULT_MODE_GENERATED_SELECTION
                        if _uses_hf_space_full_output(adapter, request.mode, parameters)
                        else _composition_result_mode(request.mode, parameters.result_mode)
                    ),
                    "adapter_returns_full_output": _adapter_returns_full_output(adapter),
                    "artifact_dir": str(artifact_dir),
                    "seam_metrics": seam_metrics,
                    "postprocessors": postprocessor_diagnostics,
                },
            )
            self.artifacts.save_json(job.id, "result.json", result.metadata)
            await self.jobs.complete(job, result)
        except GenerationCancelled:
            job.cancel_requested = True
            self.artifacts.save_json(job.id, "error.json", {"status": constants.JOB_CANCELLED})
            await self.jobs.cancel(job.id)
        except AppError as exc:
            reason = exc.details.get("reason")
            self.artifacts.save_json(
                job.id,
                "error.json",
                {"status": constants.JOB_FAILED, "message": exc.message, "details": exc.details},
            )
            await self.jobs.fail(job, f"{exc.message} {reason}" if reason else exc.message)
        except Exception as exc:
            self.artifacts.save_json(
                job.id,
                "error.json",
                {"status": constants.JOB_FAILED, "message": str(exc)},
            )
            await self.jobs.fail(job, str(exc))

    def _prepare_source_for_generation(
        self,
        adapter: Any,
        image: Any,
        mask: Any,
        parameters: Any,
        mode: str,
    ) -> Any:
        if mode == constants.GENERATION_MODE_INPAINT:
            return image.convert("RGB")
        if _uses_hf_space_full_output(adapter, mode, parameters):
            return image.convert("RGB")
        if mode == constants.GENERATION_MODE_OUTPAINT and _adapter_returns_full_output(adapter):
            source = image.convert("RGB")
            generation_mask = normalize_mask(mask, source.size)
            return Image.composite(
                Image.new("RGB", source.size, (0, 0, 0)),
                source,
                generation_mask,
            )
        return prepare_source_image(
            image,
            mask,
            parameters.fill_mode,
            image.size,
        )

    def _match_lighting_for_adapter(
        self,
        adapter: Any,
        original: Any,
        generated: Any,
        mask: Any,
    ) -> Any:
        if _adapter_returns_full_output(adapter):
            return generated
        if adapter.family != constants.FAMILY_SDXL:
            return generated
        return match_generated_lighting_to_preserved_region(original, generated, mask)

    async def _run_postprocessor(
        self,
        processor: Any,
        original: Any,
        generated: Any,
        mask: Any,
        parameters: Any,
        adapter: Any,
        progress: Any,
        is_cancelled: Any,
        metadata: dict[str, Any],
        job_id: str,
        sample_index: int,
        diagnostics: list[dict[str, Any]],
    ) -> Any:
        report_start = len(diagnostics)
        result = await asyncio.to_thread(
            processor.process,
            GenerationPostprocessorContext(
                original=original,
                generated=generated,
                mask=mask,
                parameters=parameters,
                adapter=adapter,
                progress=progress,
                is_cancelled=is_cancelled,
                metadata={
                    **metadata,
                    "artifact_dir": str(self.artifacts.job_dir(job_id)),
                    "job_id": job_id,
                    "sample_index": sample_index,
                },
                diagnostics=diagnostics,
            ),
        )
        for report in diagnostics[report_start:]:
            report.setdefault("processor_id", processor.id)
            report.setdefault("processor_category", processor.category)
            report.setdefault("sample_index", sample_index)
        if is_cancelled():
            raise GenerationCancelled()
        return result
