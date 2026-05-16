"""Pydantic API schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import constants


class ControlOption(BaseModel):
    """Selectable option for a schema-driven frontend control."""

    id: str
    label: str


class ControlSchema(BaseModel):
    """Frontend control metadata supplied by an adapter."""

    id: str
    label: str
    kind: str
    section: str
    plugin_id: str | None = None
    options: list[ControlOption] = Field(default_factory=list)
    default_value: Any | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    rows: int | None = None
    placeholder: str | None = None


class ModelSourceSchema(BaseModel):
    """Model source metadata supplied by an adapter."""

    id: str
    label: str
    request_field: str
    placeholder: str | None = None
    default_value: str | None = None


class PluginManifest(BaseModel):
    """Plugin manifest loaded from plugin.json."""

    id: str
    label: str
    version: str
    description: str = ""


class PluginInfo(BaseModel):
    """Loaded plugin status exposed by the API."""

    id: str
    label: str
    version: str
    description: str = ""
    path: str
    adapter_ids: list[str] = Field(default_factory=list)
    postprocessor_ids: list[str] = Field(default_factory=list)
    action_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    loaded: bool
    error_code: str | None = None
    error: str | None = None


class PluginActionInfo(BaseModel):
    """Serializable plugin action metadata."""

    id: str
    label: str
    description: str = ""
    plugin_id: str | None = None
    menu: str = constants.PLUGIN_ACTION_MENU_SELECTION
    controls: list[ControlSchema] = Field(default_factory=list)
    default_values: dict[str, Any] = Field(default_factory=dict)


class PluginToolInfo(BaseModel):
    """Serializable plugin-provided editor tool metadata."""

    id: str
    label: str
    description: str = ""
    plugin_id: str | None = None
    action_id: str
    icon: str = "puzzle"
    icon_color: str | None = None
    accent_color: str | None = None
    result_label: str | None = None
    target: str = constants.PLUGIN_TOOL_TARGET_FRAME
    live_preview: bool = False
    controls: list[ControlSchema] = Field(default_factory=list)
    default_values: dict[str, Any] = Field(default_factory=dict)


class PluginActionRunRequest(BaseModel):
    """Image/action payload sent to a plugin action."""

    image: str
    controls: dict[str, Any] = Field(default_factory=dict)
    target: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginActionResult(BaseModel):
    """Generic plugin action result."""

    action_id: str
    text: str | None = None
    image: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class PostprocessorInfo(BaseModel):
    """Serializable generation postprocessor metadata."""

    id: str
    label: str
    description: str = ""
    plugin_id: str | None = None
    category: str = constants.POSTPROCESSOR_CATEGORY_GENERATION
    default_order: int = constants.DEFAULT_POSTPROCESSOR_ORDER


class AdapterCapabilities(BaseModel):
    """Capabilities exposed by a model adapter."""

    inpaint: bool = False
    outpaint: bool = False
    img2img: bool = False
    txt2img: bool = False
    lora: bool = False
    controlnet: bool = False
    ip_adapter: bool = False
    textual_inversion: bool = False
    safety_checker: bool = False
    schedulers: list[str] = Field(default_factory=list)
    from_single_file: bool = False


class AdapterInfo(BaseModel):
    """Serializable adapter metadata."""

    id: str
    label: str
    family: str
    description: str
    default_model_id: str | None = None
    capabilities: AdapterCapabilities
    loaded: bool = False
    plugin_id: str | None = None
    model_sources: list[ModelSourceSchema] = Field(default_factory=list)
    load_controls: list[ControlSchema] = Field(default_factory=list)
    generation_controls: list[ControlSchema] = Field(default_factory=list)
    generation_defaults: dict[str, Any] = Field(default_factory=dict)
    postprocessors: list[PostprocessorInfo] = Field(default_factory=list)


class LoraConfig(BaseModel):
    """LoRA weight reference for a generation request."""

    path: str
    scale: float = 1.0


class TextualInversionConfig(BaseModel):
    """Textual inversion reference for a generation request."""

    path: str
    token: str | None = None


class ModelLoadRequest(BaseModel):
    """Model loading options for an adapter."""

    adapter_id: str
    model_id: str | None = None
    local_path: str | None = None
    single_file_path: str | None = None
    model_url: str | None = None
    controlnet_model_id: str | None = None
    controlnet_local_path: str | None = None
    device: str = constants.DEFAULT_DEVICE
    dtype: str = constants.DEFAULT_DTYPE
    safety_checker: bool = True
    loras: list[LoraConfig] = Field(default_factory=list)
    textual_inversions: list[TextualInversionConfig] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Loaded model state."""

    adapter_id: str
    adapter_label: str
    model_id: str | None = None
    local_path: str | None = None
    single_file_path: str | None = None
    model_url: str | None = None
    device: str
    dtype: str
    loaded: bool


class ModelLoadProgress(BaseModel):
    """Current model load progress."""

    status: str
    adapter_id: str | None = None
    source: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str
    files_done: int | None = None
    files_total: int | None = None
    file_name: str | None = None
    file_bytes_done: int | None = None
    file_bytes_total: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    updated_at: str


class PersistedModelLoad(BaseModel):
    """Model load record persisted to disk."""

    adapter_id: str
    adapter_label: str
    model_id: str | None = None
    local_path: str | None = None
    single_file_path: str | None = None
    model_url: str | None = None
    device: str
    dtype: str
    safety_checker: bool
    loaded_at: str
    unloaded_at: str | None = None


class PersistedProject(BaseModel):
    """Project activity summary persisted to disk."""

    project_id: str
    created_at: str
    updated_at: str
    generation_count: int = 0
    last_job_id: str | None = None
    last_prompt: str = ""
    last_status: str = ""
    width: int | None = None
    height: int | None = None


class PersistedGeneration(BaseModel):
    """Generation job summary persisted to disk."""

    job_id: str
    project_id: str | None = None
    adapter_id: str
    prompt: str = ""
    status: str
    created_at: str
    updated_at: str
    width: int
    height: int
    sample_count: int
    result_count: int = 0
    error: str | None = None


class PersistedPluginState(BaseModel):
    """Persisted plugin enablement state."""

    plugin_id: str
    enabled: bool
    updated_at: str


class PersistentState(BaseModel):
    """Application state persisted in the local JSON file."""

    version: int = constants.PERSISTENCE_VERSION
    updated_at: str
    current_model: PersistedModelLoad | None = None
    model_loads: list[PersistedModelLoad] = Field(default_factory=list)
    projects: list[PersistedProject] = Field(default_factory=list)
    generations: list[PersistedGeneration] = Field(default_factory=list)
    plugin_states: list[PersistedPluginState] = Field(default_factory=list)


class RuntimeDeviceInfo(BaseModel):
    """CUDA device visible to PyTorch."""

    id: str
    name: str
    total_memory: int | None = None
    free_memory: int | None = None


class RuntimeInfo(BaseModel):
    """Backend hardware/runtime state."""

    torch_version: str | None
    torchvision_version: str | None
    cuda_available: bool
    cuda_version: str | None
    devices: list[RuntimeDeviceInfo]
    preferred_device: str
    preferred_dtype: str
    note: str


class GenerationParameters(BaseModel):
    """Generation parameters common to inpaint and outpaint operations."""

    model_config = ConfigDict(extra="allow")

    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(default=constants.DEFAULT_WIDTH, ge=1, le=4096)
    height: int = Field(default=constants.DEFAULT_HEIGHT, ge=1, le=4096)
    steps: int = Field(default=constants.DEFAULT_STEPS, ge=2, le=150)
    guidance_scale: float = Field(default=constants.DEFAULT_GUIDANCE, ge=0.0, le=30.0)
    strength: float = Field(default=constants.DEFAULT_STRENGTH, ge=0.05, le=1.0)
    seed: int | None = None
    random_seed: bool = True
    sample_count: int = Field(default=constants.DEFAULT_SAMPLE_COUNT, ge=1, le=8)
    scheduler: str = constants.SCHEDULER_DPM_SOLVER
    safety_checker: bool = True
    img2img: bool = False
    fill_mode: str = constants.FILL_OPENCV_NS
    correction_pipeline: list[str] = Field(default_factory=list)
    inpaint_area: str = constants.INPAINT_AREA_WHOLE_SELECTION
    mask_crop_padding: int = Field(default=constants.DEFAULT_MASK_CROP_PADDING, ge=0, le=512)
    mask_blur: int = Field(default=constants.DEFAULT_MASK_BLUR, ge=0, le=64)
    outpaint_max_width: int = Field(default=constants.DEFAULT_OUTPAINT_MAX_WIDTH, ge=512, le=4096)
    outpaint_max_height: int = Field(default=constants.DEFAULT_OUTPAINT_MAX_HEIGHT, ge=512, le=4096)
    result_mode: str = constants.RESULT_MODE_GENERATED_SELECTION
    conditioning_type: str = constants.CONDITIONING_TYPE_COLOR
    controlnet_conditioning_scale: float = Field(
        default=constants.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        ge=0.0,
        le=2.0,
    )
    control_guidance_start: float = Field(
        default=constants.DEFAULT_CONTROL_GUIDANCE_START,
        ge=0.0,
        le=1.0,
    )
    control_guidance_end: float = Field(
        default=constants.DEFAULT_CONTROL_GUIDANCE_END,
        ge=0.0,
        le=1.0,
    )
    loras: list[LoraConfig] = Field(default_factory=list)
    textual_inversions: list[TextualInversionConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_correction_mode(cls, value: Any) -> Any:
        """Reject the removed single-correction API field."""
        if isinstance(value, dict) and "correction_mode" in value:
            raise ValueError("correction_mode was removed. Use correction_pipeline instead.")
        return value


class GenerationConditioning(BaseModel):
    """Additional conditioning image for ControlNet-style adapters."""

    type: str = constants.CONDITIONING_TYPE_COLOR
    image: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject unknown conditioning map types."""
        if value in {constants.CONDITIONING_TYPE_COLOR, constants.CONDITIONING_TYPE_SCRIBBLE}:
            return value
        raise ValueError("conditioning type must be 'color' or 'scribble'.")


class OutpaintRequest(BaseModel):
    """Generation request encoded as data URLs."""

    adapter_id: str
    image: str
    mask: str
    conditioning: GenerationConditioning | None = None
    parameters: GenerationParameters
    mode: str = constants.GENERATION_MODE_OUTPAINT
    project_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """Reject unknown generation modes."""
        if value in {constants.GENERATION_MODE_OUTPAINT, constants.GENERATION_MODE_INPAINT}:
            return value
        raise ValueError("mode must be 'outpaint' or 'inpaint'.")


class JobCreateResponse(BaseModel):
    """Response returned when a job is created."""

    job_id: str
    status: str


class JobInfo(BaseModel):
    """Public job status."""

    id: str
    status: str
    progress: float
    message: str
    adapter_id: str
    created_at: str
    updated_at: str
    error: str | None = None
    result_count: int = 0


class JobResult(BaseModel):
    """Generated image payload for a completed job."""

    job_id: str
    images: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Backend health payload."""

    ok: bool
    version: str
    loaded_adapter_id: str | None = None
    runtime: RuntimeInfo
