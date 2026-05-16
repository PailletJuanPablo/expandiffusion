export const API_BASE_PATH = "/api";
export const ADAPTER_SD15_INPAINT = "sd15-inpaint";
export const ADAPTER_SD15_CONTROLNET_INPAINT = "sd15-controlnet-inpaint";
export const ADAPTER_SDXL_INPAINT = "sdxl-inpaint";
export const ADAPTER_SDXL_CONTROLNET_INPAINT = "sdxl-controlnet-inpaint";
export const ADAPTER_SDXL_FILL_CONTROLNET_UNION = "sdxl-fill-controlnet-union";
export const ADAPTER_SDXL_FILL_IP_REFINE = "sdxl-fill-ip-refine";
export const DEFAULT_ADAPTER_ID = ADAPTER_SDXL_FILL_IP_REFINE;
export const DEFAULT_MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning";
export const DEFAULT_CONTROLNET_MODEL_ID = "xinsir/controlnet-union-sdxl-1.0";
export const DEFAULT_CANVAS_WIDTH = 1024;
export const DEFAULT_CANVAS_HEIGHT = 768;
export const DEFAULT_SELECTION_SIZE = 512;
export const DEFAULT_BRUSH_SIZE = 48;
export const DEFAULT_ERASER_HARDNESS = 100;
export const DEFAULT_CONTROL_GUIDE_COLOR = "#f59e0b";
export const DEFAULT_CONTROL_GUIDE_STRENGTH = 100;
export const CONTROLNET_GUIDE_UI_ENABLED = false;
export const CONTROL_GUIDE_MASK_MODE_REPLACE = "replace_mask";
export const CONTROL_GUIDE_MASK_MODE_PRESERVE = "preserve_original";
export const DEFAULT_CONTROL_GUIDE_MASK_MODE = CONTROL_GUIDE_MASK_MODE_REPLACE;
export const CONTROL_GUIDE_COLORS = Object.freeze([
  "#f59e0b",
  "#ef4444",
  "#2563eb",
  "#22c55e",
  "#f8fafc",
  "#111827",
]);
export const DEFAULT_STEPS = 28;
export const DEFAULT_GUIDANCE = 7.5;
export const DEFAULT_STRENGTH = 1;
export const DEFAULT_SAMPLE_COUNT = 1;
export const DEFAULT_MASK_CROP_PADDING = 32;
export const DEFAULT_MASK_BLUR = 0;
export const MIN_BRUSH_SIZE = 4;
export const MAX_BRUSH_SIZE = 256;
export const MIN_ERASER_HARDNESS = 0;
export const MAX_ERASER_HARDNESS = 100;
export const MIN_CONTROL_GUIDE_STRENGTH = 0;
export const MAX_CONTROL_GUIDE_STRENGTH = 100;
export const DEFAULT_ZOOM = 0.72;
export const MIN_ZOOM = 0.08;
export const MAX_ZOOM = 4;
export const ZOOM_STEP = 1.08;
export const PROJECT_FILE_EXTENSION = "expd";
export const PROJECT_JSON_PATH = "project.json";
export const RASTER_ASSET_PATH = "assets/raster.png";
export const EXPORT_FILE_NAME = "expandiffusion-export.png";
export const PROJECT_FILE_NAME = "expandiffusion-project.expd";
export const MASK_STROKE_ALPHA = 0.42;
export const MASK_STROKE_COLOR = "#ffffff";
export const MASK_ERASE_COLOR = "#000000";
export const CANVAS_BACKGROUND = "#101319";
export const WHITE_PIXEL = "#ffffff";
export const BLACK_PIXEL = "#000000";

export const MODEL_SOURCE_HUB = "hub";
export const MODEL_SOURCE_LOCAL_FOLDER = "local_folder";
export const MODEL_SOURCE_SINGLE_FILE = "single_file";
export const MODEL_SOURCE_DIRECT_URL = "direct_url";
export const MODEL_SOURCE_FIELD_MODEL_ID = "model_id";
export const MODEL_SOURCE_FIELD_LOCAL_PATH = "local_path";
export const MODEL_SOURCE_FIELD_SINGLE_FILE_PATH = "single_file_path";
export const MODEL_SOURCE_FIELD_MODEL_URL = "model_url";

export const CONTROL_TEXT = "text";
export const CONTROL_TEXTAREA = "textarea";
export const CONTROL_NUMBER = "number";
export const CONTROL_SLIDER = "slider";
export const CONTROL_SELECT = "select";
export const CONTROL_SWITCH = "switch";

export const CONTROL_SECTION_BASIC = "basic";
export const CONTROL_SECTION_ADVANCED = "advanced";
export const CONTROL_SECTION_RUNTIME = "runtime";
export const CONTROL_SECTION_EXTENSIONS = "extensions";

export const POSTPROCESSOR_CATEGORY_CORRECTION = "correction";
export const PLUGIN_ACTION_MENU_SELECTION = "selection";
export const PLUGIN_TOOL_TARGET_FRAME = "frame";
export const PLUGIN_TOOL_TARGET_IMAGE = "image";

export const TOOL_SELECT = "select";
export const TOOL_OUTPAINT_FRAME = "outpaint_frame";
export const TOOL_PAN = "pan";
export const TOOL_ERASE = "erase";
export const TOOL_INPAINT_MASK = "inpaint_mask";
export const TOOL_CONTROL_GUIDE = "control_guide";
export const TOOL_IMPORT = "import";
export const PLUGIN_TOOL_PREFIX = "plugin:";

export const STROKE_PAINT = "paint";
export const STROKE_ERASE = "erase";

export const JOB_QUEUED = "queued";
export const JOB_RUNNING = "running";
export const JOB_SUCCEEDED = "succeeded";
export const JOB_FAILED = "failed";
export const JOB_CANCELLED = "cancelled";

export const ERROR_MODEL_LOAD_CANCELLED = "MODEL_LOAD_CANCELLED";

export const FILL_PATCHMATCH = "patchmatch";
export const FILL_EDGE_EXTEND = "edge_pad";
export const FILL_GAUSSIAN_NOISE = "gaussian";
export const FILL_PERLIN_NOISE = "perlin";
export const FILL_OPENCV_TELEA = "cv2_telea";
export const FILL_OPENCV_NS = "cv2_ns";
export const FILL_TRANSPARENT = "transparent";

export const INPAINT_AREA_WHOLE_SELECTION = "whole_selection";
export const INPAINT_AREA_ONLY_MASKED = "only_masked";

export const RESULT_MODE_GENERATED_SELECTION = "generated_selection";
export const RESULT_MODE_PRESERVE_KNOWN = "preserve_known";
export const RESULT_MODE_FEATHER_KNOWN = "feather_known";
export const RESULT_MODE_RESTORE_ORIGINAL_SOFT = "restore_original_soft";

export const GENERATION_MODE_OUTPAINT = "outpaint";
export const GENERATION_MODE_INPAINT = "inpaint";
export const WORKSPACE_MODE_FREE_EDIT = "free_edit";
export const WORKSPACE_MODE_EXPAND_IMAGE = "expand_image";
export const OUTPAINT_STRATEGY_LOCAL_CONTEXT = "local_context";
export const OUTPAINT_STRATEGY_SELECTED_FRAME = "selected_frame";
export const OUTPAINT_STRATEGY_FULL_CONTEXT_CROP = "full_context_crop";
export const OUTPAINT_STRATEGY_WHOLE_RESIZED = "whole_resized";
export const OUTPAINT_STRATEGY_DIRECTIONAL = "directional";
export const OUTPAINT_STRATEGY_HF_SPACE_FILL = "hf_space_fill";
export const OUTPAINT_DIRECTION_LEFT = "left";
export const OUTPAINT_DIRECTION_RIGHT = "right";
export const OUTPAINT_DIRECTION_UP = "up";
export const OUTPAINT_DIRECTION_DOWN = "down";
export const OUTPAINT_DIRECTION_AROUND = "around";
export const DIRECTIONAL_OUTPAINT_MIN_SIZE = 1024;
export const DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE = 1024;
export const DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE = 512;
export const DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE = 1024;
export const CONDITIONING_TYPE_SCRIBBLE = "scribble";
export const CONDITIONING_TYPE_COLOR = "color";

export const SCHEDULER_AUTO = "auto";
export const SCHEDULER_DPM_SOLVER = "dpmpp_2m";
export const SCHEDULER_EULER = "euler";
export const SCHEDULER_DDIM = "ddim";
export const SCHEDULER_LMS = "lms";

export const TOOL_SHORTCUTS = Object.freeze({
  SELECT: "v",
  PAN: "h",
  ERASE: "e",
  INPAINT_MASK: "b",
  CONTROL_GUIDE: "g",
  GENERATE: "Enter",
  ESCAPE: "Escape",
});

export const FILL_OPTIONS = [
  { id: FILL_PATCHMATCH, label: "patchmatch" },
  { id: FILL_EDGE_EXTEND, label: "edge_pad" },
  { id: FILL_OPENCV_NS, label: "cv2_ns" },
  { id: FILL_OPENCV_TELEA, label: "cv2_telea" },
  { id: FILL_PERLIN_NOISE, label: "perlin" },
  { id: FILL_GAUSSIAN_NOISE, label: "gaussian" },
  { id: FILL_TRANSPARENT, label: "transparent" },
];

export const SCHEDULER_OPTIONS = [
  { id: SCHEDULER_AUTO, label: "Auto" },
  { id: SCHEDULER_DPM_SOLVER, label: "DPM++ 2M" },
  { id: SCHEDULER_EULER, label: "Euler" },
  { id: SCHEDULER_DDIM, label: "DDIM" },
  { id: SCHEDULER_LMS, label: "LMS" },
];

export const MODEL_SOURCE_OPTIONS = [
  { id: MODEL_SOURCE_HUB, label: "Hugging Face model id" },
  { id: MODEL_SOURCE_LOCAL_FOLDER, label: "Local Diffusers folder" },
  { id: MODEL_SOURCE_SINGLE_FILE, label: "Local checkpoint file" },
  { id: MODEL_SOURCE_DIRECT_URL, label: "Checkpoint URL" },
];

export const INPAINT_AREA_OPTIONS = [
  { id: INPAINT_AREA_WHOLE_SELECTION, label: "whole selection" },
  { id: INPAINT_AREA_ONLY_MASKED, label: "only masked crop" },
];

export const RESULT_MODE_OPTIONS = [
  { id: RESULT_MODE_GENERATED_SELECTION, label: "generated selection" },
  { id: RESULT_MODE_PRESERVE_KNOWN, label: "preserve known" },
  { id: RESULT_MODE_FEATHER_KNOWN, label: "feather known" },
  { id: RESULT_MODE_RESTORE_ORIGINAL_SOFT, label: "restore original soft" },
];

export const OUTPAINT_STRATEGY_OPTIONS = [
  { id: OUTPAINT_STRATEGY_HF_SPACE_FILL, label: "HF Space fill" },
  { id: OUTPAINT_STRATEGY_DIRECTIONAL, label: "Directional SDXL" },
  { id: OUTPAINT_STRATEGY_LOCAL_CONTEXT, label: "Local context" },
  { id: OUTPAINT_STRATEGY_FULL_CONTEXT_CROP, label: "Full context crop" },
  { id: OUTPAINT_STRATEGY_WHOLE_RESIZED, label: "Whole resized" },
  { id: OUTPAINT_STRATEGY_SELECTED_FRAME, label: "Selected frame" },
];

export const OUTPAINT_DIRECTION_OPTIONS = [
  { id: OUTPAINT_DIRECTION_RIGHT, label: "Right" },
  { id: OUTPAINT_DIRECTION_LEFT, label: "Left" },
  { id: OUTPAINT_DIRECTION_DOWN, label: "Down" },
  { id: OUTPAINT_DIRECTION_UP, label: "Up" },
  { id: OUTPAINT_DIRECTION_AROUND, label: "Around" },
];

export const CONTROL_GUIDE_MASK_MODE_OPTIONS = [
  { id: CONTROL_GUIDE_MASK_MODE_REPLACE, label: "Prompt replaces mask" },
  { id: CONTROL_GUIDE_MASK_MODE_PRESERVE, label: "Preserve original" },
];

export type BuiltInEditorTool =
  | typeof TOOL_SELECT
  | typeof TOOL_OUTPAINT_FRAME
  | typeof TOOL_PAN
  | typeof TOOL_ERASE
  | typeof TOOL_INPAINT_MASK
  | typeof TOOL_CONTROL_GUIDE
  | typeof TOOL_IMPORT;

export type PluginEditorTool = `${typeof PLUGIN_TOOL_PREFIX}${string}`;
export type EditorTool = BuiltInEditorTool | PluginEditorTool;

export type MaskStrokeMode = typeof STROKE_PAINT | typeof STROKE_ERASE;

export type GenerationMode =
  | typeof GENERATION_MODE_OUTPAINT
  | typeof GENERATION_MODE_INPAINT;

export type WorkspaceMode =
  | typeof WORKSPACE_MODE_FREE_EDIT
  | typeof WORKSPACE_MODE_EXPAND_IMAGE;

export type OutpaintStrategy =
  | typeof OUTPAINT_STRATEGY_LOCAL_CONTEXT
  | typeof OUTPAINT_STRATEGY_FULL_CONTEXT_CROP
  | typeof OUTPAINT_STRATEGY_SELECTED_FRAME
  | typeof OUTPAINT_STRATEGY_WHOLE_RESIZED
  | typeof OUTPAINT_STRATEGY_DIRECTIONAL
  | typeof OUTPAINT_STRATEGY_HF_SPACE_FILL;

export type OutpaintDirection =
  | typeof OUTPAINT_DIRECTION_LEFT
  | typeof OUTPAINT_DIRECTION_RIGHT
  | typeof OUTPAINT_DIRECTION_UP
  | typeof OUTPAINT_DIRECTION_DOWN
  | typeof OUTPAINT_DIRECTION_AROUND;

export type ControlGuideMaskMode =
  | typeof CONTROL_GUIDE_MASK_MODE_REPLACE
  | typeof CONTROL_GUIDE_MASK_MODE_PRESERVE;

export function pluginEditorToolId(toolId: string): PluginEditorTool {
  return `${PLUGIN_TOOL_PREFIX}${toolId}`;
}

export function pluginToolIdFromEditorTool(tool: EditorTool): string | null {
  return tool.startsWith(PLUGIN_TOOL_PREFIX)
    ? tool.slice(PLUGIN_TOOL_PREFIX.length)
    : null;
}

export function isPluginEditorTool(tool: EditorTool): boolean {
  return pluginToolIdFromEditorTool(tool) !== null;
}
