import type {
  ControlGuideMaskMode,
  EditorTool,
  EraserMode,
  GenerationMode,
  MaskStrokeMode,
  OutpaintDirection,
  OutpaintStrategy,
  WorkspaceMode,
} from '../constants/domain'

export interface Point {
  x: number
  y: number
}

export interface SelectionRect {
  x: number
  y: number
  width: number
  height: number
}

export interface DocumentBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface PreparedRasterImport {
  dataUrl: string
  width: number
  height: number
  rasterBounds: DocumentBounds
  selection: SelectionRect
}

export interface ViewportState {
  x: number
  y: number
  zoom: number
}

export interface MaskStroke {
  id: string
  mode: MaskStrokeMode
  size: number
  points: Point[]
}

export interface ControlStroke {
  id: string
  size: number
  color?: string
  strength?: number
  points: Point[]
}

export interface ReferenceImageLayer {
  id: string
  dataUrl: string
  x: number
  y: number
  width: number
  height: number
  opacity: number
}

export type CanvasSelectionTarget =
  | { kind: 'none' }
  | { kind: 'frame' }
  | { kind: 'raster' }
  | { kind: 'canvas'; point?: Point; points?: Point[] }
  | { kind: 'reference'; id: string }

export interface EditorDocument {
  id: string
  width: number
  height: number
  rasterDataUrl: string | null
  rasterBounds: DocumentBounds | null
  selection: SelectionRect
  semanticMaskDataUrl: string | null
  maskStrokes: MaskStroke[]
  controlStrokes: ControlStroke[]
  references: ReferenceImageLayer[]
}

export interface GenerationParameters {
  [key: string]: unknown
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  guidance_scale: number
  strength: number
  seed: number | null
  random_seed: boolean
  sample_count: number
  scheduler: string
  safety_checker: boolean
  img2img: boolean
  fill_mode: string
  correction_pipeline: string[]
  inpaint_area: string
  mask_crop_padding: number
  mask_blur: number
  outpaint_max_width: number
  outpaint_max_height: number
  result_mode: string
  outpaint_strategy: OutpaintStrategy
  outpaint_direction: OutpaintDirection
  outpaint_generated_size: number
  outpaint_context_size: number
  outpaint_cross_size: number
  hf_space_overlap_percentage: number
  hf_space_overlap_left: boolean
  hf_space_overlap_right: boolean
  hf_space_overlap_top: boolean
  hf_space_overlap_bottom: boolean
  hf_space_resize_option: string
  hf_space_custom_resize_percentage: number
  fixed_expand_percent: number
  fixed_expand_width_percent: number
  fixed_expand_height_percent: number
  fixed_expand_output_scale: string
  fixed_expand_custom_output_scale: number
  fixed_expand_show_guides: boolean
  loras: LoraConfig[]
  textual_inversions: TextualInversionConfig[]
}

export interface LoraConfig {
  path: string
  scale: number
}

export interface TextualInversionConfig {
  path: string
  token: string | null
}

export interface ControlOption {
  id: string
  label: string
}

export interface ControlSchema {
  id: string
  label: string
  kind: string
  section: string
  plugin_id: string | null
  options: ControlOption[]
  default_value: unknown
  min: number | null
  max: number | null
  step: number | null
  rows: number | null
  placeholder: string | null
}

export interface ModelSourceSchema {
  id: string
  label: string
  request_field: string
  placeholder: string | null
  default_value: string | null
}

export interface PluginInfo {
  id: string
  label: string
  version: string
  description: string
  path: string
  adapter_ids: string[]
  postprocessor_ids: string[]
  action_ids: string[]
  tool_ids: string[]
  enabled: boolean
  loaded: boolean
  error_code: string | null
  error: string | null
}

export interface PluginActionInfo {
  id: string
  label: string
  description: string
  plugin_id: string | null
  menu: string
  controls: ControlSchema[]
  default_values: Record<string, unknown>
}

export interface PluginToolInfo {
  id: string
  label: string
  description: string
  plugin_id: string | null
  action_id: string
  icon: string
  icon_color: string | null
  accent_color: string | null
  result_label: string | null
  target: string
  live_preview: boolean
  controls: ControlSchema[]
  default_values: Record<string, unknown>
}

export interface PluginActionRunRequest {
  image: string
  controls: Record<string, unknown>
  target: Record<string, unknown>
  metadata: Record<string, unknown>
}

export interface PluginActionResult {
  action_id: string
  text: string | null
  image: string | null
  mask: string | null
  data: Record<string, unknown>
}

export interface PostprocessorInfo {
  id: string
  label: string
  description: string
  plugin_id: string | null
  category: string
  default_order: number
}

export interface AdapterCapabilities {
  inpaint: boolean
  outpaint: boolean
  img2img: boolean
  txt2img: boolean
  lora: boolean
  controlnet: boolean
  ip_adapter: boolean
  textual_inversion: boolean
  safety_checker: boolean
  schedulers: string[]
  from_single_file: boolean
}

export interface AdapterInfo {
  id: string
  label: string
  family: string
  description: string
  default_model_id: string | null
  capabilities: AdapterCapabilities
  loaded: boolean
  plugin_id: string | null
  model_sources: ModelSourceSchema[]
  load_controls: ControlSchema[]
  generation_controls: ControlSchema[]
  generation_defaults: Record<string, unknown>
  postprocessors: PostprocessorInfo[]
}

export interface ModelLoadRequest {
  adapter_id: string
  model_id: string | null
  local_path: string | null
  single_file_path: string | null
  model_url: string | null
  controlnet_model_id?: string | null
  controlnet_local_path?: string | null
  device: string
  dtype: string
  safety_checker: boolean
  loras: LoraConfig[]
  textual_inversions: TextualInversionConfig[]
}

export interface ModelInfo {
  adapter_id: string
  adapter_label: string
  model_id: string | null
  local_path: string | null
  single_file_path: string | null
  model_url: string | null
  device: string
  dtype: string
  loaded: boolean
}

export interface ModelLoadProgress {
  status: string
  adapter_id: string | null
  source: string | null
  progress: number
  message: string
  files_done: number | null
  files_total: number | null
  file_name: string | null
  file_bytes_done: number | null
  file_bytes_total: number | null
  bytes_done: number | null
  bytes_total: number | null
  updated_at: string
}

export interface PersistedModelLoad {
  adapter_id: string
  adapter_label: string
  model_id: string | null
  local_path: string | null
  single_file_path: string | null
  model_url: string | null
  device: string
  dtype: string
  safety_checker: boolean
  loaded_at: string
  unloaded_at: string | null
}

export interface PersistedProject {
  project_id: string
  created_at: string
  updated_at: string
  generation_count: number
  last_job_id: string | null
  last_prompt: string
  last_status: string
  width: number | null
  height: number | null
}

export interface PersistedGeneration {
  job_id: string
  project_id: string | null
  adapter_id: string
  prompt: string
  status: string
  created_at: string
  updated_at: string
  width: number
  height: number
  sample_count: number
  result_count: number
  error: string | null
}

export interface PersistedPluginState {
  plugin_id: string
  enabled: boolean
  updated_at: string
}

export interface PersistentState {
  version: number
  updated_at: string
  current_model: PersistedModelLoad | null
  model_loads: PersistedModelLoad[]
  projects: PersistedProject[]
  generations: PersistedGeneration[]
  plugin_states: PersistedPluginState[]
}

export interface RuntimeDeviceInfo {
  id: string
  name: string
  total_memory: number | null
  free_memory: number | null
}

export interface RuntimeInfo {
  torch_version: string | null
  torchvision_version: string | null
  cuda_available: boolean
  cuda_version: string | null
  devices: RuntimeDeviceInfo[]
  preferred_device: string
  preferred_dtype: string
  note: string
}

export interface OutpaintRequest {
  adapter_id: string
  image: string
  mask: string
  conditioning?: GenerationConditioning | null
  parameters: GenerationParameters
  mode?: GenerationMode
  project_id: string | null
  metadata: Record<string, unknown>
}

export interface GenerationConditioning {
  type: string
  image: string
}

export interface JobCreateResponse {
  job_id: string
  status: string
}

export interface JobInfo {
  id: string
  status: string
  progress: number
  message: string
  adapter_id: string
  created_at: string
  updated_at: string
  error: string | null
  result_count: number
}

export interface JobResult {
  job_id: string
  images: string[]
  metadata: Record<string, unknown>
}

export interface JobEvent {
  type: string
  job: JobInfo
}

export interface GenerationHistoryItem {
  id: string
  adapterId: string
  prompt: string
  createdAt: string
  images: string[]
  acceptedImage: string | null
  resultBounds?: DocumentBounds | null
}

export interface EditorSnapshot {
  document: EditorDocument
}

export interface PluginImagePreview {
  toolId: string
  image: string
  target: CanvasSelectionTarget
  boxes?: PluginPreviewBox[]
}

export interface PluginPreviewBox {
  id: string
  label: string
  bounds: DocumentBounds
}

export interface EditorStoreState {
  document: EditorDocument
  viewport: ViewportState
  tool: EditorTool
  canvasSelectionTarget: CanvasSelectionTarget
  brushSize: number
  eraserHardness: number
  eraserMode: EraserMode
  controlGuideEnabled: boolean
  controlGuideColor: string
  controlGuideStrength: number
  controlGuideMaskMode: ControlGuideMaskMode
  workspaceMode: WorkspaceMode
  generationMode: GenerationMode
  selectedAdapterId: string
  modelSource: string
  modelId: string
  localPath: string
  singleFilePath: string
  modelUrl: string
  controlnetModelId: string
  device: string
  dtype: string
  parameters: GenerationParameters
  currentJob: JobInfo | null
  currentJobSelection: SelectionRect | null
  generationNote: string | null
  pendingResults: string[]
  pendingResultBounds: DocumentBounds | null
  pendingResultReplacesDocument: boolean
  selectedResultIndex: number
  pluginPreview: PluginImagePreview | null
  history: GenerationHistoryItem[]
  errorMessage: string | null
  undoStack: EditorSnapshot[]
  redoStack: EditorSnapshot[]
}
