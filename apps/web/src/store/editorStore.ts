import { create } from "zustand";
import {
  ADAPTER_SDXL_INPAINT,
  DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE,
  DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
  DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
  DEFAULT_ADAPTER_ID,
  DEFAULT_BRUSH_SIZE,
  DEFAULT_CANVAS_HEIGHT,
  DEFAULT_CONTROL_GUIDE_COLOR,
  DEFAULT_CONTROL_GUIDE_MASK_MODE,
  DEFAULT_CONTROL_GUIDE_STRENGTH,
  DEFAULT_CONTROLNET_MODEL_ID,
  DEFAULT_CANVAS_WIDTH,
  DEFAULT_ERASER_HARDNESS,
  DEFAULT_MODEL_ID,
  DEFAULT_SELECTION_SIZE,
  DEFAULT_ZOOM,
  FILL_EDGE_EXTEND,
  FILL_OPENCV_NS,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  INPAINT_AREA_WHOLE_SELECTION,
  MODEL_SOURCE_HUB,
  OUTPAINT_DIRECTION_DOWN,
  OUTPAINT_DIRECTION_LEFT,
  OUTPAINT_DIRECTION_AROUND,
  OUTPAINT_DIRECTION_RIGHT,
  OUTPAINT_DIRECTION_UP,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  OUTPAINT_STRATEGY_SELECTED_FRAME,
  OUTPAINT_STRATEGY_WHOLE_RESIZED,
  RESULT_MODE_GENERATED_SELECTION,
  RESULT_MODE_PRESERVE_KNOWN,
  SCHEDULER_DPM_SOLVER,
  STROKE_ERASE,
  STROKE_PAINT,
  TOOL_INPAINT_MASK,
  TOOL_OUTPAINT_FRAME,
  TOOL_SELECT,
  type ControlGuideMaskMode,
  type EditorTool,
  type GenerationMode,
  type MaskStrokeMode,
  type OutpaintDirection,
  type OutpaintStrategy,
} from "../constants/domain";
import type {
  CanvasSelectionTarget,
  ControlStroke,
  DocumentBounds,
  EditorDocument,
  EditorStoreState,
  GenerationHistoryItem,
  GenerationParameters,
  JobInfo,
  MaskStroke,
  Point,
  PreparedRasterImport,
  PluginImagePreview,
  ReferenceImageLayer,
  SelectionRect,
} from "../domain/types";

interface EditorStoreActions {
  setTool: (tool: EditorTool) => void;
  setCanvasSelectionTarget: (target: CanvasSelectionTarget) => void;
  setViewport: (x: number, y: number, zoom: number) => void;
  setBrushSize: (size: number) => void;
  setEraserHardness: (hardness: number) => void;
  setControlGuideEnabled: (enabled: boolean) => void;
  setControlGuideColor: (color: string) => void;
  setControlGuideStrength: (strength: number) => void;
  setControlGuideMaskMode: (mode: ControlGuideMaskMode) => void;
  setGenerationMode: (mode: GenerationMode) => void;
  setSelection: (selection: SelectionRect) => void;
  moveSelection: (x: number, y: number) => void;
  resizeCanvas: (width: number, height: number) => void;
  setRaster: (
    dataUrl: string,
    width?: number,
    height?: number,
    selection?: SelectionRect,
    rasterBounds?: DocumentBounds,
  ) => void;
  setPreparedRaster: (raster: PreparedRasterImport) => void;
  updateRasterDataUrl: (dataUrl: string) => void;
  replaceRasterDataUrl: (dataUrl: string) => void;
  updateRasterBounds: (bounds: DocumentBounds) => void;
  addReference: (dataUrl: string, width: number, height: number) => void;
  updateReference: (id: string, patch: Partial<ReferenceImageLayer>) => void;
  beginMaskStroke: (mode: MaskStrokeMode, point: Point) => void;
  appendMaskPoint: (point: Point) => void;
  clearMask: () => void;
  beginControlStroke: (point: Point) => void;
  appendControlPoint: (point: Point) => void;
  clearControlGuide: () => void;
  updateParameter: <TKey extends keyof GenerationParameters>(
    key: TKey,
    value: GenerationParameters[TKey],
  ) => void;
  applyDirectionalOutpaintPreset: () => void;
  setSelectedAdapterId: (
    adapterId: string,
    defaultModelId: string | null,
    generationDefaults?: Record<string, unknown>,
  ) => void;
  setModelSource: (modelSource: string) => void;
  setModelId: (modelId: string) => void;
  setLocalPath: (localPath: string) => void;
  setSingleFilePath: (singleFilePath: string) => void;
  setModelUrl: (modelUrl: string) => void;
  setControlnetModelId: (modelId: string) => void;
  setDevice: (device: string) => void;
  setDtype: (dtype: string) => void;
  setCurrentJob: (job: JobInfo | null) => void;
  setCurrentJobSelection: (selection: SelectionRect | null) => void;
  setGenerationNote: (message: string | null) => void;
  setPendingResults: (
    images: string[],
    bounds?: DocumentBounds | null,
    replacesDocument?: boolean,
  ) => void;
  selectResult: (index: number) => void;
  acceptSelectedResult: () => void;
  rejectResults: () => void;
  setPluginPreview: (preview: PluginImagePreview) => void;
  clearPluginPreview: (toolId?: string) => void;
  pushHistory: (item: GenerationHistoryItem) => void;
  setErrorMessage: (message: string | null) => void;
  undo: () => void;
  redo: () => void;
  replaceDocument: (documentState: EditorDocument) => void;
}

export type EditorStore = EditorStoreState & EditorStoreActions;

/**
 * Editor-wide state store for document, tools, generation and history.
 */
export const useEditorStore = create<EditorStore>((set, get) => ({
  document: createInitialDocument(),
  viewport: { x: 72, y: 48, zoom: DEFAULT_ZOOM },
  tool: TOOL_OUTPAINT_FRAME,
  canvasSelectionTarget: { kind: "frame" },
  brushSize: DEFAULT_BRUSH_SIZE,
  eraserHardness: DEFAULT_ERASER_HARDNESS,
  controlGuideEnabled: false,
  controlGuideColor: DEFAULT_CONTROL_GUIDE_COLOR,
  controlGuideStrength: DEFAULT_CONTROL_GUIDE_STRENGTH,
  controlGuideMaskMode: DEFAULT_CONTROL_GUIDE_MASK_MODE,
  generationMode: GENERATION_MODE_OUTPAINT,
  selectedAdapterId: DEFAULT_ADAPTER_ID,
  modelSource: MODEL_SOURCE_HUB,
  modelId: DEFAULT_MODEL_ID,
  localPath: "",
  singleFilePath: "",
  modelUrl: "",
  controlnetModelId: DEFAULT_CONTROLNET_MODEL_ID,
  device: "auto",
  dtype: "auto",
  parameters: createInitialParameters(),
  currentJob: null,
  currentJobSelection: null,
  generationNote: null,
  pendingResults: [],
  pendingResultBounds: null,
  pendingResultReplacesDocument: false,
  selectedResultIndex: 0,
  pluginPreview: null,
  history: [],
  errorMessage: null,
  undoStack: [],
  redoStack: [],

  setTool: (tool) =>
    set((state) => ({
      tool,
      pluginPreview: state.tool === tool ? state.pluginPreview : null,
      canvasSelectionTarget:
        tool === TOOL_SELECT &&
        state.generationMode === GENERATION_MODE_OUTPAINT &&
        state.canvasSelectionTarget.kind === "none"
          ? { kind: "frame" }
          : tool === TOOL_INPAINT_MASK
            ? { kind: "none" }
            : state.canvasSelectionTarget,
    })),
  setCanvasSelectionTarget: (canvasSelectionTarget) =>
    set({ canvasSelectionTarget, pluginPreview: null }),
  setViewport: (x, y, zoom) => set({ viewport: { x, y, zoom } }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setEraserHardness: (eraserHardness) => set({ eraserHardness }),
  setControlGuideEnabled: (controlGuideEnabled) => set({ controlGuideEnabled }),
  setControlGuideColor: (controlGuideColor) => set({ controlGuideColor }),
  setControlGuideStrength: (controlGuideStrength) =>
    set({ controlGuideStrength }),
  setControlGuideMaskMode: (controlGuideMaskMode) =>
    set({ controlGuideMaskMode }),
  setGenerationMode: (generationMode) =>
    set((state) => ({
      generationMode,
      pluginPreview: null,
      canvasSelectionTarget:
        generationMode === GENERATION_MODE_INPAINT
          ? { kind: "none" }
          : { kind: "frame" },
      tool:
        generationMode === GENERATION_MODE_INPAINT
          ? TOOL_INPAINT_MASK
          : TOOL_OUTPAINT_FRAME,
      parameters:
        generationMode === GENERATION_MODE_INPAINT &&
        state.parameters.result_mode === RESULT_MODE_GENERATED_SELECTION
          ? { ...state.parameters, result_mode: RESULT_MODE_PRESERVE_KNOWN }
          : state.parameters,
    })),
  setSelection: (selection) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      selection: sanitizeSelection(selection),
    })),
  moveSelection: (x, y) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      selection: sanitizeSelection({ ...documentState.selection, x, y }),
    })),
  resizeCanvas: (width, height) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      width,
      height,
      selection: sanitizeSelection(documentState.selection),
    })),
  setRaster: (rasterDataUrl, width, height, selection, rasterBounds) =>
    commitDocument(set, get, (documentState) => {
      const nextWidth = width ?? documentState.width;
      const nextHeight = height ?? documentState.height;
      return {
        ...documentState,
        width: nextWidth,
        height: nextHeight,
        rasterDataUrl,
        rasterBounds: rasterBounds ?? {
          x: 0,
          y: 0,
          width: nextWidth,
          height: nextHeight,
        },
        selection: selection
          ? sanitizeSelection(selection)
          : centerSelection(nextWidth, nextHeight),
      };
    }),
  setPreparedRaster: (raster) =>
    get().setRaster(
      raster.dataUrl,
      raster.width,
      raster.height,
      raster.selection,
      raster.rasterBounds,
    ),
  updateRasterDataUrl: (rasterDataUrl) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      rasterDataUrl,
    })),
  replaceRasterDataUrl: (rasterDataUrl) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      rasterDataUrl,
      rasterBounds: {
        x: 0,
        y: 0,
        width: documentState.width,
        height: documentState.height,
      },
      references: [],
    })),
  updateRasterBounds: (rasterBounds) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      rasterBounds,
    })),
  addReference: (dataUrl, width, height) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      references: [
        ...documentState.references,
        {
          id: crypto.randomUUID(),
          dataUrl,
          x: Math.max(0, documentState.selection.x),
          y: Math.max(0, documentState.selection.y),
          width,
          height,
          opacity: 0.72,
        },
      ],
    })),
  updateReference: (id, patch) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      references: documentState.references.map((reference) =>
        reference.id === id ? { ...reference, ...patch } : reference,
      ),
    })),
  beginMaskStroke: (mode, point) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      maskStrokes: [
        ...documentState.maskStrokes,
        createStroke(mode, get().brushSize, point),
      ],
    })),
  appendMaskPoint: (point) =>
    set((state) => {
      const maskStrokes = state.document.maskStrokes.map((stroke, index) =>
        index === state.document.maskStrokes.length - 1
          ? { ...stroke, points: [...stroke.points, point] }
          : stroke,
      );
      return { document: { ...state.document, maskStrokes } };
    }),
  clearMask: () =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      maskStrokes: [],
    })),
  beginControlStroke: (point) =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      controlStrokes: [
        ...documentState.controlStrokes,
        createControlStroke(
          get().brushSize,
          get().controlGuideColor,
          get().controlGuideStrength,
          point,
        ),
      ],
    })),
  appendControlPoint: (point) =>
    set((state) => {
      const controlStrokes = state.document.controlStrokes.map((stroke, index) =>
        index === state.document.controlStrokes.length - 1
          ? { ...stroke, points: [...stroke.points, point] }
          : stroke,
      );
      return { document: { ...state.document, controlStrokes } };
    }),
  clearControlGuide: () =>
    commitDocument(set, get, (documentState) => ({
      ...documentState,
      controlStrokes: [],
    })),
  updateParameter: (key, value) =>
    set((state) => ({
      parameters: applyGenerationModeDefaults(
        { ...state.parameters, [key]: value },
        state.generationMode,
      ),
    })),
  applyDirectionalOutpaintPreset: () =>
    set((state) => ({
      selectedAdapterId: ADAPTER_SDXL_INPAINT,
      controlGuideEnabled: false,
      parameters: applyDirectionalOutpaintPreset(state.parameters),
    })),
  setSelectedAdapterId: (adapterId, defaultModelId, generationDefaults) =>
    set((state) => ({
      selectedAdapterId: adapterId,
      modelId: defaultModelId ?? "",
      parameters: applyGenerationModeDefaults(
        applyAdapterGenerationDefaults(state.parameters, generationDefaults),
        state.generationMode,
      ),
    })),
  setModelSource: (modelSource) => set({ modelSource }),
  setModelId: (modelId) => set({ modelId }),
  setLocalPath: (localPath) => set({ localPath }),
  setSingleFilePath: (singleFilePath) => set({ singleFilePath }),
  setModelUrl: (modelUrl) => set({ modelUrl }),
  setControlnetModelId: (controlnetModelId) => set({ controlnetModelId }),
  setDevice: (device) => set({ device }),
  setDtype: (dtype) => set({ dtype }),
  setCurrentJob: (currentJob) => set({ currentJob }),
  setCurrentJobSelection: (currentJobSelection) => set({ currentJobSelection }),
  setGenerationNote: (generationNote) => set({ generationNote }),
  setPendingResults: (
    pendingResults,
    pendingResultBounds = null,
    pendingResultReplacesDocument = false,
  ) =>
    set({
      pendingResults,
      pendingResultBounds,
      pendingResultReplacesDocument,
      selectedResultIndex: 0,
      pluginPreview: null,
    }),
  selectResult: (selectedResultIndex) => set({ selectedResultIndex }),
  acceptSelectedResult: () => {
    const state = get();
    const selected = state.pendingResults[state.selectedResultIndex];
    if (!selected) {
      return;
    }
    commitDocument(set, get, (documentState) => {
      const bounds = state.pendingResultBounds;
      if (!bounds) {
        return {
          ...documentState,
          rasterDataUrl: selected,
          rasterBounds: {
            x: 0,
            y: 0,
            width: documentState.width,
            height: documentState.height,
          },
          maskStrokes: [],
          controlStrokes: [],
          references: [],
        };
      }
      const shiftX = -bounds.x;
      const shiftY = -bounds.y;
      return {
        ...documentState,
        width: bounds.width,
        height: bounds.height,
        rasterDataUrl: selected,
        rasterBounds: {
          x: 0,
          y: 0,
          width: bounds.width,
          height: bounds.height,
        },
        selection: shiftSelection(documentState.selection, shiftX, shiftY),
        maskStrokes: [],
        controlStrokes: [],
        references: [],
      };
    });
    set((latest) => ({
      pendingResults: [],
      pendingResultBounds: null,
      pendingResultReplacesDocument: false,
      selectedResultIndex: 0,
      pluginPreview: null,
      currentJob: null,
      currentJobSelection: null,
      canvasSelectionTarget: { kind: "frame" },
      generationNote: null,
      history: latest.history.map((item, index) =>
        index === 0 ? { ...item, acceptedImage: selected } : item,
      ),
    }));
  },
  rejectResults: () =>
    set({
      pendingResults: [],
      pendingResultBounds: null,
      pendingResultReplacesDocument: false,
      selectedResultIndex: 0,
      pluginPreview: null,
      currentJob: null,
      currentJobSelection: null,
      generationNote: null,
    }),
  setPluginPreview: (pluginPreview) => set({ pluginPreview }),
  clearPluginPreview: (toolId) =>
    set((state) =>
      !state.pluginPreview ||
      (toolId !== undefined && state.pluginPreview.toolId !== toolId)
        ? {}
        : { pluginPreview: null },
    ),
  pushHistory: (item) =>
    set((state) => ({ history: [item, ...state.history].slice(0, 40) })),
  setErrorMessage: (errorMessage) => set({ errorMessage }),
  undo: () =>
    set((state) => {
      const snapshot = state.undoStack[state.undoStack.length - 1];
      if (!snapshot) {
        return {};
      }
      return {
        document: snapshot.document,
        undoStack: state.undoStack.slice(0, -1),
        redoStack: [
          { document: structuredClone(state.document) },
          ...state.redoStack,
        ],
      };
    }),
  redo: () =>
    set((state) => {
      const snapshot = state.redoStack[0];
      if (!snapshot) {
        return {};
      }
      return {
        document: snapshot.document,
        redoStack: state.redoStack.slice(1),
        undoStack: [
          ...state.undoStack,
          { document: structuredClone(state.document) },
        ],
      };
    }),
  replaceDocument: (documentState) =>
    set({
      document: documentState,
      pendingResults: [],
      pendingResultBounds: null,
      pendingResultReplacesDocument: false,
      selectedResultIndex: 0,
      generationNote: null,
      pluginPreview: null,
      canvasSelectionTarget: { kind: "frame" },
      undoStack: [],
      redoStack: [],
    }),
}));

function commitDocument(
  set: (
    partial:
      | Partial<EditorStore>
      | ((state: EditorStore) => Partial<EditorStore>),
  ) => void,
  get: () => EditorStore,
  updater: (documentState: EditorDocument) => EditorDocument,
): void {
  const previous = get().document;
  set({
    document: updater(structuredClone(previous)),
    pluginPreview: null,
    undoStack: [
      ...get().undoStack,
      { document: structuredClone(previous) },
    ].slice(-80),
    redoStack: [],
  });
}

function createInitialDocument(): EditorDocument {
  return {
    id: crypto.randomUUID(),
    width: DEFAULT_CANVAS_WIDTH,
    height: DEFAULT_CANVAS_HEIGHT,
    rasterDataUrl: null,
    rasterBounds: null,
    selection: centerSelection(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT),
    maskStrokes: [],
    controlStrokes: [],
    references: [],
  };
}

function createInitialParameters(): GenerationParameters {
  return {
    prompt: "",
    negative_prompt: "",
    width: DEFAULT_CANVAS_WIDTH,
    height: DEFAULT_CANVAS_HEIGHT,
    steps: 28,
    guidance_scale: 7.5,
    strength: 1,
    seed: null,
    random_seed: true,
    sample_count: 1,
    scheduler: SCHEDULER_DPM_SOLVER,
    safety_checker: true,
    img2img: false,
    fill_mode: FILL_OPENCV_NS,
    correction_pipeline: [],
    inpaint_area: INPAINT_AREA_WHOLE_SELECTION,
    mask_crop_padding: 32,
    mask_blur: 0,
    outpaint_max_width: 1536,
    outpaint_max_height: 1024,
    result_mode: RESULT_MODE_GENERATED_SELECTION,
    outpaint_strategy: OUTPAINT_STRATEGY_LOCAL_CONTEXT,
    outpaint_direction: OUTPAINT_DIRECTION_RIGHT,
    outpaint_generated_size: DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
    outpaint_context_size: DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE,
    outpaint_cross_size: DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
    hf_space_overlap_percentage: 10,
    hf_space_overlap_left: true,
    hf_space_overlap_right: true,
    hf_space_overlap_top: true,
    hf_space_overlap_bottom: true,
    hf_space_resize_option: "Full",
    hf_space_custom_resize_percentage: 50,
    loras: [],
    textual_inversions: [],
  };
}

function applyAdapterGenerationDefaults(
  parameters: GenerationParameters,
  defaults: Record<string, unknown> | undefined,
): GenerationParameters {
  if (!defaults) {
    return parameters;
  }
  const baseDefaults = {
    ...parameters,
    steps: numberDefault(defaults.steps, parameters.steps),
    guidance_scale: numberDefault(
      defaults.guidance_scale,
      parameters.guidance_scale,
    ),
    strength: numberDefault(defaults.strength, parameters.strength),
    seed: nullableNumberDefault(defaults.seed, parameters.seed),
    random_seed: booleanDefault(defaults.random_seed, parameters.random_seed),
    sample_count: numberDefault(defaults.sample_count, parameters.sample_count),
    scheduler: stringDefault(defaults.scheduler, parameters.scheduler),
    safety_checker: booleanDefault(
      defaults.safety_checker,
      parameters.safety_checker,
    ),
    img2img: booleanDefault(defaults.img2img, parameters.img2img),
    fill_mode: stringDefault(defaults.fill_mode, parameters.fill_mode),
    correction_pipeline: stringArrayDefault(
      defaults.correction_pipeline,
      parameters.correction_pipeline,
    ),
    inpaint_area: stringDefault(defaults.inpaint_area, parameters.inpaint_area),
    mask_crop_padding: numberDefault(
      defaults.mask_crop_padding,
      parameters.mask_crop_padding,
    ),
    mask_blur: numberDefault(defaults.mask_blur, parameters.mask_blur),
    outpaint_max_width: numberDefault(
      defaults.outpaint_max_width,
      parameters.outpaint_max_width,
    ),
    outpaint_max_height: numberDefault(
      defaults.outpaint_max_height,
      parameters.outpaint_max_height,
    ),
    result_mode: stringDefault(defaults.result_mode, parameters.result_mode),
    outpaint_strategy: outpaintStrategyDefault(
      defaults.outpaint_strategy,
      parameters.outpaint_strategy,
    ),
    outpaint_direction: outpaintDirectionDefault(
      defaults.outpaint_direction,
      parameters.outpaint_direction,
    ),
    outpaint_generated_size: numberDefault(
      defaults.outpaint_generated_size,
      parameters.outpaint_generated_size,
    ),
    outpaint_context_size: numberDefault(
      defaults.outpaint_context_size,
      parameters.outpaint_context_size,
    ),
    outpaint_cross_size: numberDefault(
      defaults.outpaint_cross_size,
      parameters.outpaint_cross_size,
    ),
    hf_space_overlap_percentage: numberDefault(
      defaults.hf_space_overlap_percentage,
      parameters.hf_space_overlap_percentage,
    ),
    hf_space_overlap_left: booleanDefault(
      defaults.hf_space_overlap_left,
      parameters.hf_space_overlap_left,
    ),
    hf_space_overlap_right: booleanDefault(
      defaults.hf_space_overlap_right,
      parameters.hf_space_overlap_right,
    ),
    hf_space_overlap_top: booleanDefault(
      defaults.hf_space_overlap_top,
      parameters.hf_space_overlap_top,
    ),
    hf_space_overlap_bottom: booleanDefault(
      defaults.hf_space_overlap_bottom,
      parameters.hf_space_overlap_bottom,
    ),
    hf_space_resize_option: stringDefault(
      defaults.hf_space_resize_option,
      parameters.hf_space_resize_option,
    ),
    hf_space_custom_resize_percentage: numberDefault(
      defaults.hf_space_custom_resize_percentage,
      parameters.hf_space_custom_resize_percentage,
    ),
  };
  return applyPluginGenerationDefaults(baseDefaults, defaults);
}

function applyDirectionalOutpaintPreset(
  parameters: GenerationParameters,
): GenerationParameters {
  return {
    ...parameters,
    outpaint_strategy: OUTPAINT_STRATEGY_DIRECTIONAL,
    outpaint_direction: OUTPAINT_DIRECTION_RIGHT,
    outpaint_generated_size: DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
    outpaint_context_size: DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE,
    outpaint_cross_size: DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
    strength: 1,
    fill_mode: FILL_EDGE_EXTEND,
    mask_blur: 0,
    inpaint_area: INPAINT_AREA_WHOLE_SELECTION,
    result_mode: RESULT_MODE_PRESERVE_KNOWN,
    sample_count: 1,
    random_seed: false,
    scheduler: SCHEDULER_DPM_SOLVER,
  };
}

function applyGenerationModeDefaults(
  parameters: GenerationParameters,
  generationMode: GenerationMode,
): GenerationParameters {
  if (
    generationMode === GENERATION_MODE_INPAINT &&
    parameters.result_mode === RESULT_MODE_GENERATED_SELECTION
  ) {
    return { ...parameters, result_mode: RESULT_MODE_PRESERVE_KNOWN };
  }
  return parameters;
}

function applyPluginGenerationDefaults(
  parameters: GenerationParameters,
  defaults: Record<string, unknown>,
): GenerationParameters {
  const next = { ...parameters };
  for (const [key, value] of Object.entries(defaults)) {
    if (BASE_GENERATION_DEFAULT_KEYS.has(key)) {
      continue;
    }
    if (typeof next[key] === "undefined") {
      next[key] = value;
    }
  }
  return next;
}

const BASE_GENERATION_DEFAULT_KEYS = new Set([
  "prompt",
  "negative_prompt",
  "width",
  "height",
  "steps",
  "guidance_scale",
  "strength",
  "seed",
  "random_seed",
  "sample_count",
  "scheduler",
  "safety_checker",
  "img2img",
  "fill_mode",
  "correction_pipeline",
  "inpaint_area",
  "mask_crop_padding",
  "mask_blur",
  "outpaint_max_width",
  "outpaint_max_height",
  "result_mode",
  "outpaint_strategy",
  "outpaint_direction",
  "outpaint_generated_size",
  "outpaint_context_size",
  "outpaint_cross_size",
  "hf_space_overlap_percentage",
  "hf_space_overlap_left",
  "hf_space_overlap_right",
  "hf_space_overlap_top",
  "hf_space_overlap_bottom",
  "hf_space_resize_option",
  "hf_space_custom_resize_percentage",
  "loras",
  "textual_inversions",
]);

function stringDefault(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function outpaintStrategyDefault(
  value: unknown,
  fallback: OutpaintStrategy,
): OutpaintStrategy {
  if (
    value === OUTPAINT_STRATEGY_LOCAL_CONTEXT ||
    value === OUTPAINT_STRATEGY_FULL_CONTEXT_CROP ||
    value === OUTPAINT_STRATEGY_SELECTED_FRAME ||
    value === OUTPAINT_STRATEGY_WHOLE_RESIZED ||
    value === OUTPAINT_STRATEGY_DIRECTIONAL ||
    value === OUTPAINT_STRATEGY_HF_SPACE_FILL
  ) {
    return value;
  }
  return fallback;
}

function outpaintDirectionDefault(
  value: unknown,
  fallback: OutpaintDirection,
): OutpaintDirection {
  if (
    value === OUTPAINT_DIRECTION_LEFT ||
    value === OUTPAINT_DIRECTION_RIGHT ||
    value === OUTPAINT_DIRECTION_UP ||
    value === OUTPAINT_DIRECTION_DOWN ||
    value === OUTPAINT_DIRECTION_AROUND
  ) {
    return value;
  }
  return fallback;
}

function stringArrayDefault(value: unknown, fallback: string[]): string[] {
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value;
  }
  return fallback;
}

function numberDefault(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nullableNumberDefault(
  value: unknown,
  fallback: number | null,
): number | null {
  if (value === null) {
    return null;
  }
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function booleanDefault(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function centerSelection(width: number, height: number): SelectionRect {
  const selectionWidth = Math.min(DEFAULT_SELECTION_SIZE, width);
  const selectionHeight = Math.min(DEFAULT_SELECTION_SIZE, height);
  return {
    x: Math.round((width - selectionWidth) / 2),
    y: Math.round((height - selectionHeight) / 2),
    width: selectionWidth,
    height: selectionHeight,
  };
}

function sanitizeSelection(selection: SelectionRect): SelectionRect {
  return {
    x: Math.round(selection.x),
    y: Math.round(selection.y),
    width: Math.max(1, Math.min(Math.round(selection.width), 4096)),
    height: Math.max(1, Math.min(Math.round(selection.height), 4096)),
  };
}

function shiftSelection(
  selection: SelectionRect,
  shiftX: number,
  shiftY: number,
): SelectionRect {
  return {
    ...selection,
    x: selection.x + shiftX,
    y: selection.y + shiftY,
  };
}

function createStroke(
  mode: MaskStrokeMode,
  size: number,
  point: Point,
): MaskStroke {
  return {
    id: crypto.randomUUID(),
    mode: mode === STROKE_ERASE ? STROKE_ERASE : STROKE_PAINT,
    size,
    points: [point],
  };
}

function createControlStroke(
  size: number,
  color: string,
  strength: number,
  point: Point,
): ControlStroke {
  return {
    id: crypto.randomUUID(),
    size,
    color,
    strength,
    points: [point],
  };
}
