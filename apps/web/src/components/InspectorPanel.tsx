import {
  Check,
  Captions,
  ChevronLeft,
  ChevronDown,
  ChevronRight,
  Eraser,
  Frame,
  Hand,
  ListOrdered,
  MousePointer2,
  Paintbrush,
  Palette,
  Play,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
  Square,
  TextSearch,
  Trash2,
  WandSparkles,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
  CONTROLNET_GUIDE_UI_ENABLED,
  CONTROL_GUIDE_MASK_MODE_OPTIONS,
  CONTROL_GUIDE_COLORS,
  DIRECTIONAL_OUTPAINT_MIN_SIZE,
  MAX_CONTROL_GUIDE_STRENGTH,
  MAX_ERASER_HARDNESS,
  MAX_BRUSH_SIZE,
  MIN_CONTROL_GUIDE_STRENGTH,
  MIN_ERASER_HARDNESS,
  MIN_BRUSH_SIZE,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  OUTPAINT_DIRECTION_OPTIONS,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_OPTIONS,
  TOOL_CONTROL_GUIDE,
  TOOL_ERASE,
  TOOL_INPAINT_MASK,
  TOOL_OUTPAINT_FRAME,
  TOOL_PAN,
  TOOL_SELECT,
  WORKSPACE_MODE_EXPAND_IMAGE,
  WORKSPACE_MODE_FREE_EDIT,
  PLUGIN_TOOL_TARGET_IMAGE,
  pluginToolIdFromEditorTool,
  type ControlGuideMaskMode,
  type EditorTool,
  type GenerationMode,
  type OutpaintStrategy,
  type WorkspaceMode,
} from "../constants/domain";
import { useMutation } from "@tanstack/react-query";
import type {
  CanvasSelectionTarget,
  ControlSchema,
  DocumentBounds,
  EditorDocument,
  GenerationParameters,
  PluginImagePreview,
  PluginActionResult,
  PluginToolInfo,
  SelectionRect,
  ViewportState,
} from "../domain/types";
import { useModelLoader } from "../hooks/useModelLoader";
import { useOutpaintJob } from "../hooks/useOutpaintJob";
import { useStudioQueries } from "../hooks/useStudioQueries";
import {
  disablePlugin,
  enablePlugin,
  runPluginAction,
  unloadModel,
} from "../lib/apiClient";
import { renderPluginActionInput } from "../lib/canvasRender";
import {
  adapterIdForControlGuideMode,
  controlnetModelIdForAdapter,
  controlnetModelIdForAdapterValue,
} from "../lib/controlGuideMode";
import { controlOptionDetailsFor } from "../lib/controlHelp";
import {
  isModelSourceReady,
  type ModelSourceValues,
} from "../lib/modelSources";
import {
  ONBOARDING_TARGET_GENERATE_BUTTON,
  ONBOARDING_TARGET_PROMPT,
  ONBOARDING_TARGET_SETUP_BUTTON,
  ONBOARDING_TARGET_TOPBAR_MODEL,
} from "../lib/onboardingTour";
import { adapterIdForWorkspaceMode, isExpandImageAdapter } from "../lib/workspaceMode";
import { useEditorStore } from "../store/editorStore";
import { CorrectionPipelineSection } from "./CorrectionPipelineSection";
import { GenerationControls } from "./GenerationControls";
import { ModelSetupDialog } from "./ModelSetupDialog";
import { NumberStepper } from "./NumberStepper";
import { OnboardingTour } from "./OnboardingTour";
import { SchemaControl } from "./SchemaControl";
import { SelectControl } from "./SelectControl";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

interface ToolContext {
  title: string;
  icon: LucideIcon;
}

const CONTROL_GUIDE_DISABLED_ADAPTERS = new Set([
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
  "sdxl-fill-ip-refine",
]);

const WORKFLOW_CONTROL_IDS = new Set([
  "outpaint_direction",
  "outpaint_generated_size",
  "outpaint_context_size",
  "outpaint_cross_size",
  "hf_space_resize_option",
  "hf_space_custom_resize_percentage",
  "hf_space_overlap_percentage",
  "hf_space_overlap_left",
  "hf_space_overlap_right",
  "hf_space_overlap_top",
  "hf_space_overlap_bottom",
]);

const FREE_EDIT_WORKFLOW_CONTROL_IDS = new Set([
  "outpaint_direction",
  "outpaint_generated_size",
  "outpaint_context_size",
  "outpaint_cross_size",
  "hf_space_resize_option",
  "hf_space_custom_resize_percentage",
  "hf_space_overlap_left",
  "hf_space_overlap_right",
  "hf_space_overlap_top",
  "hf_space_overlap_bottom",
]);

const FREE_EDIT_OUTPAINT_STRATEGY_OPTIONS = OUTPAINT_STRATEGY_OPTIONS.filter(
  (option) =>
    option.id !== OUTPAINT_STRATEGY_DIRECTIONAL &&
    option.id !== OUTPAINT_STRATEGY_HF_SPACE_FILL,
);

const FIXED_EXPAND_RENDER_SCALE_OPTIONS = [
  { id: "full", label: "Full" },
  { id: "balanced", label: "Balanced" },
  { id: "safe", label: "Safe" },
  { id: "custom", label: "Custom" },
];

export function InspectorPanel() {
  const [setupOpen, setSetupOpen] = useState(false);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [loraText, setLoraText] = useState("");
  const [textualInversionText, setTextualInversionText] = useState("");
  const activeModelSelectionSynced = useRef(false);

  const {
    runtimeQuery,
    adaptersQuery,
    modelsQuery,
    persistentStateQuery,
    pluginsQuery,
    pluginActionsQuery,
    pluginToolsQuery,
  } = useStudioQueries();

  const documentState = useEditorStore((state) => state.document);
  const selectedAdapterId = useEditorStore((state) => state.selectedAdapterId);
  const modelSource = useEditorStore((state) => state.modelSource);
  const modelId = useEditorStore((state) => state.modelId);
  const localPath = useEditorStore((state) => state.localPath);
  const singleFilePath = useEditorStore((state) => state.singleFilePath);
  const modelUrl = useEditorStore((state) => state.modelUrl);
  const device = useEditorStore((state) => state.device);
  const dtype = useEditorStore((state) => state.dtype);
  const parameters = useEditorStore((state) => state.parameters);
  const currentJob = useEditorStore((state) => state.currentJob);
  const pendingResults = useEditorStore((state) => state.pendingResults);
  const brushSize = useEditorStore((state) => state.brushSize);
  const eraserHardness = useEditorStore((state) => state.eraserHardness);
  const controlGuideEnabled = useEditorStore(
    (state) => state.controlGuideEnabled,
  );
  const controlGuideColor = useEditorStore((state) => state.controlGuideColor);
  const controlGuideStrength = useEditorStore(
    (state) => state.controlGuideStrength,
  );
  const controlGuideMaskMode = useEditorStore(
    (state) => state.controlGuideMaskMode,
  );
  const controlnetModelId = useEditorStore((state) => state.controlnetModelId);
  const workspaceMode = useEditorStore((state) => state.workspaceMode);
  const generationMode = useEditorStore((state) => state.generationMode);
  const tool = useEditorStore((state) => state.tool);
  const canvasSelectionTarget = useEditorStore(
    (state) => state.canvasSelectionTarget,
  );
  const viewport = useEditorStore((state) => state.viewport);
  const setTool = useEditorStore((state) => state.setTool);
  const setSelectedAdapterId = useEditorStore(
    (state) => state.setSelectedAdapterId,
  );
  const setModelSource = useEditorStore((state) => state.setModelSource);
  const setModelId = useEditorStore((state) => state.setModelId);
  const setLocalPath = useEditorStore((state) => state.setLocalPath);
  const setSingleFilePath = useEditorStore((state) => state.setSingleFilePath);
  const setModelUrl = useEditorStore((state) => state.setModelUrl);
  const setDevice = useEditorStore((state) => state.setDevice);
  const setDtype = useEditorStore((state) => state.setDtype);
  const setGenerationMode = useEditorStore((state) => state.setGenerationMode);
  const updateParameter = useEditorStore((state) => state.updateParameter);
  const setSelection = useEditorStore((state) => state.setSelection);
  const setBrushSize = useEditorStore((state) => state.setBrushSize);
  const setEraserHardness = useEditorStore(
    (state) => state.setEraserHardness,
  );
  const setControlGuideEnabled = useEditorStore(
    (state) => state.setControlGuideEnabled,
  );
  const setControlGuideColor = useEditorStore(
    (state) => state.setControlGuideColor,
  );
  const setControlGuideStrength = useEditorStore(
    (state) => state.setControlGuideStrength,
  );
  const setControlGuideMaskMode = useEditorStore(
    (state) => state.setControlGuideMaskMode,
  );
  const setControlnetModelId = useEditorStore(
    (state) => state.setControlnetModelId,
  );
  const setWorkspaceMode = useEditorStore((state) => state.setWorkspaceMode);
  const clearMask = useEditorStore((state) => state.clearMask);
  const clearControlGuide = useEditorStore((state) => state.clearControlGuide);
  const updateRasterDataUrl = useEditorStore(
    (state) => state.updateRasterDataUrl,
  );
  const updateReference = useEditorStore((state) => state.updateReference);
  const setPluginPreview = useEditorStore((state) => state.setPluginPreview);
  const clearPluginPreview = useEditorStore(
    (state) => state.clearPluginPreview,
  );
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage);

  const adapters = useMemo(
    () => adaptersQuery.data ?? [],
    [adaptersQuery.data],
  );
  const selectedAdapter = useMemo(
    () =>
      adapters.find((adapter) => adapter.id === selectedAdapterId) ??
      adapters[0],
    [adapters, selectedAdapterId],
  );
  const loadedModel = modelsQuery.data?.find(
    (model) => model.adapter_id === selectedAdapterId,
  );
  const activeModel = modelsQuery.data?.find((model) => model.loaded);
  const pluginTools = pluginToolsQuery.data ?? [];
  const activePluginToolId = pluginToolIdFromEditorTool(tool);
  const activePluginTool =
    pluginTools.find((pluginTool) => pluginTool.id === activePluginToolId) ??
    null;
  const toolContext = getToolContext(tool, activePluginTool);
  const expandImageActive = workspaceMode === WORKSPACE_MODE_EXPAND_IMAGE;
  const expandImageAdapterActive = isExpandImageAdapter(selectedAdapterId);
  const generationToolPanelActive = !activePluginTool;
  const selectFrameActive =
    workspaceMode === WORKSPACE_MODE_FREE_EDIT &&
    tool === TOOL_SELECT &&
    generationMode === GENERATION_MODE_OUTPAINT &&
    canvasSelectionTarget.kind === "frame";
  const directionalOutpaintActive =
    generationMode === GENERATION_MODE_OUTPAINT &&
    parameters.outpaint_strategy === OUTPAINT_STRATEGY_DIRECTIONAL;
  const hfSpaceFillActive =
    generationMode === GENERATION_MODE_OUTPAINT &&
    parameters.outpaint_strategy === OUTPAINT_STRATEGY_HF_SPACE_FILL;
  const controlGuideDisabledForAdapter =
    expandImageActive || CONTROL_GUIDE_DISABLED_ADAPTERS.has(selectedAdapterId);
  const controlGuideUnavailable =
    !CONTROLNET_GUIDE_UI_ENABLED || controlGuideDisabledForAdapter;
  const controlGuideAvailable = !controlGuideUnavailable;
  const generationPanelActive =
    generationToolPanelActive &&
    (expandImageActive ||
      tool === TOOL_OUTPAINT_FRAME ||
      tool === TOOL_INPAINT_MASK ||
      (CONTROLNET_GUIDE_UI_ENABLED && tool === TOOL_CONTROL_GUIDE) ||
      selectFrameActive);
  const inspectorClassName = panelCollapsed
    ? "inspector-panel inspector-panel-collapsed"
    : generationPanelActive
      ? "inspector-panel"
      : "inspector-panel inspector-panel-contextual";
  const modelLoaded = Boolean(loadedModel?.loaded);
  const sourceValues: ModelSourceValues = {
    modelId,
    localPath,
    singleFilePath,
    modelUrl,
  };
  const sourceReady = isModelSourceReady(
    selectedAdapter,
    modelSource,
    sourceValues,
  );
  const generationActionLabel =
    expandImageActive
      ? "Expand image"
      : generationMode === GENERATION_MODE_INPAINT
      ? "Generate inpaint"
      : "Generate outpaint";
  const hiddenGenerationControlIds =
    expandImageActive || directionalOutpaintActive
      ? WORKFLOW_CONTROL_IDS
      : generationMode === GENERATION_MODE_OUTPAINT
        ? FREE_EDIT_WORKFLOW_CONTROL_IDS
      : undefined;
  const controlGuideTargetAdapterId = adapterIdForControlGuideMode(
    true,
    selectedAdapterId,
    adapters,
  );
  const controlGuideLabel =
    selectedAdapter?.capabilities.controlnet || controlGuideTargetAdapterId
      ? "ControlNet sketch"
      : "Native sketch";
  const effectiveControlnetModelId = controlnetModelIdForAdapterValue(
    selectedAdapter,
    controlnetModelId,
  );

  const selectAdapterById = (adapterId: string) => {
    const adapter = adapters.find((item) => item.id === adapterId);
    if (!adapter) {
      return;
    }
    setSelectedAdapterId(
      adapterId,
      adapter.default_model_id,
      adapter.generation_defaults,
    );
  };

  const setOutpaintStrategy = (strategy: OutpaintStrategy) => {
    if (
      strategy === OUTPAINT_STRATEGY_DIRECTIONAL ||
      strategy === OUTPAINT_STRATEGY_HF_SPACE_FILL
    ) {
      return;
    }
    updateParameter("outpaint_strategy", strategy);
  };

  const changeWorkspaceMode = (mode: WorkspaceMode) => {
    if (mode !== WORKSPACE_MODE_EXPAND_IMAGE) {
      setWorkspaceMode(mode);
      return;
    }
    const adapterId = adapterIdForWorkspaceMode(mode, selectedAdapterId, adapters);
    if (adapterId !== selectedAdapterId) {
      selectAdapterById(adapterId);
    }
    setWorkspaceMode(mode);
  };

  const setControlGuideMode = (enabled: boolean) => {
    if (!CONTROLNET_GUIDE_UI_ENABLED) {
      setControlGuideEnabled(false);
      if (tool === TOOL_CONTROL_GUIDE) {
        setGenerationMode(generationMode);
      }
      return;
    }
    setControlGuideEnabled(enabled);
    const adapterId = adapterIdForControlGuideMode(
      enabled,
      selectedAdapterId,
      adapters,
    );
    if (adapterId) {
      setControlnetModelId(
        controlnetModelIdForAdapter(
          adapters.find((adapter) => adapter.id === adapterId),
        ) ?? controlnetModelId,
      );
      selectAdapterById(adapterId);
    }
    if (enabled) {
      clearPluginPreview();
      setTool(TOOL_CONTROL_GUIDE);
      return;
    }
    if (tool === TOOL_CONTROL_GUIDE) {
      setGenerationMode(generationMode);
    }
  };

  useEffect(() => {
    if (
      activeModelSelectionSynced.current ||
      !activeModel ||
      adapters.length === 0
    ) {
      return;
    }
    activeModelSelectionSynced.current = true;
    if (activeModel.adapter_id === selectedAdapterId) {
      return;
    }
    const adapter = adapters.find((item) => item.id === activeModel.adapter_id);
    if (!adapter) {
      return;
    }
    setSelectedAdapterId(
      adapter.id,
      activeModel.model_id ?? adapter.default_model_id,
      adapter.generation_defaults,
    );
  }, [activeModel, adapters, selectedAdapterId, setSelectedAdapterId]);

  useEffect(() => {
    if (!controlGuideUnavailable) {
      return;
    }
    if (controlGuideEnabled) {
      setControlGuideEnabled(false);
    }
    if (tool === TOOL_CONTROL_GUIDE) {
      setGenerationMode(generationMode);
    }
  }, [
    controlGuideEnabled,
    controlGuideUnavailable,
    generationMode,
    setControlGuideEnabled,
    setGenerationMode,
    tool,
  ]);

  useEffect(() => {
    if (expandImageActive && !expandImageAdapterActive) {
      setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
    }
  }, [
    expandImageActive,
    expandImageAdapterActive,
    setWorkspaceMode,
  ]);

  const loadMutation = useModelLoader({
    selectedAdapterId,
    selectedAdapter,
    modelSource,
    sourceValues,
    device,
    dtype,
    safetyChecker: Boolean(parameters.safety_checker),
    controlnetModelId: effectiveControlnetModelId,
    loraText,
    textualInversionText,
    onLoaded: async () => {
      await modelsQuery.refetch();
      await runtimeQuery.refetch();
      await persistentStateQuery.refetch();
      setSetupOpen(false);
    },
    onChanged: async () => {
      await modelsQuery.refetch();
      await runtimeQuery.refetch();
      await persistentStateQuery.refetch();
    },
  });

  const { generateMutation, running, cancelRunningJob } = useOutpaintJob({
    selectedAdapter,
    loraText,
    textualInversionText,
    onPersistentStateRefresh: () => {
      void persistentStateQuery.refetch();
    },
  });

  const unloadMutation = useMutation({
    mutationFn: unloadModel,
    onSuccess: async () => {
      await modelsQuery.refetch();
      await persistentStateQuery.refetch();
    },
    onError: (error) => {
      setErrorMessage(
        error instanceof Error ? error.message : "Model unload failed.",
      );
    },
  });

  const pluginMutation = useMutation({
    mutationFn: ({ pluginId, enabled }: { pluginId: string; enabled: boolean }) =>
      enabled ? enablePlugin(pluginId) : disablePlugin(pluginId),
    onSuccess: async () => {
      await pluginsQuery.refetch();
      await adaptersQuery.refetch();
      await pluginActionsQuery.refetch();
      await pluginToolsQuery.refetch();
      await modelsQuery.refetch();
      await persistentStateQuery.refetch();
    },
    onError: (error) => {
      setErrorMessage(
        error instanceof Error ? error.message : "Plugin update failed.",
      );
    },
  });

  const onboardingProgress = {
    modelLoaded,
    modelLoading: loadMutation.isPending,
    imageLoaded: Boolean(documentState.rasterDataUrl),
    outpaintReady:
      modelLoaded &&
      Boolean(documentState.rasterDataUrl) &&
      workspaceMode === WORKSPACE_MODE_FREE_EDIT &&
      generationMode === GENERATION_MODE_OUTPAINT,
    generationStarted:
      Boolean(currentJob) || pendingResults.length > 0 || generateMutation.isPending,
  };
  const onboardingGenerationDisabled =
    !modelLoaded ||
    !documentState.rasterDataUrl ||
    running ||
    generateMutation.isPending;
  const startOnboardingModelLoad = () => {
    setSetupOpen(true);
    if (!sourceReady || loadMutation.isPending || modelLoaded) {
      return;
    }
    loadMutation.mutate();
  };
  const useOnboardingOutpaint = () => {
    setPanelCollapsed(false);
    setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
    setGenerationMode(GENERATION_MODE_OUTPAINT);
    setTool(TOOL_OUTPAINT_FRAME);
  };
  const generateFromOnboarding = () => {
    if (onboardingGenerationDisabled) {
      return;
    }
    generateMutation.mutate();
  };

  return (
    <>
      <header className="studio-topbar">
        <div className="topbar-project">
          <span className="topbar-label">Project</span>
          <strong>{documentState.id.slice(0, 8)}</strong>
        </div>
        <div className="topbar-model" data-tour-id={ONBOARDING_TARGET_TOPBAR_MODEL}>
          <span className="topbar-label">Profile</span>
          <strong>
            {selectedAdapter?.label ?? selectedAdapterId}
            <span
              className={modelLoaded ? "topbar-ready-dot" : "topbar-idle-dot"}
            />
            <span>{modelLoaded ? "Ready" : "Not loaded"}</span>
          </strong>
        </div>
        <div className="topbar-actions">
          <Button
            type="button"
            variant="secondary"
            size="compact"
            data-tour-id={ONBOARDING_TARGET_SETUP_BUTTON}
            onClick={() => setSetupOpen(true)}
          >
            <Settings2 size={16} />
            Setup
          </Button>
          <Button
            type="button"
            variant="primary"
            size="compact"
            className="topbar-generate-button"
            data-tour-id={ONBOARDING_TARGET_GENERATE_BUTTON}
            disabled={!modelLoaded || running || generateMutation.isPending}
            onClick={() => generateMutation.mutate()}
          >
            <WandSparkles size={16} />
            {running || generateMutation.isPending
              ? "Generating"
              : generationActionLabel}
          </Button>
        </div>
      </header>

      <aside className={inspectorClassName}>
        {panelCollapsed ? (
          <>
            <Button
              type="button"
              variant="ghost"
              size="smallIcon"
              className="inspector-collapse-button"
              onClick={() => setPanelCollapsed(false)}
              title="Expand inspector"
            >
              <ChevronLeft size={16} />
            </Button>
            <div className="collapsed-inspector-content">
              {expandImageActive && generationToolPanelActive ? (
                <Frame size={19} />
              ) : (
                <toolContext.icon size={19} />
              )}
              {generationPanelActive ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="smallIcon"
                  className="mini-action-button"
                  disabled={
                    !modelLoaded || running || generateMutation.isPending
                  }
                  onClick={() => generateMutation.mutate()}
                  title={generationActionLabel}
                >
                  <WandSparkles size={17} />
                </Button>
              ) : null}
              <Button
                type="button"
                variant="secondary"
                size="smallIcon"
                className="mini-action-button"
                onClick={() => setSetupOpen(true)}
                title="Model setup"
              >
                <Settings2 size={17} />
              </Button>
            </div>
          </>
        ) : (
          <>
            <div className="inspector-panel-header">
              <div className="tool-panel-header">
                {expandImageActive && generationToolPanelActive ? (
                  <Frame size={17} />
                ) : (
                  <toolContext.icon size={17} />
                )}
                <strong>
                  {expandImageActive && generationToolPanelActive
                    ? "Expand image"
                    : toolContext.title}
                </strong>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="smallIcon"
                className="inspector-collapse-button"
                onClick={() => setPanelCollapsed(true)}
                title="Collapse inspector"
              >
                <ChevronRight size={16} />
              </Button>
            </div>
            <ScrollArea className="inspector-scroll">
              {generationPanelActive ? (
                <Tabs defaultValue="generate" className="inspector-tabs">
                  <TabsList>
                    <TabsTrigger value="generate">
                      <WandSparkles size={14} />
                      Generate
                    </TabsTrigger>
                    <TabsTrigger value="workflow">
                      <ListOrdered size={14} />
                      Workflow
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="generate">
                    <GenerationWorkflowSection
                      generationMode={generationMode}
                      outpaintStrategy={parameters.outpaint_strategy}
                      parameters={parameters}
                      controlGuideEnabled={controlGuideEnabled}
                      controlGuideLabel={controlGuideLabel}
                      hfSpaceFillActive={hfSpaceFillActive}
                      controlGuideAvailable={controlGuideAvailable}
                      onWorkspaceModeChange={changeWorkspaceMode}
                      onOutpaintStrategyChange={setOutpaintStrategy}
                      onParameterChange={updateParameter}
                      onControlGuideEnabledChange={setControlGuideMode}
                      workspaceMode={workspaceMode}
                      fixedExpandAvailable={expandImageAdapterActive}
                    />
                    {expandImageActive || directionalOutpaintActive ? null : (
                      <ToolContextSections
                        tool={tool}
                        generationMode={generationMode}
                        documentState={documentState}
                        canvasSelectionTarget={canvasSelectionTarget}
                        viewport={viewport}
                        brushSize={brushSize}
                        eraserHardness={eraserHardness}
                        controlGuideColor={controlGuideColor}
                        controlGuideStrength={controlGuideStrength}
                        controlGuideMaskMode={controlGuideMaskMode}
                        onSelectionChange={setSelection}
                        onBrushSizeChange={setBrushSize}
                        onEraserHardnessChange={setEraserHardness}
                        onControlGuideColorChange={setControlGuideColor}
                        onControlGuideStrengthChange={setControlGuideStrength}
                        onControlGuideMaskModeChange={setControlGuideMaskMode}
                        onClearMask={clearMask}
                        onClearControlGuide={clearControlGuide}
                      />
                    )}
                    <div data-tour-id={ONBOARDING_TARGET_PROMPT}>
                      <GenerationControls
                        controls={selectedAdapter?.generation_controls ?? []}
                        loadControls={selectedAdapter?.load_controls ?? []}
                        generationMode={generationMode}
                        parameters={parameters}
                        loraText={loraText}
                        textualInversionText={textualInversionText}
                        hiddenControlIds={hiddenGenerationControlIds}
                        onParameterChange={updateParameter}
                        onLoraTextChange={setLoraText}
                        onTextualInversionTextChange={setTextualInversionText}
                      />
                    </div>
                  </TabsContent>
                  <TabsContent value="workflow">
                    <CorrectionPipelineSection
                      postprocessors={selectedAdapter?.postprocessors ?? []}
                      controls={selectedAdapter?.generation_controls ?? []}
                      parameters={parameters}
                      onParameterChange={updateParameter}
                    />
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="inspector-tool-panel">
                  {activePluginTool ? (
                    <PluginToolPanel
                      key={activePluginTool.id}
                      tool={activePluginTool}
                      documentState={documentState}
                      canvasSelectionTarget={canvasSelectionTarget}
                      onSelectionChange={setSelection}
                      onRasterDataUrlChange={updateRasterDataUrl}
                      onReferenceChange={updateReference}
                      onPluginPreviewChange={setPluginPreview}
                      onPluginPreviewClear={clearPluginPreview}
                      onError={setErrorMessage}
                    />
                  ) : (
                    <ToolContextSections
                      tool={tool}
                      generationMode={generationMode}
                      documentState={documentState}
                      canvasSelectionTarget={canvasSelectionTarget}
                      viewport={viewport}
                      brushSize={brushSize}
                      eraserHardness={eraserHardness}
                      controlGuideColor={controlGuideColor}
                      controlGuideStrength={controlGuideStrength}
                      controlGuideMaskMode={controlGuideMaskMode}
                      onSelectionChange={setSelection}
                      onBrushSizeChange={setBrushSize}
                      onEraserHardnessChange={setEraserHardness}
                      onControlGuideColorChange={setControlGuideColor}
                      onControlGuideStrengthChange={setControlGuideStrength}
                      onControlGuideMaskModeChange={setControlGuideMaskMode}
                      onClearMask={clearMask}
                      onClearControlGuide={clearControlGuide}
                    />
                  )}
                </div>
              )}
            </ScrollArea>

            {generationPanelActive ? (
              <section className="panel-section sticky-actions">
                {currentJob ? (
                  <div className="job-block">
                    <div className="job-row">
                      <span>
                        {currentJob.status}: {currentJob.message}
                      </span>
                      <span>{Math.round(currentJob.progress * 100)}%</span>
                    </div>
                    <Progress value={Math.round(currentJob.progress * 100)} />
                  </div>
                ) : null}
                <Button
                  type="button"
                  variant="primary"
                  size="large"
                  className="sticky-generate-action"
                  disabled={
                    !modelLoaded || running || generateMutation.isPending
                  }
                  onClick={() => generateMutation.mutate()}
                >
                  <WandSparkles size={17} />
                  {running || generateMutation.isPending
                    ? "Generating"
                    : generationActionLabel}
                </Button>
                {running ? (
                  <Button
                    type="button"
                    variant="danger"
                    size="large"
                    className="sticky-icon-action"
                    title="Cancel job"
                    aria-label="Cancel job"
                    onClick={cancelRunningJob}
                  >
                    <Square size={15} />
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="secondary"
                    size="large"
                    className="sticky-icon-action"
                    title="Retry generation"
                    aria-label="Retry generation"
                    disabled={!modelLoaded}
                    onClick={() => generateMutation.mutate()}
                  >
                    <Play size={15} />
                  </Button>
                )}
              </section>
            ) : null}
          </>
        )}
      </aside>

      {setupOpen ? (
        <ModelSetupDialog
          adapters={adapters}
          selectedAdapter={selectedAdapter}
          runtime={runtimeQuery.data}
          selectedAdapterId={selectedAdapterId}
          modelSource={modelSource}
          sourceValues={sourceValues}
          device={device}
          dtype={dtype}
          controlnetModelId={effectiveControlnetModelId}
          safetyChecker={Boolean(parameters.safety_checker)}
          loraText={loraText}
          textualInversionText={textualInversionText}
          activeModel={activeModel}
          loading={loadMutation.isPending}
          loadProgress={loadMutation.loadProgress}
          loadDisabled={!sourceReady}
          unloading={unloadMutation.isPending}
          cancelPending={loadMutation.cancelPending}
          plugins={pluginsQuery.data ?? []}
          pluginControls={selectedAdapter?.generation_controls ?? []}
          pluginPostprocessors={selectedAdapter?.postprocessors ?? []}
          parameters={parameters}
          pendingPluginId={
            pluginMutation.isPending && pluginMutation.variables
              ? pluginMutation.variables.pluginId
              : null
          }
          onClose={() => setSetupOpen(false)}
          onAdapterChange={(adapterId) => {
            const adapter = adapters.find((item) => item.id === adapterId);
            const adapterControlnetModelId = controlnetModelIdForAdapter(adapter);
            setSelectedAdapterId(
              adapterId,
              adapter?.default_model_id ?? null,
              adapter?.generation_defaults,
            );
            if (adapterControlnetModelId) {
              setControlnetModelId(adapterControlnetModelId);
            }
          }}
          onModelSourceChange={setModelSource}
          onModelIdChange={setModelId}
          onLocalPathChange={setLocalPath}
          onSingleFilePathChange={setSingleFilePath}
          onModelUrlChange={setModelUrl}
          onControlnetModelIdChange={setControlnetModelId}
          onDeviceChange={setDevice}
          onDtypeChange={setDtype}
          onSafetyCheckerChange={(value) =>
            updateParameter("safety_checker", value)
          }
          onLoraTextChange={setLoraText}
          onTextualInversionTextChange={setTextualInversionText}
          onLoad={() => loadMutation.mutate()}
          onUnload={() => {
            if (activeModel?.loaded) {
              unloadMutation.mutate(activeModel.adapter_id);
            }
          }}
          onCancelLoad={loadMutation.cancelLoad}
          onPluginToggle={(plugin) =>
            pluginMutation.mutate({
              pluginId: plugin.id,
              enabled: !plugin.enabled,
            })
          }
          onParameterChange={updateParameter}
        />
      ) : null}
      <OnboardingTour
        progress={onboardingProgress}
        modelSetupOpen={setupOpen}
        modelLoadProgress={loadMutation.loadProgress}
        imageBounds={documentState.rasterBounds}
        outpaintFrame={documentState.selection}
        viewport={viewport}
        modelLoadDisabled={!sourceReady}
        generationDisabled={onboardingGenerationDisabled}
        onOpenSetup={() => setSetupOpen(true)}
        onLoadModel={startOnboardingModelLoad}
        onUseOutpaint={useOnboardingOutpaint}
        onGenerate={generateFromOnboarding}
      />
    </>
  );
}

function OutpaintMethodSwitch({
  value,
  fixedExpandAvailable,
  onChange,
}: {
  value: WorkspaceMode;
  fixedExpandAvailable: boolean;
  onChange: (mode: WorkspaceMode) => void;
}) {
  return (
    <div className="outpaint-method-switch" aria-label="Outpaint method">
      <button
        type="button"
        className={
          value === WORKSPACE_MODE_FREE_EDIT
            ? "outpaint-method-option outpaint-method-option-active"
            : "outpaint-method-option"
        }
        aria-pressed={value === WORKSPACE_MODE_FREE_EDIT}
        onClick={() => onChange(WORKSPACE_MODE_FREE_EDIT)}
      >
        <Paintbrush size={15} />
        <span>Free frame</span>
      </button>
      <button
        type="button"
        className={
          value === WORKSPACE_MODE_EXPAND_IMAGE
            ? "outpaint-method-option outpaint-method-option-active"
            : "outpaint-method-option"
        }
        aria-pressed={value === WORKSPACE_MODE_EXPAND_IMAGE}
        disabled={!fixedExpandAvailable}
        title={
          fixedExpandAvailable
            ? "Use fixed SDXL Fill expansion."
            : "Load the SDXL Fill profile to use fixed expansion."
        }
        onClick={() => onChange(WORKSPACE_MODE_EXPAND_IMAGE)}
      >
        <Frame size={15} />
        <span>Fixed expand</span>
      </button>
    </div>
  );
}

function PluginToolPanel({
  tool,
  documentState,
  canvasSelectionTarget,
  onSelectionChange,
  onRasterDataUrlChange,
  onReferenceChange,
  onPluginPreviewChange,
  onPluginPreviewClear,
  onError,
}: {
  tool: PluginToolInfo;
  documentState: EditorDocument;
  canvasSelectionTarget: CanvasSelectionTarget;
  onSelectionChange: (selection: SelectionRect) => void;
  onRasterDataUrlChange: (dataUrl: string) => void;
  onReferenceChange: (
    id: string,
    patch: Partial<EditorDocument["references"][number]>,
  ) => void;
  onPluginPreviewChange: (preview: PluginImagePreview) => void;
  onPluginPreviewClear: (toolId?: string) => void;
  onError: (message: string | null) => void;
}) {
  const [controlValues, setControlValues] = useState<Record<string, unknown>>(
    {},
  );
  const [result, setResult] = useState<{
    result: PluginActionResult;
    target: CanvasSelectionTarget;
  } | null>(null);
  const [previewPending, setPreviewPending] = useState(false);
  const previewSequence = useRef(0);
  const actionTarget = useMemo(
    () => getPluginToolActionTarget(tool, canvasSelectionTarget),
    [canvasSelectionTarget, tool],
  );
  const processingDisabled = mutationTargetDisabled(tool, actionTarget);
  const livePreview =
    tool.live_preview && tool.target === PLUGIN_TOOL_TARGET_IMAGE;
  const controlPayload = useMemo(
    () => pluginToolControlPayload(tool, controlValues),
    [tool, controlValues],
  );
  const actionTargetKey = pluginActionTargetKey(actionTarget);

  const mutation = useMutation({
    mutationFn: async () => {
      const target = getPluginToolActionTarget(tool, canvasSelectionTarget);
      if (!target) {
        throw new Error("Select an uploaded image on the canvas first.");
      }
      return runToolAction(tool, documentState, target, controlPayload);
    },
    onSuccess: (nextResult) => {
      setResult(nextResult);
      onError(null);
    },
    onError: (error) => {
      onError(error instanceof Error ? error.message : "Plugin tool failed.");
    },
  });

  useEffect(() => {
    if (!livePreview) {
      return;
    }
    return () => onPluginPreviewClear(tool.id);
  }, [livePreview, onPluginPreviewClear, tool.id]);

  useEffect(() => {
    if (!livePreview) {
      return;
    }
    if (!actionTarget || processingDisabled) {
      const sequence = previewSequence.current + 1;
      previewSequence.current = sequence;
      const resetTimer = window.setTimeout(() => {
        if (previewSequence.current !== sequence) {
          return;
        }
        setPreviewPending(false);
        setResult(null);
        onPluginPreviewClear(tool.id);
      }, 0);
      return () => window.clearTimeout(resetTimer);
    }
    const target = actionTarget;
    const sequence = previewSequence.current + 1;
    previewSequence.current = sequence;
    const timer = window.setTimeout(() => {
      setPreviewPending(true);
      runToolAction(tool, documentState, target, controlPayload)
        .then((nextResult) => {
          if (previewSequence.current !== sequence) {
            return;
          }
          setResult(nextResult);
          if (nextResult.result.image) {
            onPluginPreviewChange({
              toolId: tool.id,
              image: nextResult.result.image,
              target,
            });
          } else {
            onPluginPreviewClear(tool.id);
          }
          setPreviewPending(false);
          onError(null);
        })
        .catch((error) => {
          if (previewSequence.current !== sequence) {
            return;
          }
          setResult(null);
          setPreviewPending(false);
          onPluginPreviewClear(tool.id);
          onError(error instanceof Error ? error.message : "Plugin tool failed.");
        });
    }, 240);
    return () => window.clearTimeout(timer);
  }, [
    actionTarget,
    actionTargetKey,
    controlPayload,
    documentState,
    livePreview,
    onError,
    onPluginPreviewChange,
    onPluginPreviewClear,
    processingDisabled,
    tool,
  ]);

  const applyLivePreview = () => {
    if (!result?.result.image || !canApplyPluginImageResult(result.target)) {
      return;
    }
    previewSequence.current += 1;
    applyPluginImageResult(
      result.result.image,
      result.target,
      onRasterDataUrlChange,
      onReferenceChange,
    );
    setResult(null);
    setControlValues({});
    setPreviewPending(false);
    onPluginPreviewClear(tool.id);
    onError(null);
  };

  const resetLivePreview = () => {
    previewSequence.current += 1;
    setResult(null);
    setControlValues({});
    setPreviewPending(false);
    onPluginPreviewClear(tool.id);
    onError(null);
  };

  return (
    <>
      {tool.target === PLUGIN_TOOL_TARGET_IMAGE ? (
        <PluginImageTargetSection
          documentState={documentState}
          canvasSelectionTarget={canvasSelectionTarget}
        />
      ) : (
        <SelectionSection
          documentState={documentState}
          onSelectionChange={onSelectionChange}
        />
      )}
      <section className="panel-section panel-section-compact plugin-tool-panel">
        {tool.description ? (
          <div className="persistence-empty">{tool.description}</div>
        ) : null}
        {tool.controls.length > 0 ? (
          <div className="plugin-control-list">
            {tool.controls.map((control) => (
              <SchemaControl
                key={control.id}
                control={control}
                value={pluginToolControlValue(tool, control, controlValues)}
                disabled={!livePreview && mutation.isPending}
                onChange={(_id, value) =>
                  setControlValues((current) => ({
                    ...current,
                    [control.id]: value,
                  }))
                }
              />
            ))}
          </div>
        ) : null}
        {livePreview ? (
          <div className="plugin-live-preview">
            <div className="plugin-preview-status">
              <Palette size={15} />
              <span>
                {previewPending
                  ? "Updating selected image"
                  : result?.result.image
                    ? "Live preview on canvas"
                    : "Select an image to preview"}
              </span>
            </div>
            <div className="plugin-preview-actions">
              <Button
                type="button"
                variant="secondary"
                size="compact"
                disabled={
                  previewPending ||
                  !result?.result.image ||
                  !canApplyPluginImageResult(result.target)
                }
                onClick={applyLivePreview}
              >
                <Check size={14} />
                Apply preview
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="compact"
                onClick={resetLivePreview}
              >
                <RotateCcw size={14} />
                Reset
              </Button>
            </div>
          </div>
        ) : (
          <Button
            type="button"
            variant="secondary"
            size="large"
            disabled={mutation.isPending || processingDisabled}
            onClick={() => mutation.mutate()}
          >
            <PluginToolButtonIcon icon={tool.icon} />
            {mutation.isPending ? "Processing" : "Process selection"}
          </Button>
        )}
        {!livePreview && result?.result.text ? (
          <div className="plugin-action-result">
            <div className="plugin-control-heading">
              <TextSearch size={15} />
              <span>{tool.result_label ?? "Result"}</span>
            </div>
            <textarea readOnly value={result.result.text} />
          </div>
        ) : null}
        {!livePreview && result?.result.image ? (
          <div className="plugin-action-result plugin-image-result">
            <img src={result.result.image} alt="Plugin result preview" />
            <Button
              type="button"
              variant="secondary"
              size="compact"
              disabled={!canApplyPluginImageResult(result.target)}
              onClick={() => {
                if (result.result.image) {
                  applyPluginImageResult(
                    result.result.image,
                    result.target,
                    onRasterDataUrlChange,
                    onReferenceChange,
                  );
                }
                onError(null);
              }}
            >
              <Check size={14} />
              Apply to image
            </Button>
          </div>
        ) : null}
      </section>
    </>
  );
}

function PluginImageTargetSection({
  documentState,
  canvasSelectionTarget,
}: {
  documentState: EditorDocument;
  canvasSelectionTarget: CanvasSelectionTarget;
}) {
  const imageSelection = getSelectedImageContext(
    documentState,
    canvasSelectionTarget,
  );
  return imageSelection ? (
    <ImageSelectionSection selection={imageSelection} />
  ) : (
    <section className="tool-selection-context">
      <strong>Uploaded image</strong>
      <span>Select a base or reference image on the canvas.</span>
    </section>
  );
}

function PluginToolButtonIcon({ icon }: { icon: string }) {
  if (icon === "captions") {
    return <Captions size={16} />;
  }
  if (icon === "palette") {
    return <Palette size={16} />;
  }
  if (icon === "sliders-horizontal") {
    return <SlidersHorizontal size={16} />;
  }
  return <TextSearch size={16} />;
}

function getPluginToolActionTarget(
  tool: PluginToolInfo,
  canvasSelectionTarget: CanvasSelectionTarget,
): CanvasSelectionTarget | null {
  if (tool.target === PLUGIN_TOOL_TARGET_IMAGE) {
    return canvasSelectionTarget.kind === "raster" ||
      canvasSelectionTarget.kind === "reference"
      ? canvasSelectionTarget
      : null;
  }
  return { kind: "frame" };
}

function mutationTargetDisabled(
  tool: PluginToolInfo,
  target: CanvasSelectionTarget | null,
): boolean {
  return tool.target === PLUGIN_TOOL_TARGET_IMAGE && target === null;
}

function canApplyPluginImageResult(target: CanvasSelectionTarget): boolean {
  return target.kind === "raster" || target.kind === "reference";
}

function applyPluginImageResult(
  image: string,
  target: CanvasSelectionTarget,
  onRasterDataUrlChange: (dataUrl: string) => void,
  onReferenceChange: (
    id: string,
    patch: Partial<EditorDocument["references"][number]>,
  ) => void,
): void {
  if (target.kind === "raster") {
    onRasterDataUrlChange(image);
    return;
  }
  if (target.kind === "reference") {
    onReferenceChange(target.id, { dataUrl: image });
  }
}

async function runToolAction(
  tool: PluginToolInfo,
  documentState: EditorDocument,
  target: CanvasSelectionTarget,
  controls: Record<string, unknown>,
): Promise<{ result: PluginActionResult; target: CanvasSelectionTarget }> {
  const input = await renderPluginActionInput(documentState, target);
  const nextResult = await runPluginAction(tool.action_id, {
    image: input.image,
    controls,
    target: input.target,
    metadata: {
      document_id: documentState.id,
      tool_id: tool.id,
    },
  });
  return { result: nextResult, target };
}

function pluginActionTargetKey(target: CanvasSelectionTarget | null): string {
  if (!target) {
    return "none";
  }
  if (target.kind === "reference") {
    return `${target.kind}:${target.id}`;
  }
  return target.kind;
}

function pluginToolControlPayload(
  tool: PluginToolInfo,
  values: Record<string, unknown>,
): Record<string, unknown> {
  const payload: Record<string, unknown> = { ...tool.default_values };
  for (const control of tool.controls) {
    payload[control.id] = pluginToolControlValue(tool, control, values);
  }
  return payload;
}

function pluginToolControlValue(
  tool: PluginToolInfo,
  control: ControlSchema,
  values: Record<string, unknown>,
): unknown {
  if (control.id in values) {
    return values[control.id];
  }
  if (control.id in tool.default_values) {
    return tool.default_values[control.id];
  }
  return control.default_value;
}

function GenerationWorkflowSection({
  workspaceMode,
  generationMode,
  outpaintStrategy,
  parameters,
  controlGuideEnabled,
  controlGuideLabel,
  hfSpaceFillActive,
  controlGuideAvailable,
  onWorkspaceModeChange,
  onOutpaintStrategyChange,
  onParameterChange,
  onControlGuideEnabledChange,
  fixedExpandAvailable,
}: {
  workspaceMode: WorkspaceMode;
  generationMode: GenerationMode;
  outpaintStrategy: OutpaintStrategy;
  parameters: GenerationParameters;
  controlGuideEnabled: boolean;
  controlGuideLabel: string;
  hfSpaceFillActive: boolean;
  controlGuideAvailable: boolean;
  onWorkspaceModeChange: (mode: WorkspaceMode) => void;
  onOutpaintStrategyChange: (strategy: OutpaintStrategy) => void;
  onParameterChange: <TKey extends keyof GenerationParameters>(
    key: TKey,
    value: GenerationParameters[TKey],
  ) => void;
  onControlGuideEnabledChange: (enabled: boolean) => void;
  fixedExpandAvailable: boolean;
}) {
  const expandImageActive = workspaceMode === WORKSPACE_MODE_EXPAND_IMAGE;
  const directionalActive =
    generationMode === GENERATION_MODE_OUTPAINT &&
    outpaintStrategy === OUTPAINT_STRATEGY_DIRECTIONAL;
  const guideDisabled =
    expandImageActive || directionalActive || hfSpaceFillActive || !controlGuideAvailable;
  const guideEnabled = controlGuideEnabled && !guideDisabled;
  const guideDescription = guideDisabled
    ? "Guide input is already handled by the active mode."
    : "Optional sketch input that steers the generated area without switching tools.";
  return (
    <section className="panel-section generation-workflow-section">
      <div className="section-heading">
        <span>{expandImageActive ? "Expansion mode" : "Active tool"}</span>
      </div>
      {generationMode === GENERATION_MODE_OUTPAINT ? (
        <OutpaintMethodSwitch
          value={workspaceMode}
          fixedExpandAvailable={fixedExpandAvailable}
          onChange={onWorkspaceModeChange}
        />
      ) : null}
      {expandImageActive ? (
        <div className="expand-image-controls">
          <div className="expand-direction-grid" aria-label="Expansion direction">
            {OUTPAINT_DIRECTION_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                className={
                  parameters.outpaint_direction === option.id
                    ? "expand-direction-option expand-direction-option-active"
                    : "expand-direction-option"
                }
                aria-pressed={parameters.outpaint_direction === option.id}
                onClick={() =>
                  onParameterChange(
                    "outpaint_direction",
                    option.id as GenerationParameters["outpaint_direction"],
                  )
                }
              >
                {option.label}
              </button>
            ))}
          </div>
          {parameters.outpaint_direction === "around" ? (
            <div className="expand-size-grid">
              <SliderField
                label="Width growth"
                value={parameters.fixed_expand_width_percent}
                min={5}
                max={200}
                step={5}
                valueSuffix="%"
                onChange={(value) =>
                  onParameterChange("fixed_expand_width_percent", value)
                }
              />
              <SliderField
                label="Height growth"
                value={parameters.fixed_expand_height_percent}
                min={5}
                max={200}
                step={5}
                valueSuffix="%"
                onChange={(value) =>
                  onParameterChange("fixed_expand_height_percent", value)
                }
              />
            </div>
          ) : (
            <SliderField
              label="New area"
              value={parameters.fixed_expand_percent}
              min={5}
              max={200}
              step={5}
              valueSuffix="%"
              onChange={(value) =>
                onParameterChange("fixed_expand_percent", value)
              }
            />
          )}
          <SliderField
            label="Overlap"
            value={parameters.hf_space_overlap_percentage}
            min={1}
            max={50}
            step={1}
            valueSuffix="%"
            onChange={(value) =>
              onParameterChange("hf_space_overlap_percentage", value)
            }
          />
          <div className="render-scale-field">
            <div className="label-row">
              <span>Render scale</span>
              <strong>{fixedExpandScaleLabel(parameters)}</strong>
            </div>
            <div className="render-scale-grid" aria-label="Render scale">
              {FIXED_EXPAND_RENDER_SCALE_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={
                    parameters.fixed_expand_output_scale === option.id
                      ? "workflow-option workflow-option-active"
                      : "workflow-option"
                  }
                  aria-pressed={parameters.fixed_expand_output_scale === option.id}
                  onClick={() =>
                    onParameterChange("fixed_expand_output_scale", option.id)
                  }
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          {parameters.fixed_expand_output_scale === "custom" ? (
            <SliderField
              label="Custom scale"
              value={parameters.fixed_expand_custom_output_scale}
              min={25}
              max={100}
              step={5}
              valueSuffix="%"
              onChange={(value) =>
                onParameterChange("fixed_expand_custom_output_scale", value)
              }
            />
          ) : null}
          <div className="fixed-expand-toggle-row">
            <span>Preview guides</span>
            <OverlapToggle
              label={parameters.fixed_expand_show_guides ? "On" : "Off"}
              checked={parameters.fixed_expand_show_guides}
              onChange={(value) =>
                onParameterChange("fixed_expand_show_guides", value)
              }
            />
          </div>
          <div className="expand-overlap-grid" aria-label="Overlap sides">
            <OverlapToggle
              label="Left"
              checked={parameters.hf_space_overlap_left}
              onChange={(value) =>
                onParameterChange("hf_space_overlap_left", value)
              }
            />
            <OverlapToggle
              label="Right"
              checked={parameters.hf_space_overlap_right}
              onChange={(value) =>
                onParameterChange("hf_space_overlap_right", value)
              }
            />
            <OverlapToggle
              label="Top"
              checked={parameters.hf_space_overlap_top}
              onChange={(value) =>
                onParameterChange("hf_space_overlap_top", value)
              }
            />
            <OverlapToggle
              label="Bottom"
              checked={parameters.hf_space_overlap_bottom}
              onChange={(value) =>
                onParameterChange("hf_space_overlap_bottom", value)
              }
            />
          </div>
        </div>
      ) : generationMode === GENERATION_MODE_OUTPAINT ? (
        <>
          {fixedExpandAvailable && !directionalActive ? (
            <SliderField
              label="Overlap"
              value={Math.max(parameters.hf_space_overlap_percentage, 20)}
              min={20}
              max={50}
              step={1}
              valueSuffix="%"
              onChange={(value) =>
                onParameterChange("hf_space_overlap_percentage", value)
              }
            />
          ) : null}
          {!hfSpaceFillActive ? (
            <SelectControl
              label="Frame context"
              value={outpaintStrategy}
              options={FREE_EDIT_OUTPAINT_STRATEGY_OPTIONS}
              optionDetails={controlOptionDetailsFor("outpaint_strategy")}
              onChange={(value) => onOutpaintStrategyChange(value as OutpaintStrategy)}
            />
          ) : null}
          {directionalActive ? (
            <div className="directional-outpaint-controls">
              <SelectControl
                label="Direction"
                value={parameters.outpaint_direction}
                options={OUTPAINT_DIRECTION_OPTIONS}
                optionDetails={controlOptionDetailsFor("outpaint_direction")}
                onChange={(value) =>
                  onParameterChange(
                    "outpaint_direction",
                    value as GenerationParameters["outpaint_direction"],
                  )
                }
              />
              <label className="field-label">
                Generated size
                <input
                  type="number"
                  min={DIRECTIONAL_OUTPAINT_MIN_SIZE}
                  step={64}
                  value={parameters.outpaint_generated_size}
                  onChange={(event) =>
                    onParameterChange(
                      "outpaint_generated_size",
                      sanitizeDirectionalNumber(event.target.value),
                    )
                  }
                />
                <span className="control-help-text">
                  Pixels generated along the chosen direction.
                </span>
              </label>
              <label className="field-label">
                Context
                <input
                  type="number"
                  min={1}
                  step={64}
                  value={parameters.outpaint_context_size}
                  onChange={(event) =>
                    onParameterChange(
                      "outpaint_context_size",
                      sanitizePositiveNumber(event.target.value, 512),
                    )
                  }
                />
                <span className="control-help-text">
                  Existing pixels included so the new area matches the image.
                </span>
              </label>
              <label className="field-label">
                Cross axis
                <input
                  type="number"
                  min={DIRECTIONAL_OUTPAINT_MIN_SIZE}
                  step={64}
                  value={parameters.outpaint_cross_size}
                  onChange={(event) =>
                    onParameterChange(
                      "outpaint_cross_size",
                      sanitizeDirectionalNumber(event.target.value),
                    )
                  }
                />
                <span className="control-help-text">
                  Size across the direction, such as height for left or right.
                </span>
              </label>
            </div>
          ) : null}
        </>
      ) : null}
      {workspaceMode === WORKSPACE_MODE_FREE_EDIT && controlGuideAvailable ? (
        <>
          <div className="section-heading">
            <span>Guide</span>
          </div>
          <p className="control-help-text">{guideDescription}</p>
          <div className="workflow-segmented-control" aria-label="Sketch guide">
            <button
              type="button"
              className={
                controlGuideEnabled
                  ? "workflow-option"
                  : "workflow-option workflow-option-active"
              }
              aria-pressed={!guideEnabled}
              onClick={() => onControlGuideEnabledChange(false)}
            >
              <MousePointer2 size={15} />
              None
            </button>
            <button
              type="button"
              className={
                guideEnabled
                  ? "workflow-option workflow-option-active"
                  : "workflow-option"
              }
              aria-pressed={guideEnabled}
              disabled={guideDisabled}
              onClick={() => onControlGuideEnabledChange(true)}
            >
              <Palette size={15} />
              {controlGuideLabel}
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}

function OverlapToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={
        checked
          ? "expand-overlap-toggle expand-overlap-toggle-active"
          : "expand-overlap-toggle"
      }
      aria-pressed={checked}
      onClick={() => onChange(!checked)}
    >
      {label}
    </button>
  );
}

function fixedExpandScaleLabel(parameters: GenerationParameters): string {
  if (parameters.fixed_expand_output_scale === "safe") {
    return "60%";
  }
  if (parameters.fixed_expand_output_scale === "balanced") {
    return "75%";
  }
  if (parameters.fixed_expand_output_scale === "custom") {
    return `${parameters.fixed_expand_custom_output_scale}%`;
  }
  return "100%";
}

function sanitizeDirectionalNumber(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return DIRECTIONAL_OUTPAINT_MIN_SIZE;
  }
  return Math.max(DIRECTIONAL_OUTPAINT_MIN_SIZE, Math.round(parsed));
}

function sanitizePositiveNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(1, Math.round(parsed));
}

function ToolContextSections({
  tool,
  generationMode,
  documentState,
  canvasSelectionTarget,
  viewport,
  brushSize,
  eraserHardness,
  controlGuideColor,
  controlGuideStrength,
  controlGuideMaskMode,
  onSelectionChange,
  onBrushSizeChange,
  onEraserHardnessChange,
  onControlGuideColorChange,
  onControlGuideStrengthChange,
  onControlGuideMaskModeChange,
  onClearMask,
  onClearControlGuide,
}: {
  tool: EditorTool;
  generationMode: GenerationMode;
  documentState: EditorDocument;
  canvasSelectionTarget: CanvasSelectionTarget;
  viewport: ViewportState;
  brushSize: number;
  eraserHardness: number;
  controlGuideColor: string;
  controlGuideStrength: number;
  controlGuideMaskMode: ControlGuideMaskMode;
  onSelectionChange: (selection: SelectionRect) => void;
  onBrushSizeChange: (value: number) => void;
  onEraserHardnessChange: (value: number) => void;
  onControlGuideColorChange: (value: string) => void;
  onControlGuideStrengthChange: (value: number) => void;
  onControlGuideMaskModeChange: (value: ControlGuideMaskMode) => void;
  onClearMask: () => void;
  onClearControlGuide: () => void;
}) {
  if (!CONTROLNET_GUIDE_UI_ENABLED && tool === TOOL_CONTROL_GUIDE) {
    return null;
  }

  if (tool === TOOL_CONTROL_GUIDE) {
    return (
      <section className="tool-control-section">
        <SliderField
          label="Brush size"
          value={brushSize}
          min={MIN_BRUSH_SIZE}
          max={MAX_BRUSH_SIZE}
          step={1}
          onChange={onBrushSizeChange}
        />
        <ControlGuideColorPicker
          value={controlGuideColor}
          onChange={onControlGuideColorChange}
        />
        <SliderField
          label="Stroke strength"
          value={controlGuideStrength}
          min={MIN_CONTROL_GUIDE_STRENGTH}
          max={MAX_CONTROL_GUIDE_STRENGTH}
          step={1}
          valueSuffix="%"
          onChange={onControlGuideStrengthChange}
        />
        {generationMode === GENERATION_MODE_INPAINT ? (
          <ControlGuideMaskModeField
            value={controlGuideMaskMode}
            onChange={onControlGuideMaskModeChange}
          />
        ) : null}
        <Button type="button" variant="secondary" onClick={onClearControlGuide}>
          <Trash2 size={15} />
          Clear guide
        </Button>
      </section>
    );
  }

  if (tool === TOOL_INPAINT_MASK) {
    return (
      <section className="tool-control-section">
        <SliderField
          label="Brush size"
          value={brushSize}
          min={MIN_BRUSH_SIZE}
          max={MAX_BRUSH_SIZE}
          step={1}
          onChange={onBrushSizeChange}
        />
        <Button type="button" variant="secondary" onClick={onClearMask}>
          <Trash2 size={15} />
          Clear mask
        </Button>
      </section>
    );
  }

  if (tool === TOOL_ERASE) {
    return (
      <section className="tool-control-section">
        <SliderField
          label="Brush size"
          value={brushSize}
          min={MIN_BRUSH_SIZE}
          max={MAX_BRUSH_SIZE}
          step={1}
          onChange={onBrushSizeChange}
        />
        <SliderField
          label="Hardness"
          value={eraserHardness}
          min={MIN_ERASER_HARDNESS}
          max={MAX_ERASER_HARDNESS}
          step={1}
          valueSuffix="%"
          onChange={onEraserHardnessChange}
        />
      </section>
    );
  }

  if (tool === TOOL_PAN) {
    return (
      <section className="tool-control-section">
        <div className="tool-metric-grid">
          <ToolMetric label="X" value={String(Math.round(viewport.x))} />
          <ToolMetric label="Y" value={String(Math.round(viewport.y))} />
          <ToolMetric
            label="Zoom"
            value={`${Math.round(viewport.zoom * 100)}%`}
          />
        </div>
      </section>
    );
  }

  if (
    tool === TOOL_OUTPAINT_FRAME ||
    (tool === TOOL_SELECT && canvasSelectionTarget.kind === "frame")
  ) {
    return (
      <SelectionSection
        documentState={documentState}
        onSelectionChange={onSelectionChange}
      />
    );
  }

  if (tool === TOOL_SELECT) {
    const imageSelection = getSelectedImageContext(
      documentState,
      canvasSelectionTarget,
    );
    return imageSelection ? (
      <ImageSelectionSection selection={imageSelection} />
    ) : (
      <SelectSection />
    );
  }

  return null;
}

function SelectSection() {
  return (
    <section className="tool-selection-context">
      <strong>Select</strong>
      <span>Choose an image or the outpaint frame on the canvas.</span>
    </section>
  );
}

function ImageSelectionSection({
  selection,
}: {
  selection: { label: string; bounds: DocumentBounds };
}) {
  return (
    <section className="tool-selection-context">
      <div className="selection-context-heading">
        <strong>{selection.label}</strong>
        <span>
          {Math.round(selection.bounds.width)} x{" "}
          {Math.round(selection.bounds.height)}
        </span>
      </div>
      <div className="tool-metric-grid">
        <ToolMetric label="X" value={String(Math.round(selection.bounds.x))} />
        <ToolMetric label="Y" value={String(Math.round(selection.bounds.y))} />
        <ToolMetric
          label="Width"
          value={String(Math.round(selection.bounds.width))}
        />
        <ToolMetric
          label="Height"
          value={String(Math.round(selection.bounds.height))}
        />
      </div>
    </section>
  );
}

function getSelectedImageContext(
  documentState: EditorDocument,
  canvasSelectionTarget: CanvasSelectionTarget,
): { label: string; bounds: DocumentBounds } | null {
  if (canvasSelectionTarget.kind === "raster" && documentState.rasterDataUrl) {
    return {
      label: "Base image",
      bounds: documentState.rasterBounds ?? {
        x: 0,
        y: 0,
        width: documentState.width,
        height: documentState.height,
      },
    };
  }
  if (canvasSelectionTarget.kind === "reference") {
    const reference = documentState.references.find(
      (item) => item.id === canvasSelectionTarget.id,
    );
    if (!reference) {
      return null;
    }
    return {
      label: "Reference image",
      bounds: {
        x: reference.x,
        y: reference.y,
        width: reference.width,
        height: reference.height,
      },
    };
  }
  return null;
}

function SelectionSection({
  documentState,
  onSelectionChange,
}: {
  documentState: EditorDocument;
  onSelectionChange: (selection: SelectionRect) => void;
}) {
  const [open, setOpen] = useState(true);
  const CountIcon = open ? ChevronDown : ChevronRight;

  return (
    <section
      className={
        open
          ? "panel-section generation-control-section generation-control-section-open"
          : "panel-section generation-control-section"
      }
    >
      <button
        type="button"
        className="generation-section-trigger"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="generation-section-title">
          <Frame size={16} />
          <span>Frame</span>
        </span>
        <span className="generation-section-meta">
          <span>
            {documentState.selection.width} x {documentState.selection.height}
          </span>
          <CountIcon size={15} />
        </span>
      </button>
      {open ? (
        <div className="generation-section-body">
          <div className="field-grid">
            <NumericField
              label="Width"
              value={documentState.selection.width}
              min={1}
              max={4096}
              onChange={(value) =>
                onSelectionChange({ ...documentState.selection, width: value })
              }
            />
            <NumericField
              label="Height"
              value={documentState.selection.height}
              min={1}
              max={4096}
              onChange={(value) =>
                onSelectionChange({ ...documentState.selection, height: value })
              }
            />
          </div>
          <div className="field-grid">
            <NumericField
              label="Left"
              value={documentState.selection.x}
              min={-16384}
              max={16384}
              onChange={(value) =>
                onSelectionChange({ ...documentState.selection, x: value })
              }
            />
            <NumericField
              label="Top"
              value={documentState.selection.y}
              min={-16384}
              max={16384}
              onChange={(value) =>
                onSelectionChange({ ...documentState.selection, y: value })
              }
            />
          </div>
        </div>
      ) : null}
    </section>
  );
}

function ToolMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="tool-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NumericField({
  label,
  value,
  min,
  max,
  disabled = false,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <NumberStepper
      label={label}
      value={value}
      min={min}
      max={max}
      disabled={disabled}
      onChange={onChange}
    />
  );
}

function ControlGuideColorPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="control-guide-color-field">
      <div className="label-row">
        <span>Color</span>
        <input
          type="color"
          value={value}
          aria-label="Control guide color"
          onChange={(event) => onChange(event.target.value)}
        />
      </div>
      <div className="control-guide-swatches" aria-label="Control guide swatches">
        {CONTROL_GUIDE_COLORS.map((color) => (
          <button
            key={color}
            type="button"
            className={
              color === value
                ? "control-guide-swatch control-guide-swatch-active"
                : "control-guide-swatch"
            }
            style={{ backgroundColor: color }}
            aria-label={`Use ${color}`}
            onClick={() => onChange(color)}
          />
        ))}
      </div>
    </div>
  );
}

function ControlGuideMaskModeField({
  value,
  onChange,
}: {
  value: ControlGuideMaskMode;
  onChange: (value: ControlGuideMaskMode) => void;
}) {
  return (
    <label className="field-label">
      <span>Masked area guide</span>
      <select
        value={value}
        onChange={(event) =>
          onChange(event.target.value as ControlGuideMaskMode)
        }
      >
        {CONTROL_GUIDE_MASK_MODE_OPTIONS.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  valueSuffix = "",
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  valueSuffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field-label">
      <span className="label-row">
        {label}
        <strong>
          {formatSliderValue(value, step)}
          {valueSuffix}
        </strong>
      </span>
      <input
        className="range-input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function formatSliderValue(value: number, step: number): string {
  if (Number.isInteger(step)) {
    return String(Math.round(value));
  }
  return value.toFixed(step < 0.1 ? 2 : 1);
}

function getToolContext(
  tool: EditorTool,
  activePluginTool: PluginToolInfo | null,
): ToolContext {
  if (activePluginTool) {
    return {
      title: activePluginTool.label,
      icon: pluginToolContextIcon(activePluginTool.icon),
    };
  }
  if (tool === TOOL_ERASE) {
    return {
      title: "Pixel eraser",
      icon: Eraser,
    };
  }
  if (tool === TOOL_PAN) {
    return {
      title: "Pan",
      icon: Hand,
    };
  }
  if (tool === TOOL_INPAINT_MASK) {
    return {
      title: "Inpaint mask",
      icon: Paintbrush,
    };
  }
  if (tool === TOOL_CONTROL_GUIDE) {
    return {
      title: "Control guide",
      icon: Palette,
    };
  }
  if (tool === TOOL_OUTPAINT_FRAME) {
    return {
      title: "Outpaint",
      icon: Frame,
    };
  }
  if (tool === TOOL_SELECT) {
    return {
      title: "Select",
      icon: MousePointer2,
    };
  }
  return {
    title: "Tool",
    icon: MousePointer2,
  };
}

function pluginToolContextIcon(icon: string): LucideIcon {
  if (icon === "captions") {
    return Captions;
  }
  if (icon === "palette") {
    return Palette;
  }
  if (icon === "sliders-horizontal") {
    return SlidersHorizontal;
  }
  return TextSearch;
}
