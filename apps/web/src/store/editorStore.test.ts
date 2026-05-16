import { describe, expect, it } from "vitest";
import {
  ADAPTER_SDXL_FILL_IP_REFINE,
  DEFAULT_ERASER_HARDNESS,
  DEFAULT_CONTROL_GUIDE_COLOR,
  DEFAULT_CONTROL_GUIDE_MASK_MODE,
  DEFAULT_CONTROL_GUIDE_STRENGTH,
  DEFAULT_ADAPTER_ID,
  CONTROL_GUIDE_MASK_MODE_PRESERVE,
  FILL_EDGE_EXTEND,
  FILL_TRANSPARENT,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  INPAINT_AREA_ONLY_MASKED,
  INPAINT_AREA_WHOLE_SELECTION,
  OUTPAINT_DIRECTION_RIGHT,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_WHOLE_RESIZED,
  RESULT_MODE_GENERATED_SELECTION,
  RESULT_MODE_PRESERVE_KNOWN,
  SCHEDULER_AUTO,
  SCHEDULER_DPM_SOLVER,
  TOOL_CONTROL_GUIDE,
  TOOL_INPAINT_MASK,
  TOOL_OUTPAINT_FRAME,
  TOOL_SELECT,
  WORKSPACE_MODE_EXPAND_IMAGE,
  WORKSPACE_MODE_FREE_EDIT,
} from "../constants/domain";
import { useEditorStore } from "./editorStore";

describe("editorStore", () => {
  it("starts on the recommended SDXL Fill profile with ControlNet sketch disabled", () => {
    expect(useEditorStore.getState().selectedAdapterId).toBe(DEFAULT_ADAPTER_ID);
    expect(DEFAULT_ADAPTER_ID).toBe("sdxl-fill-ip-refine");
    expect(useEditorStore.getState().modelId).toBe("SG161222/RealVisXL_V5.0_Lightning");
    expect(useEditorStore.getState().parameters.outpaint_strategy).toBe(
      OUTPAINT_STRATEGY_HF_SPACE_FILL,
    );
    expect(useEditorStore.getState().parameters.steps).toBe(8);
    expect(useEditorStore.getState().parameters.guidance_scale).toBe(1.5);
    expect(useEditorStore.getState().controlGuideEnabled).toBe(false);
  });

  it("toggles ControlNet sketch independently from the edit mode", () => {
    useEditorStore.setState({
      controlGuideEnabled: false,
      generationMode: GENERATION_MODE_OUTPAINT,
    });

    useEditorStore.getState().setControlGuideEnabled(true);

    expect(useEditorStore.getState().controlGuideEnabled).toBe(true);
    expect(useEditorStore.getState().generationMode).toBe(GENERATION_MODE_OUTPAINT);

    useEditorStore.getState().setGenerationMode(GENERATION_MODE_INPAINT);

    expect(useEditorStore.getState().controlGuideEnabled).toBe(true);
    expect(useEditorStore.getState().generationMode).toBe(GENERATION_MODE_INPAINT);
  });

  it("applies adapter generation defaults when the adapter changes", () => {
    useEditorStore.setState((state) => ({
      selectedAdapterId: "sd15-inpaint",
      modelId: "stable-diffusion-v1-5/stable-diffusion-inpainting",
      parameters: {
        ...state.parameters,
        prompt: "keep this prompt",
        steps: 28,
        guidance_scale: 7.5,
        strength: 1,
        scheduler: SCHEDULER_DPM_SOLVER,
      },
    }));

    useEditorStore
      .getState()
      .setSelectedAdapterId("flux-fill", "black-forest-labs/FLUX.1-Fill-dev", {
        prompt: "",
        steps: 50,
        guidance_scale: 30,
        strength: 1,
        scheduler: SCHEDULER_AUTO,
      });

    const state = useEditorStore.getState();

    expect(state.selectedAdapterId).toBe("flux-fill");
    expect(state.modelId).toBe("black-forest-labs/FLUX.1-Fill-dev");
    expect(state.parameters.prompt).toBe("keep this prompt");
    expect(state.parameters.steps).toBe(50);
    expect(state.parameters.guidance_scale).toBe(30);
    expect(state.parameters.scheduler).toBe(SCHEDULER_AUTO);
  });

  it("switches inpaint mode to brush masking and preserves known pixels", () => {
    useEditorStore.setState((state) => ({
      tool: TOOL_SELECT,
      generationMode: GENERATION_MODE_OUTPAINT,
      parameters: {
        ...state.parameters,
        result_mode: RESULT_MODE_GENERATED_SELECTION,
      },
    }));

    useEditorStore.getState().setGenerationMode(GENERATION_MODE_INPAINT);

    expect(useEditorStore.getState().tool).toBe(TOOL_INPAINT_MASK);
    expect(useEditorStore.getState().canvasSelectionTarget).toEqual({
      kind: "none",
    });
    expect(useEditorStore.getState().parameters.result_mode).toBe(
      RESULT_MODE_PRESERVE_KNOWN,
    );
  });

  it("rejects generated-selection result mode while inpaint is active", () => {
    useEditorStore.setState((state) => ({
      generationMode: GENERATION_MODE_INPAINT,
      parameters: {
        ...state.parameters,
        result_mode: RESULT_MODE_PRESERVE_KNOWN,
      },
    }));

    useEditorStore
      .getState()
      .updateParameter("result_mode", RESULT_MODE_GENERATED_SELECTION);

    expect(useEditorStore.getState().parameters.result_mode).toBe(
      RESULT_MODE_PRESERVE_KNOWN,
    );
  });

  it("keeps inpaint adapter changes on a mask-preserving result mode", () => {
    useEditorStore.setState((state) => ({
      generationMode: GENERATION_MODE_INPAINT,
      parameters: {
        ...state.parameters,
        result_mode: RESULT_MODE_PRESERVE_KNOWN,
      },
    }));

    useEditorStore
      .getState()
      .setSelectedAdapterId("sd15-controlnet-inpaint", "stable-diffusion-v1-5/stable-diffusion-inpainting", {
        result_mode: RESULT_MODE_GENERATED_SELECTION,
      });

    expect(useEditorStore.getState().parameters.result_mode).toBe(
      RESULT_MODE_PRESERVE_KNOWN,
    );
  });

  it("switches outpaint mode back to the frame selector", () => {
    useEditorStore.setState({
      tool: TOOL_INPAINT_MASK,
      generationMode: GENERATION_MODE_INPAINT,
    });

    useEditorStore.getState().setGenerationMode(GENERATION_MODE_OUTPAINT);

    expect(useEditorStore.getState().tool).toBe(TOOL_OUTPAINT_FRAME);
    expect(useEditorStore.getState().canvasSelectionTarget).toEqual({
      kind: "frame",
    });
    expect(useEditorStore.getState().generationMode).toBe(
      GENERATION_MODE_OUTPAINT,
    );
  });

  it("switches expand image mode to fixed SDXL Fill expansion without a free frame", () => {
    useEditorStore.setState((state) => ({
      workspaceMode: WORKSPACE_MODE_FREE_EDIT,
      tool: TOOL_OUTPAINT_FRAME,
      canvasSelectionTarget: { kind: "frame" },
      generationMode: GENERATION_MODE_INPAINT,
      controlGuideEnabled: true,
      parameters: {
        ...state.parameters,
        outpaint_strategy: OUTPAINT_STRATEGY_WHOLE_RESIZED,
      },
    }));

    useEditorStore.getState().setWorkspaceMode(WORKSPACE_MODE_EXPAND_IMAGE);

    const state = useEditorStore.getState();
    expect(state.workspaceMode).toBe(WORKSPACE_MODE_EXPAND_IMAGE);
    expect(state.generationMode).toBe(GENERATION_MODE_OUTPAINT);
    expect(state.tool).toBe(TOOL_SELECT);
    expect(state.canvasSelectionTarget).toEqual({ kind: "none" });
    expect(state.controlGuideEnabled).toBe(false);
    expect(state.parameters.outpaint_strategy).toBe(OUTPAINT_STRATEGY_HF_SPACE_FILL);
  });

  it("keeps free edit mode on a movable outpaint frame without forcing adapter changes", () => {
    useEditorStore.setState((state) => ({
      workspaceMode: WORKSPACE_MODE_EXPAND_IMAGE,
      selectedAdapterId: "custom-adapter",
      tool: TOOL_SELECT,
      canvasSelectionTarget: { kind: "none" },
      generationMode: GENERATION_MODE_OUTPAINT,
      parameters: {
        ...state.parameters,
        outpaint_strategy: OUTPAINT_STRATEGY_DIRECTIONAL,
      },
    }));

    useEditorStore.getState().setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);

    const state = useEditorStore.getState();
    expect(state.workspaceMode).toBe(WORKSPACE_MODE_FREE_EDIT);
    expect(state.selectedAdapterId).toBe("custom-adapter");
    expect(state.generationMode).toBe(GENERATION_MODE_OUTPAINT);
    expect(state.tool).toBe(TOOL_OUTPAINT_FRAME);
    expect(state.canvasSelectionTarget).toEqual({ kind: "frame" });
  });

  it("routes the legacy directional preset to SDXL Fill fixed expansion", () => {
    useEditorStore.setState((state) => ({
      selectedAdapterId: "sd15-controlnet-inpaint",
      controlGuideEnabled: true,
      parameters: {
        ...state.parameters,
        outpaint_strategy: OUTPAINT_STRATEGY_WHOLE_RESIZED,
        fill_mode: FILL_TRANSPARENT,
        mask_blur: 32,
        inpaint_area: INPAINT_AREA_ONLY_MASKED,
        result_mode: RESULT_MODE_GENERATED_SELECTION,
        sample_count: 4,
        random_seed: true,
        strength: 0.35,
        scheduler: SCHEDULER_AUTO,
      },
    }));

    useEditorStore.getState().applyDirectionalOutpaintPreset();

    const state = useEditorStore.getState();
    expect(state.selectedAdapterId).toBe(ADAPTER_SDXL_FILL_IP_REFINE);
    expect(state.controlGuideEnabled).toBe(false);
    expect(state.parameters.outpaint_strategy).toBe(OUTPAINT_STRATEGY_HF_SPACE_FILL);
    expect(state.parameters.outpaint_direction).toBe(OUTPAINT_DIRECTION_RIGHT);
    expect(state.parameters.outpaint_generated_size).toBe(1024);
    expect(state.parameters.outpaint_context_size).toBe(512);
    expect(state.parameters.outpaint_cross_size).toBe(1024);
    expect(state.parameters.strength).toBe(1);
    expect(state.parameters.fill_mode).toBe(FILL_EDGE_EXTEND);
    expect(state.parameters.mask_blur).toBe(0);
    expect(state.parameters.inpaint_area).toBe(INPAINT_AREA_WHOLE_SELECTION);
    expect(state.parameters.result_mode).toBe(RESULT_MODE_PRESERVE_KNOWN);
    expect(state.parameters.sample_count).toBe(1);
    expect(state.parameters.random_seed).toBe(false);
    expect(state.parameters.scheduler).toBe(SCHEDULER_DPM_SOLVER);
  });

  it("updates eraser hardness independently from brush size", () => {
    useEditorStore.setState({
      brushSize: 48,
      eraserHardness: DEFAULT_ERASER_HARDNESS,
    });

    useEditorStore.getState().setEraserHardness(35);

    expect(useEditorStore.getState().eraserHardness).toBe(35);
    expect(useEditorStore.getState().brushSize).toBe(48);
  });

  it("records and clears control guide strokes separately from the inpaint mask", () => {
    useEditorStore.setState((state) => ({
      document: {
        ...state.document,
        maskStrokes: [],
        controlStrokes: [],
      },
      brushSize: 32,
      controlGuideColor: "#1e88e5",
      controlGuideStrength: 64,
      tool: TOOL_CONTROL_GUIDE,
      generationMode: GENERATION_MODE_OUTPAINT,
    }));

    useEditorStore.getState().beginMaskStroke("paint", { x: 4, y: 8 });
    useEditorStore.getState().beginControlStroke({ x: 16, y: 24 });
    useEditorStore.getState().appendControlPoint({ x: 48, y: 64 });

    expect(useEditorStore.getState().document.maskStrokes).toHaveLength(1);
    expect(useEditorStore.getState().document.controlStrokes).toHaveLength(1);
    expect(useEditorStore.getState().document.controlStrokes[0].points).toEqual([
      { x: 16, y: 24 },
      { x: 48, y: 64 },
    ]);
    expect(useEditorStore.getState().document.controlStrokes[0].color).toBe(
      "#1e88e5",
    );
    expect(useEditorStore.getState().document.controlStrokes[0].strength).toBe(
      64,
    );

    useEditorStore.getState().clearControlGuide();

    expect(useEditorStore.getState().document.maskStrokes).toHaveLength(1);
    expect(useEditorStore.getState().document.controlStrokes).toHaveLength(0);
  });

  it("updates the active control guide color", () => {
    useEditorStore.setState({
      controlGuideColor: DEFAULT_CONTROL_GUIDE_COLOR,
    });

    useEditorStore.getState().setControlGuideColor("#f43f5e");

    expect(useEditorStore.getState().controlGuideColor).toBe("#f43f5e");
  });

  it("updates the active control guide stroke strength", () => {
    useEditorStore.setState({
      controlGuideStrength: DEFAULT_CONTROL_GUIDE_STRENGTH,
    });

    useEditorStore.getState().setControlGuideStrength(55);

    expect(useEditorStore.getState().controlGuideStrength).toBe(55);
  });

  it("updates the inpaint control guide mask mode", () => {
    useEditorStore.setState({
      controlGuideMaskMode: DEFAULT_CONTROL_GUIDE_MASK_MODE,
    });

    useEditorStore
      .getState()
      .setControlGuideMaskMode(CONTROL_GUIDE_MASK_MODE_PRESERVE);

    expect(useEditorStore.getState().controlGuideMaskMode).toBe(
      CONTROL_GUIDE_MASK_MODE_PRESERVE,
    );
  });
});
