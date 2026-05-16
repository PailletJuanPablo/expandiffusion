import { useEffect, useState } from "react";
import {
  CONTROLNET_GUIDE_UI_ENABLED,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  TOOL_ERASE,
  TOOL_CONTROL_GUIDE,
  TOOL_PAN,
  TOOL_SELECT,
  TOOL_SHORTCUTS,
  WORKSPACE_MODE_FREE_EDIT,
} from "../constants/domain";
import { CanvasWorkspace } from "./CanvasWorkspace";
import { Filmstrip } from "./Filmstrip";
import { InspectorPanel } from "./InspectorPanel";
import { LeftToolbar } from "./LeftToolbar";
import { StatusBar } from "./StatusBar";
import { useEditorStore } from "../store/editorStore";
import { isEditableShortcutTarget } from "../lib/keyboardTargets";

export function Studio() {
  const [filmstripCollapsed, setFilmstripCollapsed] = useState(false);
  const setTool = useEditorStore((state) => state.setTool);
  const setGenerationMode = useEditorStore((state) => state.setGenerationMode);
  const setWorkspaceMode = useEditorStore((state) => state.setWorkspaceMode);
  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);
  const rejectResults = useEditorStore((state) => state.rejectResults);
  const pendingResults = useEditorStore((state) => state.pendingResults);
  const selectedResultIndex = useEditorStore(
    (state) => state.selectedResultIndex,
  );
  const selectResult = useEditorStore((state) => state.selectResult);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isEditableShortcutTarget(event.target)) {
        return;
      }

      const key = event.key.toLowerCase();
      if (event.ctrlKey && key === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
        return;
      }
      if (
        (event.ctrlKey && key === "y") ||
        (event.ctrlKey && event.shiftKey && key === "z")
      ) {
        event.preventDefault();
        redo();
        return;
      }
      if (key === TOOL_SHORTCUTS.SELECT) {
        setTool(TOOL_SELECT);
      }
      if (key === TOOL_SHORTCUTS.PAN) {
        setTool(TOOL_PAN);
      }
      if (key === TOOL_SHORTCUTS.ERASE) {
        setTool(TOOL_ERASE);
      }
      if (key === TOOL_SHORTCUTS.INPAINT_MASK) {
        setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
        setGenerationMode(GENERATION_MODE_INPAINT);
      }
      if (CONTROLNET_GUIDE_UI_ENABLED && key === TOOL_SHORTCUTS.CONTROL_GUIDE) {
        setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
        setGenerationMode(GENERATION_MODE_OUTPAINT);
        setTool(TOOL_CONTROL_GUIDE);
      }
      if (event.key === TOOL_SHORTCUTS.ESCAPE) {
        rejectResults();
      }
      if (pendingResults.length > 0 && event.key === "ArrowLeft") {
        event.preventDefault();
        selectResult(Math.max(0, selectedResultIndex - 1));
      }
      if (pendingResults.length > 0 && event.key === "ArrowRight") {
        event.preventDefault();
        selectResult(
          Math.min(pendingResults.length - 1, selectedResultIndex + 1),
        );
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    pendingResults.length,
    redo,
    rejectResults,
    selectResult,
    selectedResultIndex,
    setGenerationMode,
    setWorkspaceMode,
    setTool,
    undo,
  ]);

  return (
    <div
      className={
        filmstripCollapsed
          ? "studio-root studio-root-filmstrip-collapsed"
          : "studio-root"
      }
    >
      <LeftToolbar />
      <CanvasWorkspace />
      <InspectorPanel />
      <Filmstrip
        collapsed={filmstripCollapsed}
        onCollapsedChange={setFilmstripCollapsed}
      />
      <StatusBar />
    </div>
  );
}
