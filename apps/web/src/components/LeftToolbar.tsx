import {
  Captions,
  Eraser,
  FileArchive,
  Frame,
  Hand,
  MousePointer2,
  Paintbrush,
  Palette,
  Puzzle,
  RotateCcw,
  RotateCw,
  Save,
  SlidersHorizontal,
  TextSearch,
  Upload,
} from "lucide-react";
import {
  useRef,
  type CSSProperties,
  type ChangeEvent,
  type ReactNode,
} from "react";
import {
  CONTROLNET_GUIDE_UI_ENABLED,
  GENERATION_MODE_OUTPAINT,
  GENERATION_MODE_INPAINT,
  TOOL_CONTROL_GUIDE,
  TOOL_ERASE,
  TOOL_PAN,
  TOOL_SELECT,
  PLUGIN_TOOL_TARGET_FRAME,
  PLUGIN_TOOL_TARGET_IMAGE,
  WORKSPACE_MODE_FREE_EDIT,
  pluginEditorToolId,
  pluginToolIdFromEditorTool,
  type EditorTool,
} from "../constants/domain";
import type { PluginToolInfo } from "../domain/types";
import { useStudioQueries } from "../hooks/useStudioQueries";
import {
  measureImage,
  prepareRasterImport,
  readFileAsDataUrl,
} from "../lib/canvasRender";
import {
  ONBOARDING_TARGET_TOOLBAR,
  ONBOARDING_TARGET_UPLOAD_BUTTON,
} from "../lib/onboardingTour";
import { loadProjectArchive, saveProjectArchive } from "../lib/projectArchive";
import { useEditorStore } from "../store/editorStore";
import { TooltipButton } from "./TooltipButton";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

export function LeftToolbar() {
  const { pluginToolsQuery } = useStudioQueries();
  const tool = useEditorStore((state) => state.tool);
  const generationMode = useEditorStore((state) => state.generationMode);
  const setTool = useEditorStore((state) => state.setTool);
  const setGenerationMode = useEditorStore((state) => state.setGenerationMode);
  const setWorkspaceMode = useEditorStore((state) => state.setWorkspaceMode);
  const setCanvasSelectionTarget = useEditorStore(
    (state) => state.setCanvasSelectionTarget,
  );
  const canvasSelectionTarget = useEditorStore(
    (state) => state.canvasSelectionTarget,
  );
  const documentState = useEditorStore((state) => state.document);
  const setPreparedRaster = useEditorStore((state) => state.setPreparedRaster);
  const addReference = useEditorStore((state) => state.addReference);
  const replaceDocument = useEditorStore((state) => state.replaceDocument);
  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage);

  const importImages = async (files: File[]) => {
    let baseImageImported = Boolean(documentState.rasterDataUrl);
    for (const file of files) {
      const dataUrl = await readFileAsDataUrl(file);
      if (!baseImageImported) {
        const raster = await prepareRasterImport(dataUrl);
        setPreparedRaster(raster);
        baseImageImported = true;
        continue;
      }
      const dimensions = await measureImage(dataUrl);
      addReference(dataUrl, dimensions.width, dimensions.height);
      setTool(TOOL_SELECT);
    }
  };

  const handleRasterFile = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0) {
      return;
    }
    importImages(files).catch((error) =>
      setErrorMessage(
        error instanceof Error ? error.message : "Image import failed.",
      ),
    );
  };

  const loadProject = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.item(0);
    event.target.value = "";
    if (!file) {
      return;
    }
    loadProjectArchive(file)
      .then(replaceDocument)
      .catch((error) =>
        setErrorMessage(
          error instanceof Error ? error.message : "Project load failed.",
        ),
      );
  };
  const pluginTools = pluginToolsQuery.data ?? [];
  const activePluginToolId = pluginToolIdFromEditorTool(tool);
  const renderToolbarItem = (children: ReactNode, key?: string) => (
    <ToolbarItem key={key}>{children}</ToolbarItem>
  );

  return (
    <>
      <aside
        className="left-toolbar"
        aria-label="Editor tools"
        data-tour-id={ONBOARDING_TARGET_TOOLBAR}
      >
        <div className="toolbar-section" aria-label="Selection">
          {renderToolbarItem(
            <ToolbarTool
              tool={TOOL_SELECT}
              currentTool={tool}
              label="Select"
              onSelect={setTool}
            >
              <MousePointer2 size={18} />
            </ToolbarTool>,
          )}
        </div>
        <div className="toolbar-divider" />
        <div className="toolbar-section" aria-label="Generation actions">
          {renderToolbarItem(
            <ToolbarAction
              active={generationMode === GENERATION_MODE_OUTPAINT}
              label="Outpaint frame"
              onSelect={() => {
                setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
                setGenerationMode(GENERATION_MODE_OUTPAINT);
              }}
            >
              <Frame size={18} />
            </ToolbarAction>,
          )}
          {renderToolbarItem(
            <ToolbarAction
              active={generationMode === GENERATION_MODE_INPAINT}
              label="Inpaint mask"
              onSelect={() => {
                setWorkspaceMode(WORKSPACE_MODE_FREE_EDIT);
                setGenerationMode(GENERATION_MODE_INPAINT);
              }}
            >
              <Paintbrush size={18} />
            </ToolbarAction>,
          )}
          {CONTROLNET_GUIDE_UI_ENABLED
            ? renderToolbarItem(
                <ToolbarAction
                  active={tool === TOOL_CONTROL_GUIDE}
                  label="Sketch guide"
                  onSelect={() => setTool(TOOL_CONTROL_GUIDE)}
                >
                  <Palette size={18} />
                </ToolbarAction>,
              )
            : null}
        </div>
        <div className="toolbar-divider" />
        <div className="toolbar-section" aria-label="Canvas tools">
          {renderToolbarItem(
            <ToolbarTool
              tool={TOOL_PAN}
              currentTool={tool}
              label="Pan"
              onSelect={setTool}
            >
              <Hand size={18} />
            </ToolbarTool>,
          )}
          {renderToolbarItem(
            <ToolbarTool
              tool={TOOL_ERASE}
              currentTool={tool}
              label="Erase pixels"
              onSelect={setTool}
            >
              <Eraser size={18} />
            </ToolbarTool>,
          )}
        </div>
        <div className="toolbar-divider" />
        <div className="toolbar-section" aria-label="Image import">
          {renderToolbarItem(
            <ToolbarFileButton
              label="Add images"
              accept="image/*"
              multiple
              dataTourId={ONBOARDING_TARGET_UPLOAD_BUTTON}
              onChange={handleRasterFile}
            >
              <Upload size={18} />
            </ToolbarFileButton>,
          )}
        </div>
        {pluginTools.length > 0 ? (
          <>
            <div className="toolbar-divider" />
            <div className="toolbar-section" aria-label="Plugin tools">
              {pluginTools.map((pluginTool) =>
                renderToolbarItem(
                  <PluginToolbarTool
                    tool={pluginTool}
                    active={activePluginToolId === pluginTool.id}
                    onSelect={() => {
                      if (pluginTool.target === PLUGIN_TOOL_TARGET_FRAME) {
                        setCanvasSelectionTarget({ kind: "frame" });
                      }
                      if (
                        pluginTool.target === PLUGIN_TOOL_TARGET_IMAGE &&
                        canvasSelectionTarget.kind === "frame"
                      ) {
                        setCanvasSelectionTarget({ kind: "none" });
                      }
                      setTool(pluginEditorToolId(pluginTool.id));
                    }}
                  />,
                  pluginTool.id,
                ),
              )}
            </div>
          </>
        ) : null}
        <div className="toolbar-spacer" />
        <div
          className="toolbar-section toolbar-section-secondary"
          aria-label="Edit history"
        >
          {renderToolbarItem(
            <TooltipButton label="Undo" onClick={undo}>
              <RotateCcw size={18} />
            </TooltipButton>,
          )}
          {renderToolbarItem(
            <TooltipButton label="Redo" onClick={redo}>
              <RotateCw size={18} />
            </TooltipButton>,
          )}
        </div>
        <div
          className="toolbar-section toolbar-section-secondary"
          aria-label="Project files"
        >
          {renderToolbarItem(
            <TooltipButton
              label="Save .expd project"
              onClick={() => {
                saveProjectArchive(documentState).catch((error) =>
                  setErrorMessage(
                    error instanceof Error
                      ? error.message
                      : "Project save failed.",
                  ),
                );
              }}
            >
              <Save size={18} />
            </TooltipButton>,
          )}
          {renderToolbarItem(
            <ToolbarFileButton
              label="Load .expd project"
              accept=".expd"
              onChange={loadProject}
            >
              <FileArchive size={18} />
            </ToolbarFileButton>,
          )}
        </div>
      </aside>
    </>
  );
}

function ToolbarItem({ children }: { children: ReactNode }) {
  return <div className="toolbar-item">{children}</div>;
}

function PluginToolbarTool({
  tool,
  active,
  onSelect,
}: {
  tool: PluginToolInfo;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <TooltipButton
      label={tool.label}
      active={active}
      className="plugin-toolbar-button"
      style={pluginToolStyle(tool)}
      onClick={onSelect}
    >
      {renderPluginToolIcon(tool.icon)}
    </TooltipButton>
  );
}

interface ToolbarFileButtonProps {
  label: string;
  accept: string;
  multiple?: boolean;
  dataTourId?: string;
  children: ReactNode;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}

function ToolbarFileButton({
  label,
  accept,
  multiple = false,
  dataTourId,
  children,
  onChange,
}: ToolbarFileButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="icon-file-button"
          aria-label={label}
          data-tour-id={dataTourId}
          onClick={() => inputRef.current?.click()}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="right" sideOffset={10}>
        {label}
      </TooltipContent>
      <input
        ref={inputRef}
        className="toolbar-file-input"
        type="file"
        accept={accept}
        multiple={multiple}
        tabIndex={-1}
        onChange={onChange}
      />
    </Tooltip>
  );
}

interface ToolbarActionProps {
  active: boolean;
  label: string;
  children: ReactNode;
  onSelect: () => void;
}

function ToolbarAction({
  active,
  label,
  children,
  onSelect,
}: ToolbarActionProps) {
  return (
    <TooltipButton label={label} active={active} onClick={onSelect}>
      {children}
    </TooltipButton>
  );
}

interface ToolbarToolProps {
  tool: EditorTool;
  currentTool: EditorTool;
  label: string;
  children: ReactNode;
  onSelect: (tool: EditorTool) => void;
}

function ToolbarTool({
  tool,
  currentTool,
  label,
  children,
  onSelect,
}: ToolbarToolProps) {
  return (
    <TooltipButton
      label={label}
      active={tool === currentTool}
      onClick={() => onSelect(tool)}
    >
      {children}
    </TooltipButton>
  );
}

function renderPluginToolIcon(icon: string) {
  if (icon === "captions") {
    return <Captions size={18} />;
  }
  if (icon === "palette") {
    return <Palette size={18} />;
  }
  if (icon === "text-search") {
    return <TextSearch size={18} />;
  }
  if (icon === "sliders-horizontal") {
    return <SlidersHorizontal size={18} />;
  }
  return <Puzzle size={18} />;
}

function pluginToolStyle(tool: PluginToolInfo): CSSProperties {
  return {
    "--plugin-tool-icon-color": tool.icon_color ?? "var(--color-accent-violet)",
    "--plugin-tool-accent-color":
      tool.accent_color ?? tool.icon_color ?? "var(--color-accent-violet)",
  } as CSSProperties;
}
