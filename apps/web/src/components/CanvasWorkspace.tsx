import Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useEffect, useRef, useState } from "react";
import {
  Circle,
  Group,
  Image,
  Layer,
  Line,
  Rect,
  Stage,
  Text,
  Transformer,
} from "react-konva";
import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import {
  DEFAULT_CONTROL_GUIDE_STRENGTH,
  GENERATION_MODE_OUTPAINT,
  MAX_CONTROL_GUIDE_STRENGTH,
  MAX_ZOOM,
  MIN_CONTROL_GUIDE_STRENGTH,
  MIN_ZOOM,
  TOOL_CONTROL_GUIDE,
  TOOL_ERASE,
  TOOL_INPAINT_MASK,
  TOOL_OUTPAINT_FRAME,
  TOOL_PAN,
  TOOL_SELECT,
  ZOOM_STEP,
  STROKE_PAINT,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  isPluginEditorTool,
  pluginToolIdFromEditorTool,
  PLUGIN_TOOL_TARGET_IMAGE,
} from "../constants/domain";
import type {
  CanvasSelectionTarget,
  ControlStroke,
  DocumentBounds,
  MaskStroke,
  Point,
  ReferenceImageLayer,
} from "../domain/types";
import { useImageElement } from "../hooks/useImageElement";
import { useStudioQueries } from "../hooks/useStudioQueries";
import { eraseRasterStroke } from "../lib/canvasRender";
import { useEditorStore } from "../store/editorStore";
import { CANVAS_THEME } from "../theme/canvasTheme";
import { Button } from "./ui/button";

const CANVAS_GRID_MIN = -4096;
const CANVAS_GRID_MAX = 4096;
const CANVAS_GRID_STEP = 64;
const CANVAS_GRID_MAJOR_STEP = 256;

export function CanvasWorkspace() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const selectionRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const [size, setSize] = useState({ width: 900, height: 700 });
  const [isDrawing, setIsDrawing] = useState(false);
  const [erasePoints, setErasePoints] = useState<Point[]>([]);
  const [cursorPoint, setCursorPoint] = useState<Point | null>(null);
  const [shiftPressed, setShiftPressed] = useState(false);
  const { pluginToolsQuery } = useStudioQueries();

  const documentState = useEditorStore((state) => state.document);
  const canvasSelectionTarget = useEditorStore(
    (state) => state.canvasSelectionTarget,
  );
  const activeReferenceId =
    canvasSelectionTarget.kind === "reference" &&
    documentState.references.some(
      (reference) => reference.id === canvasSelectionTarget.id,
    )
      ? canvasSelectionTarget.id
      : null;
  const rasterSelected =
    canvasSelectionTarget.kind === "raster" &&
    Boolean(documentState.rasterDataUrl);
  const viewport = useEditorStore((state) => state.viewport);
  const tool = useEditorStore((state) => state.tool);
  const generationMode = useEditorStore((state) => state.generationMode);
  const outpaintStrategy = useEditorStore(
    (state) => state.parameters.outpaint_strategy,
  );
  const directionalOutpaintActive =
    generationMode === GENERATION_MODE_OUTPAINT &&
    outpaintStrategy === OUTPAINT_STRATEGY_DIRECTIONAL;
  const panModifierActive = shiftPressed;
  const panning = tool === TOOL_PAN || panModifierActive;
  const frameToolActive = tool === TOOL_OUTPAINT_FRAME && !panModifierActive;
  const selectToolActive = tool === TOOL_SELECT && !panModifierActive;
  const pluginToolActive = isPluginEditorTool(tool) && !panModifierActive;
  const activePluginToolId = pluginToolIdFromEditorTool(tool);
  const activePluginTool = (pluginToolsQuery.data ?? []).find(
    (pluginTool) => pluginTool.id === activePluginToolId,
  );
  const pluginImageToolActive =
    pluginToolActive && activePluginTool?.target === PLUGIN_TOOL_TARGET_IMAGE;
  const pluginFrameToolActive = pluginToolActive && !pluginImageToolActive;
  const frameSelected = canvasSelectionTarget.kind === "frame";
  const frameVisible =
    pluginFrameToolActive ||
    (generationMode === GENERATION_MODE_OUTPAINT &&
      !directionalOutpaintActive &&
      (frameToolActive || selectToolActive));
  const frameSelectionActive =
    pluginFrameToolActive ||
    (generationMode === GENERATION_MODE_OUTPAINT &&
      !directionalOutpaintActive &&
      (frameToolActive || (selectToolActive && frameSelected)));
  const imageSelectionToolActive = selectToolActive || pluginImageToolActive;
  const rasterImage = useImageElement(documentState.rasterDataUrl);
  const pluginPreview = useEditorStore((state) => state.pluginPreview);
  const activePluginPreview =
    pluginPreview &&
    pluginPreview.toolId === activePluginToolId &&
    targetsMatch(pluginPreview.target, canvasSelectionTarget)
      ? pluginPreview
      : null;
  const pluginPreviewImage = useImageElement(activePluginPreview?.image ?? null);
  const rasterDisplayImage =
    activePluginPreview?.target.kind === "raster" && pluginPreviewImage
      ? pluginPreviewImage
      : rasterImage;
  const pendingResults = useEditorStore((state) => state.pendingResults);
  const pendingResultBounds = useEditorStore(
    (state) => state.pendingResultBounds,
  );
  const pendingResultReplacesDocument = useEditorStore(
    (state) => state.pendingResultReplacesDocument,
  );
  const selectedResultIndex = useEditorStore(
    (state) => state.selectedResultIndex,
  );
  const brushSize = useEditorStore((state) => state.brushSize);
  const controlGuideEnabled = useEditorStore(
    (state) => state.controlGuideEnabled,
  );
  const controlGuideColor = useEditorStore((state) => state.controlGuideColor);
  const controlGuideStrength = useEditorStore(
    (state) => state.controlGuideStrength,
  );
  const eraserHardness = useEditorStore((state) => state.eraserHardness);
  const previewImage = useImageElement(
    pendingResults[selectedResultIndex] ?? null,
  );
  const pendingReplacementActive =
    pendingResultReplacesDocument && pendingResults.length > 0;
  const currentJob = useEditorStore((state) => state.currentJob);
  const currentJobSelection = useEditorStore(
    (state) => state.currentJobSelection,
  );
  const setViewport = useEditorStore((state) => state.setViewport);
  const setCanvasSelectionTarget = useEditorStore(
    (state) => state.setCanvasSelectionTarget,
  );
  const moveSelection = useEditorStore((state) => state.moveSelection);
  const setSelection = useEditorStore((state) => state.setSelection);
  const replaceRasterDataUrl = useEditorStore(
    (state) => state.replaceRasterDataUrl,
  );
  const updateRasterBounds = useEditorStore(
    (state) => state.updateRasterBounds,
  );
  const beginMaskStroke = useEditorStore((state) => state.beginMaskStroke);
  const appendMaskPoint = useEditorStore((state) => state.appendMaskPoint);
  const beginControlStroke = useEditorStore((state) => state.beginControlStroke);
  const appendControlPoint = useEditorStore((state) => state.appendControlPoint);
  const selectResult = useEditorStore((state) => state.selectResult);
  const acceptSelectedResult = useEditorStore(
    (state) => state.acceptSelectedResult,
  );
  const rejectResults = useEditorStore((state) => state.rejectResults);
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const transformer = transformerRef.current;
    const selection = selectionRef.current;
    if (!transformer || !selection || !frameSelectionActive) {
      transformer?.nodes([]);
      transformer?.getLayer()?.batchDraw();
      return;
    }
    transformer.nodes([selection]);
    transformer.getLayer()?.batchDraw();
  }, [frameSelectionActive, documentState.selection]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Shift") {
        setShiftPressed(true);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Shift") {
        setShiftPressed(false);
      }
    };
    const handleBlur = () => setShiftPressed(false);

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  const getDocumentPoint = (): Point | null => {
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition();
    if (!stage || !pointer) {
      return null;
    }
    return {
      x: (pointer.x - viewport.x) / viewport.zoom,
      y: (pointer.y - viewport.y) / viewport.zoom,
    };
  };

  const handleWheel = (event: KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault();
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition();
    if (!stage || !pointer) {
      return;
    }
    const direction = event.evt.deltaY > 0 ? -1 : 1;
    const nextZoom = clampZoom(
      direction > 0 ? viewport.zoom * ZOOM_STEP : viewport.zoom / ZOOM_STEP,
    );
    const mousePoint = {
      x: (pointer.x - viewport.x) / viewport.zoom,
      y: (pointer.y - viewport.y) / viewport.zoom,
    };
    setViewport(
      pointer.x - mousePoint.x * nextZoom,
      pointer.y - mousePoint.y * nextZoom,
      nextZoom,
    );
  };

  const handleStageDragEnd = (event: KonvaEventObject<DragEvent>) => {
    const stage = stageRef.current;
    if (!stage || event.target !== stage) {
      return;
    }
    setViewport(stage.x(), stage.y(), viewport.zoom);
  };

  const handlePointerDown = (event: KonvaEventObject<MouseEvent>) => {
    if (panning) {
      return;
    }
    if (pluginFrameToolActive && event.target === stageRef.current) {
      setCanvasSelectionTarget({ kind: "frame" });
      return;
    }
    if (tool === TOOL_SELECT && event.target === stageRef.current) {
      setCanvasSelectionTarget(
        generationMode === GENERATION_MODE_OUTPAINT
          ? { kind: "frame" }
          : { kind: "none" },
      );
    }
    if (tool === TOOL_CONTROL_GUIDE) {
      const point = getDocumentPoint();
      if (!point) {
        return;
      }
      beginControlStroke(point);
      setIsDrawing(true);
      return;
    }
    if (tool === TOOL_INPAINT_MASK) {
      const point = getDocumentPoint();
      if (!point) {
        return;
      }
      beginMaskStroke(STROKE_PAINT, point);
      setIsDrawing(true);
      return;
    }
    if (tool !== TOOL_ERASE) {
      return;
    }
    const point = getDocumentPoint();
    if (!point) {
      return;
    }
    setErasePoints([point]);
    setIsDrawing(true);
  };

  const handlePointerMove = () => {
    const point = getDocumentPoint();
    setCursorPoint(point);
    if (panning || !isDrawing) {
      return;
    }
    if (!point) {
      return;
    }
    if (tool === TOOL_CONTROL_GUIDE) {
      appendControlPoint(point);
      return;
    }
    if (tool === TOOL_INPAINT_MASK) {
      appendMaskPoint(point);
      return;
    }
    if (tool !== TOOL_ERASE) {
      return;
    }
    setErasePoints((points) => [...points, point]);
  };

  const handlePointerLeave = () => {
    setCursorPoint(null);
    handlePointerUp();
  };

  const handlePointerUp = () => {
    if (isDrawing && tool === TOOL_ERASE) {
      const points = erasePoints;
      setErasePoints([]);
      setIsDrawing(false);
      if (points.length > 0) {
        eraseRasterStroke(documentState, points, brushSize, eraserHardness)
          .then(replaceRasterDataUrl)
          .catch((error) =>
            setErrorMessage(
              error instanceof Error ? error.message : "Erase failed.",
            ),
          );
      }
      return;
    }
    if (isDrawing && (tool === TOOL_INPAINT_MASK || tool === TOOL_CONTROL_GUIDE)) {
      setIsDrawing(false);
      return;
    }
    setIsDrawing(false);
  };

  const handleSelectionTransformEnd = () => {
    const node = selectionRef.current;
    if (!node) {
      return;
    }
    const width = Math.max(16, node.width() * node.scaleX());
    const height = Math.max(16, node.height() * node.scaleY());
    node.scaleX(1);
    node.scaleY(1);
    setSelection({
      x: node.x(),
      y: node.y(),
      width,
      height,
    });
  };

  const jobRunning =
    currentJob?.status === "queued" || currentJob?.status === "running";
  const jobSelection = currentJobSelection ?? documentState.selection;
  const eraserHardnessRadius = getEraserHardnessRadius(
    brushSize,
    eraserHardness,
  );
  return (
    <main className="canvas-shell" ref={containerRef}>
      {previewImage ? (
        <ResultControls
          count={pendingResults.length}
          index={selectedResultIndex}
          onPrevious={() => selectResult(Math.max(0, selectedResultIndex - 1))}
          onNext={() =>
            selectResult(
              Math.min(pendingResults.length - 1, selectedResultIndex + 1),
            )
          }
          onAccept={acceptSelectedResult}
          onCancel={rejectResults}
        />
      ) : null}
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        x={viewport.x}
        y={viewport.y}
        scaleX={viewport.zoom}
        scaleY={viewport.zoom}
        draggable={panning}
        onDragEnd={handleStageDragEnd}
        onWheel={handleWheel}
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={handlePointerLeave}
      >
        <Layer listening={false}>
          <Rect
            x={-4000}
            y={-4000}
            width={8000}
            height={8000}
            fill={CANVAS_THEME.background}
          />
          <CanvasGrid />
        </Layer>
        <Layer>
          {!pendingReplacementActive && rasterDisplayImage ? (
            <RasterNode
              image={rasterDisplayImage}
              bounds={documentState.rasterBounds}
              selected={rasterSelected}
              selectable={imageSelectionToolActive}
              onSelect={() => {
                setCanvasSelectionTarget({ kind: "raster" });
              }}
              onChange={updateRasterBounds}
            />
          ) : null}
          {!pendingReplacementActive
            ? documentState.references.map((reference) => (
                <ReferenceNode
                  key={reference.id}
                  reference={reference}
                  selected={reference.id === activeReferenceId}
                  selectable={imageSelectionToolActive}
                  previewImage={
                    activePluginPreview?.target.kind === "reference" &&
                    activePluginPreview.target.id === reference.id
                      ? pluginPreviewImage
                      : null
                  }
                  onSelect={(id) => {
                    setCanvasSelectionTarget({ kind: "reference", id });
                  }}
                />
              ))
            : null}
          {previewImage ? (
            <Image
              image={previewImage}
              x={pendingResultBounds?.x ?? 0}
              y={pendingResultBounds?.y ?? 0}
              width={pendingResultBounds?.width ?? documentState.width}
              height={pendingResultBounds?.height ?? documentState.height}
              listening={false}
            />
          ) : null}
          {tool === TOOL_INPAINT_MASK
            ? documentState.maskStrokes.map((stroke) => (
                <MaskStrokeNode key={stroke.id} stroke={stroke} />
              ))
            : null}
          {documentState.controlStrokes.map((stroke) => (
            <ControlStrokeNode
              key={stroke.id}
              stroke={stroke}
              active={controlGuideEnabled}
            />
          ))}
          {tool === TOOL_ERASE && erasePoints.length > 0 ? (
            <Line
              points={erasePoints.flatMap((point) => [point.x, point.y])}
              stroke={CANVAS_THEME.eraserStroke}
              strokeWidth={brushSize}
              opacity={0.72}
              lineCap="round"
              lineJoin="round"
              listening={false}
            />
          ) : null}
          {tool === TOOL_ERASE && !panning && cursorPoint ? (
            <>
              <Circle
                x={cursorPoint.x}
                y={cursorPoint.y}
                radius={brushSize / 2}
                fill={CANVAS_THEME.eraserPreviewFill}
                stroke={CANVAS_THEME.eraserPreviewStroke}
                strokeWidth={1 / viewport.zoom}
                listening={false}
              />
              {eraserHardnessRadius > 0 &&
              eraserHardnessRadius < brushSize / 2 ? (
                <Circle
                  x={cursorPoint.x}
                  y={cursorPoint.y}
                  radius={eraserHardnessRadius}
                  stroke={CANVAS_THEME.eraserPreviewStroke}
                  strokeWidth={1 / viewport.zoom}
                  opacity={0.46}
                  listening={false}
                />
              ) : null}
            </>
          ) : null}
          {tool === TOOL_INPAINT_MASK && !panning && cursorPoint ? (
            <Circle
              x={cursorPoint.x}
              y={cursorPoint.y}
              radius={brushSize / 2}
              fill={CANVAS_THEME.inpaintMaskPreviewFill}
              stroke={CANVAS_THEME.inpaintMaskPreviewStroke}
              strokeWidth={1 / viewport.zoom}
              listening={false}
            />
          ) : null}
          {tool === TOOL_CONTROL_GUIDE && !panning && cursorPoint ? (
            <Circle
              x={cursorPoint.x}
              y={cursorPoint.y}
              radius={brushSize / 2}
              fill={controlGuideColor}
              stroke={controlGuideColor}
              opacity={controlGuideCursorOpacity(
                controlGuideEnabled,
                controlGuideStrength,
              )}
              strokeWidth={1 / viewport.zoom}
              listening={false}
            />
          ) : null}
          {jobRunning && currentJob ? (
            <JobProgressOverlay job={currentJob} selection={jobSelection} />
          ) : null}
          {frameVisible ? (
            <Rect
              ref={selectionRef}
              x={documentState.selection.x}
              y={documentState.selection.y}
              width={documentState.selection.width}
              height={documentState.selection.height}
              stroke={CANVAS_THEME.selectionStroke}
              strokeWidth={2 / viewport.zoom}
              dash={[10 / viewport.zoom, 6 / viewport.zoom]}
              fill={CANVAS_THEME.selectionFill}
              listening={frameSelectionActive && !panning}
              draggable={frameSelectionActive}
              onMouseDown={() => {
                setCanvasSelectionTarget({ kind: "frame" });
              }}
              onDragEnd={(event) =>
                moveSelection(event.target.x(), event.target.y())
              }
              onTransformEnd={handleSelectionTransformEnd}
            />
          ) : null}
          {frameSelectionActive ? (
            <Transformer
              ref={transformerRef}
              rotateEnabled={false}
              anchorSize={10 / viewport.zoom}
              borderStroke={CANVAS_THEME.selectionStroke}
              anchorStroke={CANVAS_THEME.selectionStroke}
              anchorFill={CANVAS_THEME.transformAnchorFill}
            />
          ) : null}
        </Layer>
      </Stage>
    </main>
  );
}

function CanvasGrid() {
  const coordinates = [];
  for (
    let coordinate = CANVAS_GRID_MIN;
    coordinate <= CANVAS_GRID_MAX;
    coordinate += CANVAS_GRID_STEP
  ) {
    coordinates.push(coordinate);
  }
  return (
    <>
      {coordinates.map((coordinate) => {
        const major = coordinate % CANVAS_GRID_MAJOR_STEP === 0;
        return (
          <Line
            key={`grid-x-${coordinate}`}
            points={[coordinate, CANVAS_GRID_MIN, coordinate, CANVAS_GRID_MAX]}
            stroke={
              major ? CANVAS_THEME.gridMajorStroke : CANVAS_THEME.gridStroke
            }
            strokeWidth={major ? 1.2 : 1}
            listening={false}
          />
        );
      })}
      {coordinates.map((coordinate) => {
        const major = coordinate % CANVAS_GRID_MAJOR_STEP === 0;
        return (
          <Line
            key={`grid-y-${coordinate}`}
            points={[CANVAS_GRID_MIN, coordinate, CANVAS_GRID_MAX, coordinate]}
            stroke={
              major ? CANVAS_THEME.gridMajorStroke : CANVAS_THEME.gridStroke
            }
            strokeWidth={major ? 1.2 : 1}
            listening={false}
          />
        );
      })}
    </>
  );
}

function MaskStrokeNode({ stroke }: { stroke: MaskStroke }) {
  if (stroke.mode !== STROKE_PAINT || stroke.points.length === 0) {
    return null;
  }
  if (stroke.points.length === 1) {
    return (
      <Circle
        x={stroke.points[0].x}
        y={stroke.points[0].y}
        radius={stroke.size / 2}
        fill={CANVAS_THEME.inpaintMaskStroke}
        opacity={0.42}
        listening={false}
      />
    );
  }
  return (
    <Line
      points={stroke.points.flatMap((point) => [point.x, point.y])}
      stroke={CANVAS_THEME.inpaintMaskStroke}
      strokeWidth={stroke.size}
      opacity={0.42}
      lineCap="round"
      lineJoin="round"
      listening={false}
    />
  );
}

function ControlStrokeNode({
  stroke,
  active,
}: {
  stroke: ControlStroke;
  active: boolean;
}) {
  if (stroke.points.length === 0) {
    return null;
  }
  const color = stroke.color ?? CANVAS_THEME.controlGuideStroke;
  const opacity = active ? controlStrokeOpacity(stroke.strength) : 0.26;
  if (stroke.points.length === 1) {
    return (
      <Circle
        x={stroke.points[0].x}
        y={stroke.points[0].y}
        radius={stroke.size / 2}
        fill={color}
        opacity={opacity}
        listening={false}
      />
    );
  }
  return (
    <Line
      points={stroke.points.flatMap((point) => [point.x, point.y])}
      stroke={color}
      strokeWidth={stroke.size}
      opacity={opacity}
      lineCap="round"
      lineJoin="round"
      listening={false}
    />
  );
}

function controlGuideCursorOpacity(active: boolean, strength: number): number {
  if (!active) {
    return 0.1;
  }
  return 0.08 + normalizedControlStrength(strength) * 0.28;
}

function controlStrokeOpacity(strength: number | undefined): number {
  const nextStrength = typeof strength === "number" && Number.isFinite(strength)
    ? strength
    : DEFAULT_CONTROL_GUIDE_STRENGTH;
  return 0.18 + normalizedControlStrength(nextStrength) * 0.56;
}

function normalizedControlStrength(strength: number): number {
  return (
    Math.max(
      MIN_CONTROL_GUIDE_STRENGTH,
      Math.min(MAX_CONTROL_GUIDE_STRENGTH, strength),
    ) / MAX_CONTROL_GUIDE_STRENGTH
  );
}

function RasterNode({
  image,
  bounds,
  selected,
  selectable,
  onSelect,
  onChange,
}: {
  image: HTMLImageElement;
  bounds: DocumentBounds | null;
  selected: boolean;
  selectable: boolean;
  onSelect: () => void;
  onChange: (bounds: DocumentBounds) => void;
}) {
  const imageRef = useRef<Konva.Image>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const imageBounds = bounds ?? {
    x: 0,
    y: 0,
    width: image.naturalWidth,
    height: image.naturalHeight,
  };

  useEffect(() => {
    const transformer = transformerRef.current;
    const imageNode = imageRef.current;
    if (!transformer || !imageNode || !selected || !selectable) {
      transformer?.nodes([]);
      transformer?.getLayer()?.batchDraw();
      return;
    }
    transformer.nodes([imageNode]);
    transformer.getLayer()?.batchDraw();
  }, [selected, selectable, imageBounds.width, imageBounds.height]);

  const finishTransform = () => {
    const node = imageRef.current;
    if (!node) {
      return;
    }
    const width = Math.max(16, node.width() * node.scaleX());
    const height = Math.max(16, node.height() * node.scaleY());
    node.scaleX(1);
    node.scaleY(1);
    onChange({
      x: Math.round(node.x()),
      y: Math.round(node.y()),
      width: Math.round(width),
      height: Math.round(height),
    });
  };

  return (
    <>
      <Image
        ref={imageRef}
        image={image}
        x={imageBounds.x}
        y={imageBounds.y}
        width={imageBounds.width}
        height={imageBounds.height}
        listening={selectable}
        draggable={selectable}
        onMouseDown={(event) => {
          event.cancelBubble = true;
          onSelect();
        }}
        onTap={(event) => {
          event.cancelBubble = true;
          onSelect();
        }}
        onDragEnd={(event) =>
          onChange({
            ...imageBounds,
            x: Math.round(event.target.x()),
            y: Math.round(event.target.y()),
          })
        }
        onTransformEnd={finishTransform}
      />
      {selected && selectable ? (
        <Transformer
          ref={transformerRef}
          rotateEnabled={false}
          anchorSize={10}
          borderStroke={CANVAS_THEME.activeLayerStroke}
          anchorStroke={CANVAS_THEME.activeLayerStroke}
          anchorFill={CANVAS_THEME.transformAnchorFill}
          boundBoxFunc={(oldBox, newBox) =>
            newBox.width < 16 || newBox.height < 16 ? oldBox : newBox
          }
        />
      ) : null}
    </>
  );
}

function ResultControls({
  count,
  index,
  onPrevious,
  onNext,
  onAccept,
  onCancel,
}: {
  count: number;
  index: number;
  onPrevious: () => void;
  onNext: () => void;
  onAccept: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="result-controls-overlay">
      <Button
        type="button"
        variant="secondary"
        size="compact"
        disabled={index <= 0}
        onClick={onPrevious}
      >
        <ChevronLeft size={16} />
      </Button>
      <span className="result-counter">
        {index + 1} / {count}
      </span>
      <Button
        type="button"
        variant="secondary"
        size="compact"
        disabled={index >= count - 1}
        onClick={onNext}
      >
        <ChevronRight size={16} />
      </Button>
      <Button type="button" variant="accept" size="compact" onClick={onAccept}>
        <Check size={16} />
        Accept
      </Button>
      <Button
        type="button"
        variant="secondary"
        size="compact"
        onClick={onCancel}
      >
        <X size={16} />
        Cancel
      </Button>
    </div>
  );
}

function JobProgressOverlay({
  job,
  selection,
}: {
  job: { progress: number; message: string };
  selection: { x: number; y: number; width: number; height: number };
}) {
  const progress = Math.round(job.progress * 100);
  return (
    <Group listening={false}>
      <Rect
        x={selection.x}
        y={selection.y}
        width={selection.width}
        height={selection.height}
        fill={CANVAS_THEME.jobOverlayFill}
        stroke={CANVAS_THEME.activeLayerStroke}
        strokeWidth={2}
      />
      <Rect
        x={selection.x}
        y={selection.y + selection.height - 8}
        width={Math.max(2, selection.width * job.progress)}
        height={8}
        fill={CANVAS_THEME.activeLayerStroke}
      />
      <Text
        x={selection.x + 12}
        y={selection.y + Math.max(12, selection.height / 2 - 16)}
        width={Math.max(40, selection.width - 24)}
        align="center"
        text={`${progress}%\n${job.message}`}
        fill={CANVAS_THEME.jobText}
        fontSize={Math.max(11, Math.min(16, selection.width / 14))}
        fontStyle="bold"
        fontFamily="Inter, system-ui, sans-serif"
      />
    </Group>
  );
}

function ReferenceNode({
  reference,
  selected,
  selectable,
  previewImage,
  onSelect,
}: {
  reference: ReferenceImageLayer;
  selected: boolean;
  selectable: boolean;
  previewImage: HTMLImageElement | null;
  onSelect: (id: string) => void;
}) {
  const imageRef = useRef<Konva.Image>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const referenceImage = useImageElement(reference.dataUrl);
  const image = previewImage ?? referenceImage;
  const updateReference = useEditorStore((state) => state.updateReference);

  useEffect(() => {
    const transformer = transformerRef.current;
    const imageNode = imageRef.current;
    if (!transformer || !imageNode || !selected || !selectable) {
      transformer?.nodes([]);
      transformer?.getLayer()?.batchDraw();
      return;
    }
    transformer.nodes([imageNode]);
    transformer.getLayer()?.batchDraw();
  }, [selected, selectable, reference.width, reference.height]);

  if (!image) {
    return null;
  }

  const finishTransform = () => {
    const node = imageRef.current;
    if (!node) {
      return;
    }
    const width = Math.max(16, node.width() * node.scaleX());
    const height = Math.max(16, node.height() * node.scaleY());
    node.scaleX(1);
    node.scaleY(1);
    updateReference(reference.id, {
      x: node.x(),
      y: node.y(),
      width,
      height,
    });
  };

  return (
    <>
      <Image
        ref={imageRef}
        image={image}
        x={reference.x}
        y={reference.y}
        width={reference.width}
        height={reference.height}
        opacity={reference.opacity}
        listening={selectable}
        draggable={selectable}
        onMouseDown={(event) => {
          event.cancelBubble = true;
          onSelect(reference.id);
        }}
        onTap={(event) => {
          event.cancelBubble = true;
          onSelect(reference.id);
        }}
        onDragEnd={(event) =>
          updateReference(reference.id, {
            x: event.target.x(),
            y: event.target.y(),
          })
        }
        onTransformEnd={finishTransform}
      />
      {selected && selectable ? (
        <Transformer
          ref={transformerRef}
          rotateEnabled={false}
          anchorSize={10}
          borderStroke={CANVAS_THEME.activeLayerStroke}
          anchorStroke={CANVAS_THEME.activeLayerStroke}
          anchorFill={CANVAS_THEME.transformAnchorFill}
          boundBoxFunc={(oldBox, newBox) =>
            newBox.width < 16 || newBox.height < 16 ? oldBox : newBox
          }
        />
      ) : null}
    </>
  );
}

function clampZoom(value: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
}

function getEraserHardnessRadius(size: number, hardness: number): number {
  const normalizedHardness = Number.isFinite(hardness)
    ? Math.max(0, Math.min(100, hardness))
    : 100;
  return (size / 2) * (normalizedHardness / 100);
}

function targetsMatch(
  first: CanvasSelectionTarget,
  second: CanvasSelectionTarget,
): boolean {
  if (first.kind !== second.kind) {
    return false;
  }
  if (first.kind === "reference" && second.kind === "reference") {
    return first.id === second.id;
  }
  return first.kind === "raster";
}
