import {
  BLACK_PIXEL,
  CONDITIONING_TYPE_COLOR,
  CONTROL_GUIDE_MASK_MODE_REPLACE,
  DEFAULT_CONTROL_GUIDE_MASK_MODE,
  DEFAULT_CONTROL_GUIDE_COLOR,
  DEFAULT_CONTROL_GUIDE_STRENGTH,
  DEFAULT_ERASER_HARDNESS,
  DEFAULT_SELECTION_SIZE,
  DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE,
  DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
  DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
  DIRECTIONAL_OUTPAINT_MIN_SIZE,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  MAX_CONTROL_GUIDE_STRENGTH,
  MAX_ERASER_HARDNESS,
  MIN_CONTROL_GUIDE_STRENGTH,
  MIN_ERASER_HARDNESS,
  OUTPAINT_DIRECTION_AROUND,
  OUTPAINT_DIRECTION_DOWN,
  OUTPAINT_DIRECTION_LEFT,
  OUTPAINT_DIRECTION_RIGHT,
  OUTPAINT_DIRECTION_UP,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  OUTPAINT_STRATEGY_SELECTED_FRAME,
  OUTPAINT_STRATEGY_WHOLE_RESIZED,
  RASTER_ASSET_PATH,
  STROKE_ERASE,
  WHITE_PIXEL,
  type ControlGuideMaskMode,
  type GenerationMode,
  type OutpaintDirection,
  type OutpaintStrategy,
} from '../constants/domain'
import { AppError } from './errors'
import type {
  CanvasSelectionTarget,
  ControlStroke,
  DocumentBounds,
  EditorDocument,
  GenerationConditioning,
  Point,
  PreparedRasterImport,
  SelectionRect,
} from '../domain/types'
import { CANVAS_THEME } from '../theme/canvasTheme'

const MAX_IMPORT_CANVAS_SIZE = 4096
const MIN_IMPORT_IMAGE_EDGE = 64
const MAX_INPAINT_INPUT_SIZE = 1024
const INPAINT_MASK_CONTEXT_PADDING = 256
const MAX_PLUGIN_CANVAS_TARGET_SIZE = 1024
const OUTPAINT_CONTEXT_SIZE = 512
const OUTPAINT_DEFAULT_MAX_WIDTH = 1536
const OUTPAINT_DEFAULT_MAX_HEIGHT = 1024
const MAX_IMPORT_IMAGE_EDGE = OUTPAINT_DEFAULT_MAX_WIDTH
const WHOLE_RESIZED_MAX_INPUT_EDGE = 1024
const HF_SPACE_FILL_SIZE_MULTIPLE = 8
const TRANSPARENT_ALPHA = 0
const OUTPAINT_COMPOSITION_FEATHER_MAX = 32
const CONTROL_GUIDE_FALLBACK_RGB: [number, number, number] = [127, 127, 127]
const SEMANTIC_MASK_OVERLAY_RGB: [number, number, number] = [18, 184, 200]
const SEMANTIC_MASK_OVERLAY_ALPHA = 112
const CONTROL_GUIDE_NEIGHBORS: Array<[number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
]

interface RenderedGenerationInputs {
  image: string
  mask: string
  compositionMask: string | null
  selection: SelectionRect
  previewSelection: SelectionRect
  conditioning: GenerationConditioning | null
  replaceDocument: boolean
  directionalPlan: DirectionalOutpaintPlan | null
  renderSize?: { width: number; height: number }
}

export interface DirectionalOutpaintPlan {
  direction: OutpaintDirection
  width: number
  height: number
  generatedSize: number
  contextSize: number
  crossSize: number
  scale: number
  visibleBounds: DocumentBounds
  contextRect: DocumentBounds
  generatedRect: DocumentBounds
  drawRect: DocumentBounds
}

export interface HfSpaceFillExpansionPlan {
  selection: SelectionRect
  sourceRect: DocumentBounds
  preservedRect: DocumentBounds
  resizePercentage: number
  overlapPercentage: number
  overlap: { x: number; y: number }
  renderSize: { width: number; height: number }
}

/**
 * Create a blank PNG data URL.
 *
 * @param width - Canvas width.
 * @param height - Canvas height.
 * @param fillStyle - CSS fill color.
 * @returns PNG data URL.
 */
export function createBlankDataUrl(width: number, height: number, fillStyle: string): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = requireCanvasContext(canvas)
  context.fillStyle = fillStyle
  context.fillRect(0, 0, width, height)
  return canvas.toDataURL('image/png')
}

/**
 * Load an image element from a data URL.
 *
 * @param dataUrl - Image data URL.
 * @returns Loaded image element.
 */
export function loadImageElement(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new AppError('IMAGE_LOAD_FAILED', 'Image could not be loaded.'))
    image.src = dataUrl
  })
}

/**
 * Read a file as a browser data URL.
 *
 * @param file - Source file.
 * @returns Data URL.
 */
export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new AppError('FILE_READ_FAILED', 'File could not be read.'))
    reader.readAsDataURL(file)
  })
}

/**
 * Measure an image data URL.
 *
 * @param dataUrl - Source image.
 * @returns Intrinsic dimensions.
 */
export async function measureImage(dataUrl: string): Promise<{ width: number; height: number }> {
  const image = await loadImageElement(dataUrl)
  return { width: image.naturalWidth, height: image.naturalHeight }
}

export async function prepareRasterImport(dataUrl: string): Promise<PreparedRasterImport> {
  const image = await loadImageElement(dataUrl)
  const normalized = normalizeImportImage(image, dataUrl)
  const raster = trimTransparentImageBounds(
    normalized.source,
    normalized.dataUrl,
    normalized.width,
    normalized.height,
  )
  const sourceWidth = raster.width
  const sourceHeight = raster.height
  const padding = calculateImportPadding(sourceWidth, sourceHeight)
  const width = sourceWidth + padding * 2
  const height = sourceHeight + padding * 2
  return {
    dataUrl: raster.dataUrl,
    width,
    height,
    rasterBounds: {
      x: padding,
      y: padding,
      width: sourceWidth,
      height: sourceHeight,
    },
    selection: createInitialOutpaintSelection(sourceWidth, sourceHeight, padding, width, height),
  }
}

/**
 * Compose source image and mask payloads for backend generation.
 *
 * @param documentState - Editor document.
 * @param mode - Generation operation mode.
 * @returns Image and mask data URLs.
 */
export async function renderGenerationInputs(
  documentState: EditorDocument,
  mode: GenerationMode,
  options: {
    includeControlGuide?: boolean
    controlGuideMaskMode?: ControlGuideMaskMode
    outpaintStrategy?: OutpaintStrategy
    outpaintMaxWidth?: number
    outpaintMaxHeight?: number
    outpaintDirection?: OutpaintDirection
    outpaintGeneratedSize?: number
    outpaintContextSize?: number
    outpaintCrossSize?: number
    hfSpaceOverlapPercentage?: number
    hfSpaceFixedExpansion?: boolean
    hfSpaceResizeOption?: string
    hfSpaceCustomResizePercentage?: number
    hfSpaceOverlapLeft?: boolean
    hfSpaceOverlapRight?: boolean
    hfSpaceOverlapTop?: boolean
    hfSpaceOverlapBottom?: boolean
    fixedExpandPercent?: number
    fixedExpandWidthPercent?: number
    fixedExpandHeightPercent?: number
    fixedExpandOutputScalePercent?: number
  } = {},
): Promise<RenderedGenerationInputs> {
  if (
    mode === GENERATION_MODE_OUTPAINT &&
    options.outpaintStrategy === OUTPAINT_STRATEGY_HF_SPACE_FILL &&
    options.hfSpaceFixedExpansion
  ) {
    return renderHfSpaceFillGenerationInputs(documentState, documentState.selection, {
      direction: options.outpaintDirection,
      generatedSize: options.outpaintGeneratedSize,
      crossSize: options.outpaintCrossSize,
      overlapPercentage: options.hfSpaceOverlapPercentage,
      fixedExpansion: options.hfSpaceFixedExpansion,
      resizeOption: options.hfSpaceResizeOption,
      customResizePercentage: options.hfSpaceCustomResizePercentage,
      overlapLeft: options.hfSpaceOverlapLeft,
      overlapRight: options.hfSpaceOverlapRight,
      overlapTop: options.hfSpaceOverlapTop,
      overlapBottom: options.hfSpaceOverlapBottom,
      expansionPercent: options.fixedExpandPercent,
      widthExpansionPercent: options.fixedExpandWidthPercent,
      heightExpansionPercent: options.fixedExpandHeightPercent,
      outputScalePercent: options.fixedExpandOutputScalePercent,
    })
  }
  const fullCanvas = await renderDocumentCanvas(documentState)
  if (
    mode === GENERATION_MODE_OUTPAINT &&
    options.outpaintStrategy === OUTPAINT_STRATEGY_DIRECTIONAL
  ) {
    return renderDirectionalGenerationInputs(fullCanvas, {
      direction: options.outpaintDirection,
      generatedSize: options.outpaintGeneratedSize,
      contextSize: options.outpaintContextSize,
      crossSize: options.outpaintCrossSize,
    })
  }
  let inpaintMaskCanvas: HTMLCanvasElement | null = null
  let renderPlan: { selection: SelectionRect; renderWidth: number; renderHeight: number }
  if (mode === GENERATION_MODE_INPAINT) {
    inpaintMaskCanvas = await renderInpaintMaskCanvas(documentState)
    renderPlan = {
      selection: visibleContentSelectionFromMask(documentState, fullCanvas, inpaintMaskCanvas),
      renderWidth: 0,
      renderHeight: 0,
    }
  } else {
    renderPlan = outpaintRenderPlanFromCanvas(
      documentState.selection,
      fullCanvas,
      options.outpaintStrategy ?? OUTPAINT_STRATEGY_LOCAL_CONTEXT,
      {
        maxWidth: options.outpaintMaxWidth,
        maxHeight: options.outpaintMaxHeight,
      },
    )
  }
  const selection = renderPlan.selection
  const renderWidth = renderPlan.renderWidth || selection.width
  const renderHeight = renderPlan.renderHeight || selection.height
  const imageCanvas = document.createElement('canvas')
  imageCanvas.width = renderWidth
  imageCanvas.height = renderHeight
  const imageContext = requireCanvasContext(imageCanvas)
  imageContext.clearRect(0, 0, renderWidth, renderHeight)
  drawSelectionToCanvas(imageContext, fullCanvas, selection, renderWidth, renderHeight)

  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = renderWidth
  maskCanvas.height = renderHeight
  const maskContext = requireCanvasContext(maskCanvas)
  if (mode === GENERATION_MODE_INPAINT) {
    if (!inpaintMaskCanvas) {
      throw new AppError('INPAINT_MASK_REQUIRED', 'Paint an inpaint mask before generating.')
    }
    maskContext.drawImage(inpaintMaskCanvas, -selection.x, -selection.y)
  } else {
    const imageData = imageContext.getImageData(0, 0, renderWidth, renderHeight)
    if (!imageDataHasVisiblePixels(imageData)) {
      throw new AppError(
        'OUTPAINT_CONTEXT_REQUIRED',
        'Move the outpaint frame next to visible image content before generating.',
      )
    }
    const maskImage = createOutpaintMaskImage(
      maskContext,
      imageData,
      freeOutpaintOverlapPixels(renderWidth, renderHeight, options.hfSpaceOverlapPercentage),
    )
    imageContext.putImageData(imageData, 0, 0)
    maskContext.putImageData(maskImage, 0, 0)
  }
  const controlGuide = options.includeControlGuide
    ? renderControlGuideCanvas(
        documentState,
        selection,
        imageCanvas,
        mode === GENERATION_MODE_INPAINT ? maskCanvas : null,
        options.controlGuideMaskMode ?? DEFAULT_CONTROL_GUIDE_MASK_MODE,
      )
    : null
  return {
    image: imageCanvas.toDataURL('image/png'),
    mask: maskCanvas.toDataURL('image/png'),
    compositionMask: mode === GENERATION_MODE_OUTPAINT
      ? maskCanvas.toDataURL('image/png')
      : null,
    selection,
    previewSelection: selection,
    replaceDocument: false,
    directionalPlan: null,
    conditioning: controlGuide
      ? {
          type: CONDITIONING_TYPE_COLOR,
          image: controlGuide.toDataURL('image/png'),
        }
      : null,
  }
}

/**
 * Render the current selection target as a plugin action image payload.
 *
 * @param documentState - Editor document.
 * @param target - Selected canvas target.
 * @returns Image data URL and target metadata.
 */
export async function renderPluginActionInput(
  documentState: EditorDocument,
  target: CanvasSelectionTarget,
): Promise<{ image: string; target: Record<string, unknown> }> {
  if (target.kind === 'canvas') {
    return renderCanvasActionInput(documentState, target)
  }
  if (target.kind === 'raster') {
    return renderRasterActionInput(documentState)
  }
  if (target.kind === 'reference') {
    return renderReferenceActionInput(documentState, target.id)
  }
  return renderFrameActionInput(documentState)
}

export async function eraseSemanticMaskFromDocument(
  documentState: EditorDocument,
  maskDataUrl: string,
): Promise<string> {
  const canvas = await renderDocumentCanvas(documentState)
  const maskCanvas = await renderMaskAlphaCanvas(
    maskDataUrl,
    documentState.width,
    documentState.height,
  )
  const context = requireCanvasContext(canvas)
  context.save()
  context.globalCompositeOperation = 'destination-out'
  context.drawImage(maskCanvas, 0, 0)
  context.restore()
  return canvas.toDataURL('image/png')
}

export async function renderMaskOverlayDataUrl(maskDataUrl: string): Promise<string> {
  const mask = await loadImageElement(maskDataUrl)
  const canvas = await renderMaskAlphaCanvas(maskDataUrl, mask.naturalWidth, mask.naturalHeight)
  const context = requireCanvasContext(canvas)
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
  for (let index = 0; index < imageData.data.length; index += 4) {
    const alpha = imageData.data[index + 3]
    imageData.data[index] = SEMANTIC_MASK_OVERLAY_RGB[0]
    imageData.data[index + 1] = SEMANTIC_MASK_OVERLAY_RGB[1]
    imageData.data[index + 2] = SEMANTIC_MASK_OVERLAY_RGB[2]
    imageData.data[index + 3] = Math.round((alpha / 255) * SEMANTIC_MASK_OVERLAY_ALPHA)
  }
  context.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/png')
}

export async function renderPluginMaskToDocumentMask(
  documentState: EditorDocument,
  maskDataUrl: string,
  target: Record<string, unknown>,
): Promise<string> {
  if (target.kind !== 'canvas') {
    return maskDataUrl
  }
  const bounds = pluginCanvasTargetBounds(target)
  if (!bounds) {
    return maskDataUrl
  }
  const mask = await loadImageElement(maskDataUrl)
  const canvas = document.createElement('canvas')
  canvas.width = documentState.width
  canvas.height = documentState.height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.drawImage(mask, bounds.x, bounds.y, bounds.width, bounds.height)
  return canvas.toDataURL('image/png')
}

/**
 * Compose source image and mask payloads for backend outpainting.
 *
 * @param documentState - Editor document.
 * @returns Image and mask data URLs.
 */
export function renderOutpaintInputs(
  documentState: EditorDocument,
): Promise<{
  image: string
  mask: string
  compositionMask: string | null
  selection: SelectionRect
  conditioning: GenerationConditioning | null
}> {
  return renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT)
}

export async function composeSelectionResults(
  documentState: EditorDocument,
  selection: SelectionRect,
  resultImages: string[],
  compositionMask: string | null = null,
  options: {
    replaceDocument?: boolean
    directionalPlan?: DirectionalOutpaintPlan | null
    softCompositionMask?: boolean
  } = {},
): Promise<{ images: string[]; bounds: DocumentBounds }> {
  if (options.directionalPlan) {
    return composeDirectionalResults(documentState, resultImages, options.directionalPlan)
  }
  if (options.replaceDocument) {
    const bounds = {
      x: selection.x,
      y: selection.y,
      width: selection.width,
      height: selection.height,
    }
    const canvases = await Promise.all(
      resultImages.map(async (resultImage) => {
        const result = await loadImageElement(resultImage)
        const canvas = document.createElement('canvas')
        canvas.width = bounds.width
        canvas.height = bounds.height
        const context = requireCanvasContext(canvas)
        context.clearRect(0, 0, bounds.width, bounds.height)
        context.drawImage(result, 0, 0, bounds.width, bounds.height)
        return canvas
      }),
    )
    return cropCanvasesToSharedVisibleBounds(canvases, bounds)
  }
  const baseCanvas = await renderDocumentCanvas(documentState)
  const bounds = getCompositionBounds(documentState, selection)
  const mask = compositionMask ? await loadImageElement(compositionMask) : null
  const canvases = await Promise.all(
    resultImages.map(async (resultImage) => {
      const result = await loadImageElement(resultImage)
      const canvas = document.createElement('canvas')
      canvas.width = bounds.width
      canvas.height = bounds.height
      const context = requireCanvasContext(canvas)
      context.clearRect(0, 0, bounds.width, bounds.height)
      context.drawImage(baseCanvas, -bounds.x, -bounds.y)
      if (mask) {
        const overlayCanvas = document.createElement('canvas')
        overlayCanvas.width = bounds.width
        overlayCanvas.height = bounds.height
        const overlayContext = requireCanvasContext(overlayCanvas)
        overlayContext.clearRect(0, 0, bounds.width, bounds.height)
        overlayContext.drawImage(
          result,
          selection.x - bounds.x,
          selection.y - bounds.y,
          selection.width,
          selection.height,
        )
        overlayContext.globalCompositeOperation = 'destination-in'
        if (options.softCompositionMask) {
          drawSoftCompositionMask(
            overlayContext,
            mask,
            selection.x - bounds.x,
            selection.y - bounds.y,
            selection.width,
            selection.height,
          )
        } else {
          overlayContext.drawImage(
            mask,
            selection.x - bounds.x,
            selection.y - bounds.y,
            selection.width,
            selection.height,
          )
        }
        context.drawImage(overlayCanvas, 0, 0)
        return canvas
      }
      context.drawImage(
        result,
        selection.x - bounds.x,
        selection.y - bounds.y,
        selection.width,
        selection.height,
      )
      return canvas
    }),
  )
  return cropCanvasesToSharedVisibleBounds(canvases, bounds)
}

function cropCanvasesToSharedVisibleBounds(
  canvases: HTMLCanvasElement[],
  bounds: DocumentBounds,
): { images: string[]; bounds: DocumentBounds } {
  const visibleBounds = canvases.reduce<DocumentBounds | null>((currentBounds, canvas) => {
    const context = requireCanvasContext(canvas)
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
    const localBounds = findVisibleAlphaBounds(imageData.data, canvas.width, canvas.height)
    if (!localBounds) {
      return currentBounds
    }
    const absoluteBounds = {
      x: bounds.x + localBounds.x,
      y: bounds.y + localBounds.y,
      width: localBounds.width,
      height: localBounds.height,
    }
    return currentBounds ? unionBounds(currentBounds, absoluteBounds) : sanitizeBounds(absoluteBounds)
  }, null)
  if (!visibleBounds) {
    return { images: canvases.map((canvas) => canvas.toDataURL('image/png')), bounds }
  }
  const croppedBounds = sanitizeBounds(visibleBounds)
  return {
    images: canvases.map((canvas) =>
      cropCanvasToDataUrl(canvas, {
        x: croppedBounds.x - bounds.x,
        y: croppedBounds.y - bounds.y,
        width: croppedBounds.width,
        height: croppedBounds.height,
      }),
    ),
    bounds: croppedBounds,
  }
}

function cropCanvasToDataUrl(canvas: HTMLCanvasElement, bounds: DocumentBounds): string {
  const crop = sanitizeBounds(bounds)
  if (
    crop.x === 0 &&
    crop.y === 0 &&
    crop.width === canvas.width &&
    crop.height === canvas.height
  ) {
    return canvas.toDataURL('image/png')
  }
  const croppedCanvas = document.createElement('canvas')
  croppedCanvas.width = crop.width
  croppedCanvas.height = crop.height
  const croppedContext = requireCanvasContext(croppedCanvas)
  croppedContext.clearRect(0, 0, crop.width, crop.height)
  croppedContext.drawImage(
    canvas,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    crop.width,
    crop.height,
  )
  return croppedCanvas.toDataURL('image/png')
}

function drawSoftCompositionMask(
  context: CanvasRenderingContext2D,
  mask: HTMLImageElement,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const featherRadius = compositionMaskFeatherRadius(width, height)
  context.save()
  context.filter = `blur(${featherRadius}px)`
  context.drawImage(mask, x, y, width, height)
  context.restore()
}

function compositionMaskFeatherRadius(width: number, height: number): number {
  return Math.max(
    2,
    Math.min(OUTPAINT_COMPOSITION_FEATHER_MAX, Math.round(Math.min(width, height) * 0.025)),
  )
}

async function composeDirectionalResults(
  documentState: EditorDocument,
  resultImages: string[],
  plan: DirectionalOutpaintPlan,
): Promise<{ images: string[]; bounds: DocumentBounds }> {
  const baseCanvas = await renderDocumentCanvas(documentState)
  const generatedBounds = directionalGeneratedBounds(plan)
  const bounds = unionBounds(
    { x: 0, y: 0, width: documentState.width, height: documentState.height },
    generatedBounds,
  )
  const images = await Promise.all(
    resultImages.map(async (resultImage) => {
      const result = await loadImageElement(resultImage)
      const canvas = document.createElement('canvas')
      canvas.width = bounds.width
      canvas.height = bounds.height
      const context = requireCanvasContext(canvas)
      context.clearRect(0, 0, bounds.width, bounds.height)
      context.drawImage(baseCanvas, -bounds.x, -bounds.y)
      context.drawImage(
        result,
        plan.generatedRect.x,
        plan.generatedRect.y,
        plan.generatedRect.width,
        plan.generatedRect.height,
        generatedBounds.x - bounds.x,
        generatedBounds.y - bounds.y,
        generatedBounds.width,
        generatedBounds.height,
      )
      return canvas.toDataURL('image/png')
    }),
  )
  return { images, bounds }
}

export async function eraseRasterStroke(
  documentState: EditorDocument,
  points: Point[],
  size: number,
  hardness = DEFAULT_ERASER_HARDNESS,
): Promise<string> {
  const canvas = await renderDocumentCanvas(documentState)
  const context = requireCanvasContext(canvas)
  if (points.length === 0) {
    return canvas.toDataURL('image/png')
  }
  const normalizedHardness = clampEraserHardness(hardness)
  context.save()
  context.globalCompositeOperation = 'destination-out'
  if (normalizedHardness >= MAX_ERASER_HARDNESS) {
    drawHardEraserStroke(context, points, size)
  } else {
    drawSoftEraserStroke(context, points, size, normalizedHardness)
  }
  context.restore()
  return canvas.toDataURL('image/png')
}

export async function eraseRasterSelection(
  documentState: EditorDocument,
  selection: SelectionRect,
): Promise<string> {
  const canvas = await renderDocumentCanvas(documentState)
  const context = requireCanvasContext(canvas)
  const eraseBounds = sanitizeBounds(selection)
  context.save()
  context.globalCompositeOperation = 'destination-out'
  context.fillRect(
    eraseBounds.x,
    eraseBounds.y,
    eraseBounds.width,
    eraseBounds.height,
  )
  context.restore()
  return canvas.toDataURL('image/png')
}

function drawHardEraserStroke(
  context: CanvasRenderingContext2D,
  points: Point[],
  size: number,
): void {
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = size
  context.fillStyle = CANVAS_THEME.transparentStroke
  context.strokeStyle = CANVAS_THEME.transparentStroke
  context.beginPath()
  if (points.length === 1) {
    context.arc(points[0].x, points[0].y, size / 2, 0, Math.PI * 2)
    context.fill()
  } else {
    context.moveTo(points[0].x, points[0].y)
    for (const point of points.slice(1)) {
      context.lineTo(point.x, point.y)
    }
    context.stroke()
  }
}

function drawSoftEraserStroke(
  context: CanvasRenderingContext2D,
  points: Point[],
  size: number,
  hardness: number,
): void {
  const radius = Math.max(0.5, size / 2)
  const spacing = Math.max(1, radius / 4)
  drawSoftEraserStamp(context, points[0], radius, hardness)
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1]
    const point = points[index]
    const distance = Math.hypot(point.x - previous.x, point.y - previous.y)
    const steps = Math.max(1, Math.ceil(distance / spacing))
    for (let step = 1; step <= steps; step += 1) {
      const progress = step / steps
      drawSoftEraserStamp(
        context,
        {
          x: previous.x + (point.x - previous.x) * progress,
          y: previous.y + (point.y - previous.y) * progress,
        },
        radius,
        hardness,
      )
    }
  }
}

function drawSoftEraserStamp(
  context: CanvasRenderingContext2D,
  point: Point,
  radius: number,
  hardness: number,
): void {
  const innerRadius = radius * (hardness / MAX_ERASER_HARDNESS)
  const gradient = context.createRadialGradient(
    point.x,
    point.y,
    innerRadius,
    point.x,
    point.y,
    radius,
  )
  gradient.addColorStop(0, 'rgba(0, 0, 0, 1)')
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
  context.fillStyle = gradient
  context.beginPath()
  context.arc(point.x, point.y, radius, 0, Math.PI * 2)
  context.fill()
}

function clampEraserHardness(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_ERASER_HARDNESS
  }
  return Math.max(MIN_ERASER_HARDNESS, Math.min(MAX_ERASER_HARDNESS, value))
}

export async function renderDocumentDataUrl(documentState: EditorDocument): Promise<string> {
  const canvas = await renderDocumentCanvas(documentState)
  const context = requireCanvasContext(canvas)
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
  const bounds = findVisibleAlphaBounds(imageData.data, canvas.width, canvas.height)
  return bounds ? cropCanvasToDataUrl(canvas, bounds) : canvas.toDataURL('image/png')
}

async function renderDocumentCanvas(documentState: EditorDocument): Promise<HTMLCanvasElement> {
  const raster = documentState.rasterDataUrl
    ? await loadImageElement(documentState.rasterDataUrl)
    : null
  const references = await Promise.all(
    documentState.references.map(async (reference) => ({
      ...reference,
      image: await loadImageElement(reference.dataUrl),
    })),
  )
  const canvas = document.createElement('canvas')
  canvas.width = documentState.width
  canvas.height = documentState.height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, documentState.width, documentState.height)
  if (raster) {
    const bounds = documentState.rasterBounds ?? {
      x: 0,
      y: 0,
      width: raster.naturalWidth,
      height: raster.naturalHeight,
    }
    context.drawImage(raster, bounds.x, bounds.y, bounds.width, bounds.height)
  }
  for (const reference of references) {
    context.globalAlpha = reference.opacity
    context.drawImage(reference.image, reference.x, reference.y, reference.width, reference.height)
  }
  context.globalAlpha = 1
  return canvas
}

async function renderFrameActionInput(
  documentState: EditorDocument,
): Promise<{ image: string; target: Record<string, unknown> }> {
  const fullCanvas = await renderDocumentCanvas(documentState)
  const selection = sanitizeSelection(documentState.selection)
  const canvas = document.createElement('canvas')
  canvas.width = selection.width
  canvas.height = selection.height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, selection.width, selection.height)
  context.drawImage(fullCanvas, -selection.x, -selection.y)
  requireVisiblePixels(canvas)
  return {
    image: canvas.toDataURL('image/png'),
    target: {
      kind: 'frame',
      bounds: selection,
    },
  }
}

async function renderCanvasActionInput(
  documentState: EditorDocument,
  target: Extract<CanvasSelectionTarget, { kind: 'canvas' }>,
): Promise<{ image: string; target: Record<string, unknown> }> {
  const fullCanvas = await renderDocumentCanvas(documentState)
  const fullContext = requireCanvasContext(fullCanvas)
  const fullImage = fullContext.getImageData(0, 0, fullCanvas.width, fullCanvas.height)
  const visibleBounds = findVisibleAlphaBounds(fullImage.data, fullCanvas.width, fullCanvas.height)
  if (!visibleBounds) {
    throw new AppError('PLUGIN_ACTION_IMAGE_REQUIRED', 'Select visible image content first.')
  }
  const scale = Math.min(
    1,
    MAX_PLUGIN_CANVAS_TARGET_SIZE / Math.max(visibleBounds.width, visibleBounds.height),
  )
  const renderWidth = Math.max(1, Math.round(visibleBounds.width * scale))
  const renderHeight = Math.max(1, Math.round(visibleBounds.height * scale))
  const canvas = document.createElement('canvas')
  canvas.width = renderWidth
  canvas.height = renderHeight
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, renderWidth, renderHeight)
  context.drawImage(
    fullCanvas,
    visibleBounds.x,
    visibleBounds.y,
    visibleBounds.width,
    visibleBounds.height,
    0,
    0,
    renderWidth,
    renderHeight,
  )
  requireVisiblePixels(canvas)
  return {
    image: canvas.toDataURL('image/png'),
    target: {
      kind: 'canvas',
      bounds: visibleBounds,
      scale,
      point: target.point
        ? {
            x: (target.point.x - visibleBounds.x) * scale,
            y: (target.point.y - visibleBounds.y) * scale,
          }
        : undefined,
      visible_mask: renderVisibleMaskDataUrl(canvas),
    },
  }
}

async function renderRasterActionInput(
  documentState: EditorDocument,
): Promise<{ image: string; target: Record<string, unknown> }> {
  if (!documentState.rasterDataUrl) {
    throw new AppError('PLUGIN_ACTION_IMAGE_REQUIRED', 'Select visible image content first.')
  }
  const image = await loadImageElement(documentState.rasterDataUrl)
  const bounds = documentState.rasterBounds ?? {
    x: 0,
    y: 0,
    width: image.naturalWidth,
    height: image.naturalHeight,
  }
  return renderLayerImage(image, bounds, 'raster')
}

async function renderReferenceActionInput(
  documentState: EditorDocument,
  referenceId: string,
): Promise<{ image: string; target: Record<string, unknown> }> {
  const reference = documentState.references.find((item) => item.id === referenceId)
  if (!reference) {
    throw new AppError('PLUGIN_ACTION_IMAGE_REQUIRED', 'Select visible image content first.')
  }
  const image = await loadImageElement(reference.dataUrl)
  return renderLayerImage(
    image,
    {
      x: reference.x,
      y: reference.y,
      width: reference.width,
      height: reference.height,
    },
    'reference',
  )
}

function renderLayerImage(
  image: HTMLImageElement,
  bounds: DocumentBounds,
  kind: string,
): { image: string; target: Record<string, unknown> } {
  const width = Math.max(1, Math.round(bounds.width))
  const height = Math.max(1, Math.round(bounds.height))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = requireCanvasContext(canvas)
  context.drawImage(image, 0, 0, width, height)
  requireVisiblePixels(canvas)
  return {
    image: canvas.toDataURL('image/png'),
    target: {
      kind,
      bounds: {
        x: Math.round(bounds.x),
        y: Math.round(bounds.y),
        width,
        height,
      },
    },
  }
}

async function renderMaskAlphaCanvas(
  maskDataUrl: string,
  width: number,
  height: number,
): Promise<HTMLCanvasElement> {
  const mask = await loadImageElement(maskDataUrl)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, width, height)
  context.drawImage(mask, 0, 0, width, height)
  const imageData = context.getImageData(0, 0, width, height)
  for (let index = 0; index < imageData.data.length; index += 4) {
    const sourceAlpha = imageData.data[index + 3] / 255
    const maskAlpha = Math.max(
      imageData.data[index],
      imageData.data[index + 1],
      imageData.data[index + 2],
    )
    imageData.data[index] = 0
    imageData.data[index + 1] = 0
    imageData.data[index + 2] = 0
    imageData.data[index + 3] = Math.round(maskAlpha * sourceAlpha)
  }
  context.putImageData(imageData, 0, 0)
  return canvas
}

function pluginCanvasTargetBounds(target: Record<string, unknown>): DocumentBounds | null {
  const bounds = target.bounds
  if (!bounds || typeof bounds !== 'object') {
    return null
  }
  const rawBounds = bounds as Record<string, unknown>
  const x = finiteNumber(rawBounds.x)
  const y = finiteNumber(rawBounds.y)
  const width = finiteNumber(rawBounds.width)
  const height = finiteNumber(rawBounds.height)
  if (x === null || y === null || width === null || height === null) {
    return null
  }
  return { x, y, width, height }
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function renderVisibleMaskDataUrl(source: HTMLCanvasElement): string {
  const sourceContext = requireCanvasContext(source)
  const sourceImage = sourceContext.getImageData(0, 0, source.width, source.height)
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = source.width
  maskCanvas.height = source.height
  const maskContext = requireCanvasContext(maskCanvas)
  const maskImage = maskContext.createImageData(source.width, source.height)
  for (let index = 0; index < sourceImage.data.length; index += 4) {
    if (sourceImage.data[index + 3] <= TRANSPARENT_ALPHA) {
      continue
    }
    maskImage.data[index] = 255
    maskImage.data[index + 1] = 255
    maskImage.data[index + 2] = 255
    maskImage.data[index + 3] = 255
  }
  maskContext.putImageData(maskImage, 0, 0)
  return maskCanvas.toDataURL('image/png')
}

function requireVisiblePixels(canvas: HTMLCanvasElement): void {
  const context = requireCanvasContext(canvas)
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
  for (let index = 3; index < imageData.data.length; index += 4) {
    if (imageData.data[index] > TRANSPARENT_ALPHA) {
      return
    }
  }
  throw new AppError('PLUGIN_ACTION_IMAGE_REQUIRED', 'Select visible image content first.')
}

/**
 * Convert a data URL into a Blob.
 *
 * @param dataUrl - Source data URL.
 * @returns Binary blob.
 */
export async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const response = await fetch(dataUrl)
  return response.blob()
}

/**
 * Return the project raster asset path used by .expd archives.
 *
 * @returns Raster asset path.
 */
export function getRasterAssetPath(): string {
  return RASTER_ASSET_PATH
}

function sanitizeSelection(selection: SelectionRect): SelectionRect {
  return {
    x: Math.round(selection.x),
    y: Math.round(selection.y),
    width: Math.max(1, Math.min(Math.round(selection.width), MAX_IMPORT_CANVAS_SIZE)),
    height: Math.max(1, Math.min(Math.round(selection.height), MAX_IMPORT_CANVAS_SIZE)),
  }
}

function getCompositionBounds(documentState: EditorDocument, selection: SelectionRect): DocumentBounds {
  const minX = Math.min(0, selection.x)
  const minY = Math.min(0, selection.y)
  const maxX = Math.max(documentState.width, selection.x + selection.width)
  const maxY = Math.max(documentState.height, selection.y + selection.height)
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

function calculateImportPadding(width: number, height: number): number {
  const largestSide = Math.max(width, height)
  if (largestSide >= MAX_IMPORT_CANVAS_SIZE) {
    return 0
  }
  const availablePadding = Math.floor((MAX_IMPORT_CANVAS_SIZE - largestSide) / 2)
  return Math.max(64, Math.min(DEFAULT_SELECTION_SIZE, availablePadding))
}

function normalizeImportImage(
  image: HTMLImageElement,
  dataUrl: string,
): { source: CanvasImageSource; dataUrl: string; width: number; height: number } {
  const width = image.naturalWidth
  const height = image.naturalHeight
  const scale = importImageScale(width, height)
  if (scale === 1) {
    return { source: image, dataUrl, width, height }
  }
  const targetWidth = Math.max(1, Math.round(width * scale))
  const targetHeight = Math.max(1, Math.round(height * scale))
  const canvas = document.createElement('canvas')
  canvas.width = targetWidth
  canvas.height = targetHeight
  const context = requireCanvasContext(canvas)
  context.drawImage(image, 0, 0, targetWidth, targetHeight)
  return {
    source: canvas,
    dataUrl: canvas.toDataURL('image/png'),
    width: targetWidth,
    height: targetHeight,
  }
}

function importImageScale(width: number, height: number): number {
  const shortestSide = Math.min(width, height)
  const longestSide = Math.max(width, height)
  let scale = 1
  if (shortestSide < MIN_IMPORT_IMAGE_EDGE) {
    scale = MIN_IMPORT_IMAGE_EDGE / shortestSide
  }
  if (longestSide * scale > MAX_IMPORT_IMAGE_EDGE) {
    scale = MAX_IMPORT_IMAGE_EDGE / longestSide
  }
  return scale
}

function trimTransparentImageBounds(
  image: CanvasImageSource,
  fallbackDataUrl: string,
  sourceWidth: number,
  sourceHeight: number,
): { dataUrl: string; width: number; height: number } {
  const sourceCanvas = document.createElement('canvas')
  sourceCanvas.width = sourceWidth
  sourceCanvas.height = sourceHeight
  const sourceContext = requireCanvasContext(sourceCanvas)
  sourceContext.drawImage(image, 0, 0)
  const imageData = sourceContext.getImageData(0, 0, sourceWidth, sourceHeight)
  const bounds = findVisibleAlphaBounds(imageData.data, sourceWidth, sourceHeight)
  if (
    !bounds ||
    (bounds.x === 0 &&
      bounds.y === 0 &&
      bounds.width === sourceWidth &&
      bounds.height === sourceHeight)
  ) {
    return { dataUrl: fallbackDataUrl, width: sourceWidth, height: sourceHeight }
  }

  const targetCanvas = document.createElement('canvas')
  targetCanvas.width = bounds.width
  targetCanvas.height = bounds.height
  const targetContext = requireCanvasContext(targetCanvas)
  targetContext.drawImage(
    sourceCanvas,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    bounds.width,
    bounds.height,
  )
  return {
    dataUrl: targetCanvas.toDataURL('image/png'),
    width: bounds.width,
    height: bounds.height,
  }
}

function findVisibleAlphaBounds(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
): DocumentBounds | null {
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const alpha = pixels[(y * width + x) * 4 + 3]
      if (alpha <= TRANSPARENT_ALPHA) {
        continue
      }
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
  }
  if (maxX < minX || maxY < minY) {
    return null
  }
  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
  }
}

export function expandOutpaintSelectionWithContext(
  selection: SelectionRect,
  visibleBounds: DocumentBounds | null,
): SelectionRect {
  return planOutpaintRender(
    selection,
    visibleBounds,
    OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  ).selection
}

export function planDirectionalOutpaintRender(
  visibleBounds: DocumentBounds,
  direction: OutpaintDirection = OUTPAINT_DIRECTION_RIGHT,
  options: {
    generatedSize?: number
    contextSize?: number
    crossSize?: number
  } = {},
): DirectionalOutpaintPlan {
  const normalizedDirection = normalizeOutpaintDirection(direction)
  const generatedSize = boundedDirectionalSize(
    options.generatedSize,
    DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
  )
  const contextSize = boundedContextSize(
    options.contextSize,
    DIRECTIONAL_OUTPAINT_DEFAULT_CONTEXT_SIZE,
  )
  const crossSize = boundedDirectionalSize(
    options.crossSize,
    DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
  )

  if (
    normalizedDirection === OUTPAINT_DIRECTION_RIGHT ||
    normalizedDirection === OUTPAINT_DIRECTION_LEFT
  ) {
    const width = contextSize + generatedSize
    const height = crossSize
    const scale = crossSize / Math.max(1, visibleBounds.height)
    const contextRect = normalizedDirection === OUTPAINT_DIRECTION_RIGHT
      ? { x: 0, y: 0, width: contextSize, height }
      : { x: generatedSize, y: 0, width: contextSize, height }
    const generatedRect = normalizedDirection === OUTPAINT_DIRECTION_RIGHT
      ? { x: contextSize, y: 0, width: generatedSize, height }
      : { x: 0, y: 0, width: generatedSize, height }
    const drawRect = {
      x: normalizedDirection === OUTPAINT_DIRECTION_RIGHT
        ? contextSize - (visibleBounds.x + visibleBounds.width) * scale
        : generatedSize - visibleBounds.x * scale,
      y: -visibleBounds.y * scale,
      width: 0,
      height: 0,
    }
    return {
      direction: normalizedDirection,
      width,
      height,
      generatedSize,
      contextSize,
      crossSize,
      scale,
      visibleBounds,
      contextRect,
      generatedRect,
      drawRect,
    }
  }

  const width = crossSize
  const height = contextSize + generatedSize
  const scale = crossSize / Math.max(1, visibleBounds.width)
  const contextRect = normalizedDirection === OUTPAINT_DIRECTION_DOWN
    ? { x: 0, y: 0, width, height: contextSize }
    : { x: 0, y: generatedSize, width, height: contextSize }
  const generatedRect = normalizedDirection === OUTPAINT_DIRECTION_DOWN
    ? { x: 0, y: contextSize, width, height: generatedSize }
    : { x: 0, y: 0, width, height: generatedSize }
  const drawRect = {
    x: -visibleBounds.x * scale,
    y: normalizedDirection === OUTPAINT_DIRECTION_DOWN
      ? contextSize - (visibleBounds.y + visibleBounds.height) * scale
      : generatedSize - visibleBounds.y * scale,
    width: 0,
    height: 0,
  }
  return {
    direction: normalizedDirection,
    width,
    height,
    generatedSize,
    contextSize,
    crossSize,
    scale,
    visibleBounds,
    contextRect,
    generatedRect,
    drawRect,
  }
}

export function directionalGeneratedBounds(plan: DirectionalOutpaintPlan): DocumentBounds {
  const generatedWidth = Math.round(plan.generatedRect.width / plan.scale)
  const generatedHeight = Math.round(plan.generatedRect.height / plan.scale)
  if (plan.direction === OUTPAINT_DIRECTION_RIGHT) {
    return {
      x: plan.visibleBounds.x + plan.visibleBounds.width,
      y: plan.visibleBounds.y,
      width: generatedWidth,
      height: generatedHeight,
    }
  }
  if (plan.direction === OUTPAINT_DIRECTION_LEFT) {
    return {
      x: plan.visibleBounds.x - generatedWidth,
      y: plan.visibleBounds.y,
      width: generatedWidth,
      height: generatedHeight,
    }
  }
  if (plan.direction === OUTPAINT_DIRECTION_DOWN) {
    return {
      x: plan.visibleBounds.x,
      y: plan.visibleBounds.y + plan.visibleBounds.height,
      width: generatedWidth,
      height: generatedHeight,
    }
  }
  return {
    x: plan.visibleBounds.x,
    y: plan.visibleBounds.y - generatedHeight,
    width: generatedWidth,
    height: generatedHeight,
  }
}

function sanitizeBounds(bounds: DocumentBounds): SelectionRect {
  return {
    x: Math.round(bounds.x),
    y: Math.round(bounds.y),
    width: Math.max(1, Math.round(bounds.width)),
    height: Math.max(1, Math.round(bounds.height)),
  }
}

export function planOutpaintRender(
  selection: SelectionRect,
  visibleBounds: DocumentBounds | null,
  strategy: OutpaintStrategy,
  options: { maxWidth?: number; maxHeight?: number } = {},
): { selection: SelectionRect; renderWidth: number; renderHeight: number } {
  const selectedFrame = sanitizeSelection(selection)
  if (!visibleBounds || strategy === OUTPAINT_STRATEGY_SELECTED_FRAME) {
    return renderPlanForSelection(selectedFrame)
  }
  if (strategy === OUTPAINT_STRATEGY_FULL_CONTEXT_CROP) {
    return renderPlanForSelection(
      cropFullContextSelection(
        selectedFrame,
        visibleBounds,
        boundedOutpaintDimension(options.maxWidth, OUTPAINT_DEFAULT_MAX_WIDTH),
        boundedOutpaintDimension(options.maxHeight, OUTPAINT_DEFAULT_MAX_HEIGHT),
      ),
    )
  }
  if (strategy === OUTPAINT_STRATEGY_WHOLE_RESIZED) {
    const wholeSelection = sanitizeSelection(unionBounds(selectedFrame, visibleBounds))
    return {
      selection: wholeSelection,
      ...fitSizeWithinMaxEdge(
        wholeSelection.width,
        wholeSelection.height,
        WHOLE_RESIZED_MAX_INPUT_EDGE,
      ),
    }
  }
  const contextRect = outpaintContextRect(selectedFrame, visibleBounds)
  return renderPlanForSelection(
    contextRect ? sanitizeSelection(unionBounds(selectedFrame, contextRect)) : selectedFrame,
  )
}

function renderDirectionalGenerationInputs(
  fullCanvas: HTMLCanvasElement,
  options: {
    direction?: OutpaintDirection
    generatedSize?: number
    contextSize?: number
    crossSize?: number
  },
): RenderedGenerationInputs {
  const fullContext = requireCanvasContext(fullCanvas)
  const fullImageData = fullContext.getImageData(0, 0, fullCanvas.width, fullCanvas.height)
  const visibleBounds = findVisibleAlphaBounds(
    fullImageData.data,
    fullCanvas.width,
    fullCanvas.height,
  )
  if (!visibleBounds) {
    throw new AppError(
      'OUTPAINT_CONTEXT_REQUIRED',
      'Import visible image content before generating directional outpaint.',
    )
  }
  const plan = planDirectionalOutpaintRender(
    visibleBounds,
    options.direction ?? OUTPAINT_DIRECTION_RIGHT,
    {
      generatedSize: options.generatedSize,
      contextSize: options.contextSize,
      crossSize: options.crossSize,
    },
  )
  const imageCanvas = document.createElement('canvas')
  imageCanvas.width = plan.width
  imageCanvas.height = plan.height
  const imageContext = requireCanvasContext(imageCanvas)
  imageContext.clearRect(0, 0, plan.width, plan.height)
  drawDirectionalSourceContext(imageContext, fullCanvas, plan)

  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = plan.width
  maskCanvas.height = plan.height
  const maskContext = requireCanvasContext(maskCanvas)
  maskContext.fillStyle = BLACK_PIXEL
  maskContext.fillRect(0, 0, plan.width, plan.height)
  maskContext.fillStyle = WHITE_PIXEL
  maskContext.fillRect(
    plan.generatedRect.x,
    plan.generatedRect.y,
    plan.generatedRect.width,
    plan.generatedRect.height,
  )

  const selection = { x: 0, y: 0, width: plan.width, height: plan.height }
  return {
    image: imageCanvas.toDataURL('image/png'),
    mask: maskCanvas.toDataURL('image/png'),
    compositionMask: maskCanvas.toDataURL('image/png'),
    selection,
    previewSelection: sanitizeBounds(directionalGeneratedBounds(plan)),
    conditioning: null,
    replaceDocument: false,
    directionalPlan: plan,
  }
}

async function renderHfSpaceFillGenerationInputs(
  documentState: EditorDocument,
  selectedFrame: SelectionRect,
  options: {
    direction?: OutpaintDirection
    generatedSize?: number
    crossSize?: number
    overlapPercentage?: number
    fixedExpansion?: boolean
    resizeOption?: string
    customResizePercentage?: number
    overlapLeft?: boolean
    overlapRight?: boolean
    overlapTop?: boolean
    overlapBottom?: boolean
    expansionPercent?: number
    widthExpansionPercent?: number
    heightExpansionPercent?: number
    outputScalePercent?: number
  },
): Promise<RenderedGenerationInputs> {
  const sourceCanvas = await renderHfSpaceVisibleSource(documentState)
  const visibleBounds = visibleSourceBounds(documentState, sourceCanvas)
  const expansionPlan = options.fixedExpansion
    ? planHfSpaceFillExpansion(visibleBounds, {
        direction: options.direction,
        generatedSize: options.generatedSize,
        crossSize: options.crossSize,
        resizeOption: options.resizeOption,
        customResizePercentage: options.customResizePercentage,
        overlapPercentage: options.overlapPercentage,
        overlapLeft: options.overlapLeft,
        overlapRight: options.overlapRight,
        overlapTop: options.overlapTop,
        overlapBottom: options.overlapBottom,
        expansionPercent: options.expansionPercent,
        widthExpansionPercent: options.widthExpansionPercent,
        heightExpansionPercent: options.heightExpansionPercent,
        outputScalePercent: options.outputScalePercent,
      })
    : null
  const targetSelection = expansionPlan?.selection ?? selectedFrame
  const targetWidth = hfSpaceFillSafeDimension(Math.round(targetSelection.width))
  const targetHeight = hfSpaceFillSafeDimension(Math.round(targetSelection.height))
  const renderSize = expansionPlan?.renderSize ?? { width: targetWidth, height: targetHeight }

  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = sourceCanvas.width
  maskCanvas.height = sourceCanvas.height
  const maskContext = requireCanvasContext(maskCanvas)
  maskContext.fillStyle = BLACK_PIXEL
  maskContext.fillRect(0, 0, sourceCanvas.width, sourceCanvas.height)

  const selection = {
    x: Math.round(targetSelection.x),
    y: Math.round(targetSelection.y),
    width: targetWidth,
    height: targetHeight,
  }
  return {
    image: sourceCanvas.toDataURL('image/png'),
    mask: maskCanvas.toDataURL('image/png'),
    compositionMask: null,
    selection,
    previewSelection: selection,
    conditioning: null,
    replaceDocument: Boolean(options.fixedExpansion),
    directionalPlan: null,
    renderSize,
  }
}

export function planHfSpaceFillExpansion(
  visibleBounds: DocumentBounds,
  options: {
    direction?: OutpaintDirection
    generatedSize?: number
    crossSize?: number
    resizeOption?: string
    customResizePercentage?: number
    overlapPercentage?: number
    overlapLeft?: boolean
    overlapRight?: boolean
    overlapTop?: boolean
    overlapBottom?: boolean
    expansionPercent?: number
    widthExpansionPercent?: number
    heightExpansionPercent?: number
    outputScalePercent?: number
  } = {},
): HfSpaceFillExpansionPlan {
  const direction = options.direction ?? OUTPAINT_DIRECTION_RIGHT
  const generatedSize = boundedDirectionalSize(
    options.generatedSize,
    DIRECTIONAL_OUTPAINT_DEFAULT_GENERATED_SIZE,
  )
  const crossSize = boundedDirectionalSize(
    options.crossSize,
    DIRECTIONAL_OUTPAINT_DEFAULT_CROSS_SIZE,
  )
  const percentageSizingActive =
    typeof options.expansionPercent === 'number' ||
    typeof options.widthExpansionPercent === 'number' ||
    typeof options.heightExpansionPercent === 'number'
  const expansionPercent = boundedExpansionPercentage(options.expansionPercent, 50)
  const widthExpansionPercent = boundedExpansionPercentage(
    options.widthExpansionPercent,
    expansionPercent,
  )
  const heightExpansionPercent = boundedExpansionPercentage(
    options.heightExpansionPercent,
    expansionPercent,
  )
  const requestedWidth = percentageSizingActive
    ? fixedExpansionRequestedWidth(visibleBounds, direction, expansionPercent, widthExpansionPercent)
    : direction === OUTPAINT_DIRECTION_AROUND
      ? generatedSize
      : direction === OUTPAINT_DIRECTION_LEFT || direction === OUTPAINT_DIRECTION_RIGHT
        ? visibleBounds.width + generatedSize
        : Math.max(visibleBounds.width, crossSize)
  const requestedHeight = percentageSizingActive
    ? fixedExpansionRequestedHeight(visibleBounds, direction, expansionPercent, heightExpansionPercent)
    : direction === OUTPAINT_DIRECTION_AROUND
      ? crossSize
      : direction === OUTPAINT_DIRECTION_UP || direction === OUTPAINT_DIRECTION_DOWN
        ? visibleBounds.height + generatedSize
        : Math.max(visibleBounds.height, crossSize)
  const targetWidth = hfSpaceFillSafeDimension(Math.round(requestedWidth))
  const targetHeight = hfSpaceFillSafeDimension(Math.round(requestedHeight))
  const selection = fixedExpansionSelection(visibleBounds, direction, targetWidth, targetHeight)
  const renderSize = fixedExpansionRenderSize(
    targetWidth,
    targetHeight,
    options.outputScalePercent,
  )
  const resizePercentage = hfSpaceResizePercentage(
    options.resizeOption,
    options.customResizePercentage,
  )
  const fitScale = Math.min(
    targetWidth / Math.max(1, visibleBounds.width),
    targetHeight / Math.max(1, visibleBounds.height),
  )
  const sourceWidth = Math.max(64, Math.round(visibleBounds.width * fitScale * (resizePercentage / 100)))
  const sourceHeight = Math.max(64, Math.round(visibleBounds.height * fitScale * (resizePercentage / 100)))
  const sourceOffset = hfSpaceSourceOffset(direction, targetWidth, targetHeight, sourceWidth, sourceHeight)
  const sourceRect = {
    x: selection.x + sourceOffset.x,
    y: selection.y + sourceOffset.y,
    width: sourceWidth,
    height: sourceHeight,
  }
  const overlapPercentage = boundedOverlapPercentage(options.overlapPercentage)
  const overlap = {
    x: Math.max(1, Math.round(sourceWidth * (overlapPercentage / 100))),
    y: Math.max(1, Math.round(sourceHeight * (overlapPercentage / 100))),
  }
  const preservedRect = preservedSourceRect(sourceRect, direction, overlap, {
    left: options.overlapLeft ?? true,
    right: options.overlapRight ?? true,
    top: options.overlapTop ?? true,
    bottom: options.overlapBottom ?? true,
  })
  return {
    selection,
    sourceRect,
    preservedRect,
    resizePercentage,
    overlapPercentage,
    overlap,
    renderSize,
  }
}

function visibleSourceBounds(
  documentState: EditorDocument,
  sourceCanvas: HTMLCanvasElement,
): DocumentBounds {
  return documentState.rasterBounds ?? {
    x: 0,
    y: 0,
    width: sourceCanvas.width,
    height: sourceCanvas.height,
  }
}

function fixedExpansionSelection(
  visibleBounds: DocumentBounds,
  direction: OutpaintDirection,
  width: number,
  height: number,
): SelectionRect {
  if (direction === OUTPAINT_DIRECTION_LEFT) {
    return {
      x: Math.round(visibleBounds.x + visibleBounds.width - width),
      y: Math.round(visibleBounds.y + visibleBounds.height / 2 - height / 2),
      width,
      height,
    }
  }
  if (direction === OUTPAINT_DIRECTION_UP) {
    return {
      x: Math.round(visibleBounds.x + visibleBounds.width / 2 - width / 2),
      y: Math.round(visibleBounds.y + visibleBounds.height - height),
      width,
      height,
    }
  }
  if (direction === OUTPAINT_DIRECTION_DOWN) {
    return {
      x: Math.round(visibleBounds.x + visibleBounds.width / 2 - width / 2),
      y: Math.round(visibleBounds.y),
      width,
      height,
    }
  }
  if (direction === OUTPAINT_DIRECTION_AROUND) {
    return {
      x: Math.round(visibleBounds.x + visibleBounds.width / 2 - width / 2),
      y: Math.round(visibleBounds.y + visibleBounds.height / 2 - height / 2),
      width,
      height,
    }
  }
  return {
    x: Math.round(visibleBounds.x),
    y: Math.round(visibleBounds.y + visibleBounds.height / 2 - height / 2),
    width,
    height,
  }
}

function hfSpaceSourceOffset(
  direction: OutpaintDirection,
  targetWidth: number,
  targetHeight: number,
  sourceWidth: number,
  sourceHeight: number,
): { x: number; y: number } {
  const centeredX = Math.round((targetWidth - sourceWidth) / 2)
  const centeredY = Math.round((targetHeight - sourceHeight) / 2)
  if (direction === OUTPAINT_DIRECTION_LEFT) {
    return { x: targetWidth - sourceWidth, y: centeredY }
  }
  if (direction === OUTPAINT_DIRECTION_UP) {
    return { x: centeredX, y: targetHeight - sourceHeight }
  }
  if (direction === OUTPAINT_DIRECTION_DOWN) {
    return { x: centeredX, y: 0 }
  }
  if (direction === OUTPAINT_DIRECTION_AROUND) {
    return { x: centeredX, y: centeredY }
  }
  return { x: 0, y: centeredY }
}

function preservedSourceRect(
  sourceRect: DocumentBounds,
  direction: OutpaintDirection,
  overlap: { x: number; y: number },
  sides: { left: boolean; right: boolean; top: boolean; bottom: boolean },
): DocumentBounds {
  const patch = 2
  let left = sourceRect.x + (sides.left ? overlap.x : patch)
  let right = sourceRect.x + sourceRect.width - (sides.right ? overlap.x : patch)
  let top = sourceRect.y + (sides.top ? overlap.y : patch)
  let bottom = sourceRect.y + sourceRect.height - (sides.bottom ? overlap.y : patch)
  if (direction === OUTPAINT_DIRECTION_RIGHT) {
    left = sourceRect.x + (sides.left ? overlap.x : 0)
  }
  if (direction === OUTPAINT_DIRECTION_LEFT) {
    right = sourceRect.x + sourceRect.width - (sides.right ? overlap.x : 0)
  }
  if (direction === OUTPAINT_DIRECTION_DOWN) {
    top = sourceRect.y + (sides.top ? overlap.y : 0)
  }
  if (direction === OUTPAINT_DIRECTION_UP) {
    bottom = sourceRect.y + sourceRect.height - (sides.bottom ? overlap.y : 0)
  }
  return {
    x: Math.round(left),
    y: Math.round(top),
    width: Math.max(1, Math.round(right - left)),
    height: Math.max(1, Math.round(bottom - top)),
  }
}

async function renderHfSpaceVisibleSource(
  documentState: EditorDocument,
): Promise<HTMLCanvasElement> {
  if (documentState.rasterDataUrl) {
    const raster = await loadImageElement(documentState.rasterDataUrl)
    const bounds = documentState.rasterBounds ?? {
      x: 0,
      y: 0,
      width: raster.naturalWidth,
      height: raster.naturalHeight,
    }
    return cropVisibleCanvas(renderRasterBoundsCanvas(raster, bounds))
  }

  const fullCanvas = await renderDocumentCanvas(documentState)
  return cropVisibleCanvas(fullCanvas)
}

function renderRasterBoundsCanvas(
  raster: HTMLImageElement,
  bounds: DocumentBounds,
): HTMLCanvasElement {
  const width = Math.max(1, Math.round(bounds.width))
  const height = Math.max(1, Math.round(bounds.height))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, width, height)
  context.drawImage(raster, 0, 0, width, height)
  return canvas
}

function cropVisibleCanvas(
  canvas: HTMLCanvasElement,
): HTMLCanvasElement {
  const context = requireCanvasContext(canvas)
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
  const visibleBounds = findVisibleAlphaBounds(imageData.data, canvas.width, canvas.height)
  if (!visibleBounds) {
    throw new AppError(
      'OUTPAINT_CONTEXT_REQUIRED',
      'Import visible image content before generating HF Space outpaint.',
    )
  }

  const sourceCanvas = document.createElement('canvas')
  sourceCanvas.width = visibleBounds.width
  sourceCanvas.height = visibleBounds.height
  const sourceContext = requireCanvasContext(sourceCanvas)
  sourceContext.clearRect(0, 0, visibleBounds.width, visibleBounds.height)
  sourceContext.drawImage(
    canvas,
    visibleBounds.x,
    visibleBounds.y,
    visibleBounds.width,
    visibleBounds.height,
    0,
    0,
    visibleBounds.width,
    visibleBounds.height,
  )

  return sourceCanvas
}

function hfSpaceFillSafeDimension(value: number): number {
  const bounded = Math.max(64, value)
  return Math.max(
    HF_SPACE_FILL_SIZE_MULTIPLE,
    bounded - (bounded % HF_SPACE_FILL_SIZE_MULTIPLE),
  )
}

function fixedExpansionRequestedWidth(
  visibleBounds: DocumentBounds,
  direction: OutpaintDirection,
  expansionPercent: number,
  widthExpansionPercent: number,
): number {
  if (direction === OUTPAINT_DIRECTION_LEFT || direction === OUTPAINT_DIRECTION_RIGHT) {
    return visibleBounds.width * (1 + expansionPercent / 100)
  }
  if (direction === OUTPAINT_DIRECTION_AROUND) {
    return visibleBounds.width * (1 + widthExpansionPercent / 100)
  }
  return visibleBounds.width
}

function fixedExpansionRequestedHeight(
  visibleBounds: DocumentBounds,
  direction: OutpaintDirection,
  expansionPercent: number,
  heightExpansionPercent: number,
): number {
  if (direction === OUTPAINT_DIRECTION_UP || direction === OUTPAINT_DIRECTION_DOWN) {
    return visibleBounds.height * (1 + expansionPercent / 100)
  }
  if (direction === OUTPAINT_DIRECTION_AROUND) {
    return visibleBounds.height * (1 + heightExpansionPercent / 100)
  }
  return visibleBounds.height
}

function fixedExpansionRenderSize(
  targetWidth: number,
  targetHeight: number,
  outputScalePercent: number | undefined,
): { width: number; height: number } {
  const scalePercent = boundedOutputScalePercentage(outputScalePercent)
  return {
    width: hfSpaceFillSafeDimension(Math.round(targetWidth * (scalePercent / 100))),
    height: hfSpaceFillSafeDimension(Math.round(targetHeight * (scalePercent / 100))),
  }
}

function boundedExpansionPercentage(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.max(5, Math.min(300, Math.round(value)))
}

function boundedOutputScalePercentage(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 100
  }
  return Math.max(25, Math.min(100, Math.round(value)))
}

function hfSpaceResizePercentage(option: string | undefined, customValue: number | undefined): number {
  if (option === '50%') {
    return 50
  }
  if (option === '33%') {
    return 33
  }
  if (option === '25%') {
    return 25
  }
  if (option === 'Custom') {
    if (typeof customValue === 'number' && Number.isFinite(customValue)) {
      return Math.max(1, Math.min(100, Math.round(customValue)))
    }
    return 50
  }
  return 100
}

function boundedOverlapPercentage(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 10
  }
  return Math.max(1, Math.min(50, Math.round(value)))
}

function drawDirectionalSourceContext(
  context: CanvasRenderingContext2D,
  fullCanvas: HTMLCanvasElement,
  plan: DirectionalOutpaintPlan,
): void {
  const drawWidth = fullCanvas.width * plan.scale
  const drawHeight = fullCanvas.height * plan.scale
  context.save()
  context.beginPath()
  context.rect(
    plan.contextRect.x,
    plan.contextRect.y,
    plan.contextRect.width,
    plan.contextRect.height,
  )
  context.clip()
  context.drawImage(
    fullCanvas,
    plan.drawRect.x,
    plan.drawRect.y,
    drawWidth,
    drawHeight,
  )
  context.restore()
}

function outpaintRenderPlanFromCanvas(
  selection: SelectionRect,
  fullCanvas: HTMLCanvasElement,
  strategy: OutpaintStrategy,
  options: { maxWidth?: number; maxHeight?: number } = {},
): { selection: SelectionRect; renderWidth: number; renderHeight: number } {
  const context = requireCanvasContext(fullCanvas)
  const imageData = context.getImageData(0, 0, fullCanvas.width, fullCanvas.height)
  return planOutpaintRender(
    selection,
    findVisibleAlphaBounds(imageData.data, fullCanvas.width, fullCanvas.height),
    strategy,
    options,
  )
}

function renderPlanForSelection(
  selection: SelectionRect,
): { selection: SelectionRect; renderWidth: number; renderHeight: number } {
  return {
    selection,
    renderWidth: selection.width,
    renderHeight: selection.height,
  }
}

function fitSizeWithinMaxEdge(
  width: number,
  height: number,
  maxEdge: number,
): { renderWidth: number; renderHeight: number } {
  const scale = Math.min(1, maxEdge / Math.max(width, height))
  return {
    renderWidth: Math.max(1, Math.round(width * scale)),
    renderHeight: Math.max(1, Math.round(height * scale)),
  }
}

function cropFullContextSelection(
  selectedFrame: SelectionRect,
  visibleBounds: DocumentBounds,
  maxWidth: number,
  maxHeight: number,
): SelectionRect {
  const fullSelection = unionBounds(selectedFrame, visibleBounds)
  const width = Math.min(fullSelection.width, maxWidth)
  const height = Math.min(fullSelection.height, maxHeight)
  return sanitizeSelection({
    x: cropAxisStart(
      fullSelection.x,
      fullSelection.width,
      selectedFrame.x,
      selectedFrame.width,
      visibleBounds.x,
      visibleBounds.width,
      width,
    ),
    y: cropAxisStart(
      fullSelection.y,
      fullSelection.height,
      selectedFrame.y,
      selectedFrame.height,
      visibleBounds.y,
      visibleBounds.height,
      height,
    ),
    width,
    height,
  })
}

function cropAxisStart(
  fullStart: number,
  fullSize: number,
  selectedStart: number,
  selectedSize: number,
  visibleStart: number,
  visibleSize: number,
  cropSize: number,
): number {
  if (cropSize >= fullSize) {
    return fullStart
  }
  const fullEnd = fullStart + fullSize
  const selectedEnd = selectedStart + selectedSize
  const visibleEnd = visibleStart + visibleSize
  if (selectedStart < visibleStart) {
    return clamp(selectedStart, fullStart, fullEnd - cropSize)
  }
  if (selectedEnd > visibleEnd) {
    return clamp(selectedEnd - cropSize, fullStart, fullEnd - cropSize)
  }
  return clamp(
    Math.round(selectedStart + selectedSize / 2 - cropSize / 2),
    fullStart,
    fullEnd - cropSize,
  )
}

function boundedOutpaintDimension(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.max(1, Math.round(value))
}

function boundedDirectionalSize(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.max(DIRECTIONAL_OUTPAINT_MIN_SIZE, Math.round(value))
}

function boundedContextSize(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.max(1, Math.round(value))
}

function normalizeOutpaintDirection(direction: OutpaintDirection): OutpaintDirection {
  if (
    direction === OUTPAINT_DIRECTION_LEFT ||
    direction === OUTPAINT_DIRECTION_RIGHT ||
    direction === OUTPAINT_DIRECTION_UP ||
    direction === OUTPAINT_DIRECTION_DOWN
  ) {
    return direction
  }
  return OUTPAINT_DIRECTION_RIGHT
}

function outpaintContextRect(
  selection: SelectionRect,
  visibleBounds: DocumentBounds,
): DocumentBounds | null {
  const selectionRight = selection.x + selection.width
  const selectionBottom = selection.y + selection.height
  const visibleRight = visibleBounds.x + visibleBounds.width
  const visibleBottom = visibleBounds.y + visibleBounds.height
  if (selection.x < visibleBounds.x) {
    return {
      x: visibleBounds.x,
      ...verticalContextRange(selection, visibleBounds),
      width: Math.min(OUTPAINT_CONTEXT_SIZE, visibleBounds.width),
    }
  }
  if (selectionRight > visibleRight) {
    const width = Math.min(OUTPAINT_CONTEXT_SIZE, visibleBounds.width)
    return {
      x: visibleRight - width,
      ...verticalContextRange(selection, visibleBounds),
      width,
    }
  }
  if (selection.y < visibleBounds.y) {
    return {
      y: visibleBounds.y,
      ...horizontalContextRange(selection, visibleBounds),
      height: Math.min(OUTPAINT_CONTEXT_SIZE, visibleBounds.height),
    }
  }
  if (selectionBottom > visibleBottom) {
    const height = Math.min(OUTPAINT_CONTEXT_SIZE, visibleBounds.height)
    return {
      y: visibleBottom - height,
      ...horizontalContextRange(selection, visibleBounds),
      height,
    }
  }
  return null
}

function verticalContextRange(
  selection: SelectionRect,
  visibleBounds: DocumentBounds,
): { y: number; height: number } {
  const selectionBottom = selection.y + selection.height
  const visibleBottom = visibleBounds.y + visibleBounds.height
  const overlapTop = Math.max(selection.y, visibleBounds.y)
  const overlapBottom = Math.min(selectionBottom, visibleBottom)
  if (overlapBottom > overlapTop) {
    return { y: overlapTop, height: overlapBottom - overlapTop }
  }
  const height = Math.min(selection.height, visibleBounds.height)
  return {
    y: clamp(
      Math.round(selection.y + selection.height / 2 - height / 2),
      visibleBounds.y,
      visibleBottom - height,
    ),
    height,
  }
}

function horizontalContextRange(
  selection: SelectionRect,
  visibleBounds: DocumentBounds,
): { x: number; width: number } {
  const selectionRight = selection.x + selection.width
  const visibleRight = visibleBounds.x + visibleBounds.width
  const overlapLeft = Math.max(selection.x, visibleBounds.x)
  const overlapRight = Math.min(selectionRight, visibleRight)
  if (overlapRight > overlapLeft) {
    return { x: overlapLeft, width: overlapRight - overlapLeft }
  }
  const width = Math.min(selection.width, visibleBounds.width)
  return {
    x: clamp(
      Math.round(selection.x + selection.width / 2 - width / 2),
      visibleBounds.x,
      visibleRight - width,
    ),
    width,
  }
}

function unionBounds(first: DocumentBounds, second: DocumentBounds): SelectionRect {
  const x = Math.min(first.x, second.x)
  const y = Math.min(first.y, second.y)
  const right = Math.max(first.x + first.width, second.x + second.width)
  const bottom = Math.max(first.y + first.height, second.y + second.height)
  return {
    x,
    y,
    width: right - x,
    height: bottom - y,
  }
}

function createInitialOutpaintSelection(
  sourceWidth: number,
  sourceHeight: number,
  padding: number,
  canvasWidth: number,
  canvasHeight: number,
): SelectionRect {
  const selectionWidth = Math.min(DEFAULT_SELECTION_SIZE, canvasWidth)
  const selectionHeight = Math.min(DEFAULT_SELECTION_SIZE, canvasHeight)
  const overlap = Math.min(64, sourceWidth, selectionWidth)
  const x = padding + sourceWidth - overlap
  const y = padding + Math.round((sourceHeight - selectionHeight) / 2)
  return sanitizeSelection(
    {
      x,
      y,
      width: selectionWidth,
      height: selectionHeight,
    }
  )
}

function createOutpaintMaskImage(
  context: CanvasRenderingContext2D,
  imageData: ImageData,
  overlapPixels = 0,
): ImageData {
  const maskImage = context.createImageData(imageData.width, imageData.height)
  const maskValues = new Uint8ClampedArray(imageData.width * imageData.height)
  for (let index = 0; index < imageData.data.length; index += 4) {
    const alpha = imageData.data[index + 3]
    const value = alpha < 128 ? 255 : 0
    maskValues[index / 4] = value
  }
  const expandedMaskValues = overlapPixels > 0
    ? expandMaskValues(maskValues, imageData.width, imageData.height, overlapPixels)
    : maskValues
  for (let index = 0; index < imageData.data.length; index += 4) {
    const value = expandedMaskValues[index / 4]
    if (value === 255) {
      if (maskValues[index / 4] === 255) {
        imageData.data[index] = 0
        imageData.data[index + 1] = 0
        imageData.data[index + 2] = 0
        imageData.data[index + 3] = 0
      } else {
        imageData.data[index + 3] = 255
      }
    }
    maskImage.data[index] = value
    maskImage.data[index + 1] = value
    maskImage.data[index + 2] = value
    maskImage.data[index + 3] = 255
  }
  return maskImage
}

function expandMaskValues(
  maskValues: Uint8ClampedArray,
  width: number,
  height: number,
  overlapPixels: number,
): Uint8ClampedArray {
  const distance = new Int32Array(width * height)
  const maxDistance = width + height + overlapPixels + 1
  for (let index = 0; index < maskValues.length; index += 1) {
    distance[index] = maskValues[index] === 255 ? 0 : maxDistance
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x
      if (x > 0) {
        distance[index] = Math.min(distance[index], distance[index - 1] + 1)
      }
      if (y > 0) {
        distance[index] = Math.min(distance[index], distance[index - width] + 1)
      }
    }
  }
  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x
      if (x < width - 1) {
        distance[index] = Math.min(distance[index], distance[index + 1] + 1)
      }
      if (y < height - 1) {
        distance[index] = Math.min(distance[index], distance[index + width] + 1)
      }
    }
  }
  const expanded = new Uint8ClampedArray(maskValues.length)
  for (let index = 0; index < distance.length; index += 1) {
    expanded[index] = distance[index] <= overlapPixels ? 255 : 0
  }
  return expanded
}

function freeOutpaintOverlapPixels(
  width: number,
  height: number,
  overlapPercentage: number | undefined,
): number {
  if (typeof overlapPercentage !== 'number' || !Number.isFinite(overlapPercentage)) {
    return 0
  }
  const percentage = boundedOverlapPercentage(overlapPercentage)
  return Math.max(0, Math.min(256, Math.round(Math.min(width, height) * (percentage / 100))))
}

function imageDataHasVisiblePixels(imageData: ImageData): boolean {
  for (let index = 3; index < imageData.data.length; index += 4) {
    if (imageData.data[index] > TRANSPARENT_ALPHA) {
      return true
    }
  }
  return false
}

function drawSelectionToCanvas(
  context: CanvasRenderingContext2D,
  fullCanvas: HTMLCanvasElement,
  selection: SelectionRect,
  renderWidth: number,
  renderHeight: number,
): void {
  const scaleX = renderWidth / selection.width
  const scaleY = renderHeight / selection.height
  context.drawImage(
    fullCanvas,
    -selection.x * scaleX,
    -selection.y * scaleY,
    fullCanvas.width * scaleX,
    fullCanvas.height * scaleY,
  )
}

export function fillTransparentPixelsFromNearestKnown(
  imageData: ImageData,
  fallbackRgb: [number, number, number] = CONTROL_GUIDE_FALLBACK_RGB,
): ImageData {
  const { data, width, height } = imageData
  const queueX: number[] = []
  const queueY: number[] = []
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4
      if (data[index + 3] <= TRANSPARENT_ALPHA) {
        continue
      }
      data[index + 3] = 255
      queueX.push(x)
      queueY.push(y)
    }
  }

  if (queueX.length === 0) {
    for (let index = 0; index < data.length; index += 4) {
      data[index] = fallbackRgb[0]
      data[index + 1] = fallbackRgb[1]
      data[index + 2] = fallbackRgb[2]
      data[index + 3] = 255
    }
    return imageData
  }

  let readIndex = 0
  while (readIndex < queueX.length) {
    const x = queueX[readIndex]
    const y = queueY[readIndex]
    const sourceIndex = (y * width + x) * 4
    readIndex += 1
    for (const [offsetX, offsetY] of CONTROL_GUIDE_NEIGHBORS) {
      const nextX = x + offsetX
      const nextY = y + offsetY
      if (nextX < 0 || nextY < 0 || nextX >= width || nextY >= height) {
        continue
      }
      const nextIndex = (nextY * width + nextX) * 4
      if (data[nextIndex + 3] > TRANSPARENT_ALPHA) {
        continue
      }
      data[nextIndex] = data[sourceIndex]
      data[nextIndex + 1] = data[sourceIndex + 1]
      data[nextIndex + 2] = data[sourceIndex + 2]
      data[nextIndex + 3] = 255
      queueX.push(nextX)
      queueY.push(nextY)
    }
  }

  return imageData
}

export function clearMaskedControlGuidePixels(
  imageData: ImageData,
  maskImage: ImageData,
): ImageData {
  for (let index = 0; index < imageData.data.length; index += 4) {
    if (maskImage.data[index] < 128) {
      continue
    }
    imageData.data[index + 3] = TRANSPARENT_ALPHA
  }
  return imageData
}

async function renderInpaintMaskCanvas(documentState: EditorDocument): Promise<HTMLCanvasElement> {
  const canvas = document.createElement('canvas')
  canvas.width = documentState.width
  canvas.height = documentState.height
  const context = requireCanvasContext(canvas)
  context.fillStyle = BLACK_PIXEL
  context.fillRect(0, 0, documentState.width, documentState.height)
  if (documentState.semanticMaskDataUrl) {
    const semanticMask = await loadImageElement(documentState.semanticMaskDataUrl)
    context.drawImage(semanticMask, 0, 0, documentState.width, documentState.height)
  }
  context.lineCap = 'round'
  context.lineJoin = 'round'
  for (const stroke of documentState.maskStrokes) {
    if (stroke.points.length === 0) {
      continue
    }
    context.save()
    context.globalCompositeOperation = stroke.mode === STROKE_ERASE
      ? 'destination-out'
      : 'source-over'
    context.lineWidth = stroke.size
    context.fillStyle = WHITE_PIXEL
    context.strokeStyle = WHITE_PIXEL
    context.beginPath()
    if (stroke.points.length === 1) {
      context.arc(stroke.points[0].x, stroke.points[0].y, stroke.size / 2, 0, Math.PI * 2)
      context.fill()
    } else {
      context.moveTo(stroke.points[0].x, stroke.points[0].y)
      for (const point of stroke.points.slice(1)) {
        context.lineTo(point.x, point.y)
      }
      context.stroke()
    }
    context.restore()
  }
  return canvas
}

function renderControlGuideCanvas(
  documentState: EditorDocument,
  selection: SelectionRect,
  sourceCanvas: HTMLCanvasElement,
  maskCanvas: HTMLCanvasElement | null,
  controlGuideMaskMode: ControlGuideMaskMode,
): HTMLCanvasElement | null {
  const guideBounds = controlStrokeBounds(documentState)
  const canUseOriginalMaskGuide = Boolean(maskCanvas) &&
    controlGuideMaskMode !== CONTROL_GUIDE_MASK_MODE_REPLACE
  if (
    (!guideBounds || !boundsIntersect(guideBounds, selection)) &&
    !canUseOriginalMaskGuide
  ) {
    return null
  }
  const canvas = document.createElement('canvas')
  canvas.width = selection.width
  canvas.height = selection.height
  const context = requireCanvasContext(canvas)
  context.clearRect(0, 0, selection.width, selection.height)
  context.drawImage(sourceCanvas, 0, 0)
  const guideImage = context.getImageData(0, 0, selection.width, selection.height)
  if (maskCanvas && controlGuideMaskMode === CONTROL_GUIDE_MASK_MODE_REPLACE) {
    const maskContext = requireCanvasContext(maskCanvas)
    clearMaskedControlGuidePixels(
      guideImage,
      maskContext.getImageData(0, 0, selection.width, selection.height),
    )
  }
  fillTransparentPixelsFromNearestKnown(guideImage)
  context.putImageData(guideImage, 0, 0)
  context.lineCap = 'round'
  context.lineJoin = 'round'
  for (const stroke of documentState.controlStrokes) {
    drawControlStroke(context, stroke, selection)
  }
  return canvas
}

function drawControlStroke(
  context: CanvasRenderingContext2D,
  stroke: ControlStroke,
  selection: SelectionRect,
): void {
  if (stroke.points.length === 0) {
    return
  }
  const color = stroke.color ?? DEFAULT_CONTROL_GUIDE_COLOR
  context.save()
  context.globalAlpha = controlStrokeAlpha(stroke.strength)
  context.fillStyle = color
  context.strokeStyle = color
  context.lineWidth = stroke.size
  context.beginPath()
  if (stroke.points.length === 1) {
    context.arc(
      stroke.points[0].x - selection.x,
      stroke.points[0].y - selection.y,
      stroke.size / 2,
      0,
      Math.PI * 2,
    )
    context.fill()
    context.restore()
    return
  }
  context.moveTo(stroke.points[0].x - selection.x, stroke.points[0].y - selection.y)
  for (const point of stroke.points.slice(1)) {
    context.lineTo(point.x - selection.x, point.y - selection.y)
  }
  context.stroke()
  context.restore()
}

function controlStrokeAlpha(strength: number | undefined): number {
  const nextStrength = typeof strength === 'number' && Number.isFinite(strength)
    ? strength
    : DEFAULT_CONTROL_GUIDE_STRENGTH
  return clamp(
    nextStrength,
    MIN_CONTROL_GUIDE_STRENGTH,
    MAX_CONTROL_GUIDE_STRENGTH,
  ) / MAX_CONTROL_GUIDE_STRENGTH
}

function visibleContentSelectionFromMask(
  documentState: EditorDocument,
  fullCanvas: HTMLCanvasElement,
  maskCanvas: HTMLCanvasElement,
): SelectionRect {
  const maskBounds = combinedMaskBounds(documentState, maskCanvas)
  if (!maskBounds) {
    throw new AppError('INPAINT_MASK_REQUIRED', 'Paint an inpaint mask before generating.')
  }
  const documentBounds = {
    x: 0,
    y: 0,
    width: documentState.width,
    height: documentState.height,
  }
  if (!boundsIntersect(maskBounds, documentBounds)) {
    throw new AppError('INPAINT_MASK_REQUIRED', 'Paint an inpaint mask inside the document.')
  }
  const context = requireCanvasContext(fullCanvas)
  const imageData = context.getImageData(0, 0, fullCanvas.width, fullCanvas.height)
  const visibleBounds = findVisibleAlphaBounds(imageData.data, fullCanvas.width, fullCanvas.height)
  if (!visibleBounds) {
    throw new AppError('INPAINT_IMAGE_REQUIRED', 'Import visible image content before inpainting.')
  }
  if (!boundsIntersect(maskBounds, visibleBounds)) {
    throw new AppError('INPAINT_MASK_REQUIRED', 'Paint an inpaint mask over the visible image.')
  }
  return inpaintSelectionFromBrushCenter(visibleBounds, maskBounds)
}

function combinedMaskBounds(
  documentState: EditorDocument,
  maskCanvas: HTMLCanvasElement,
): DocumentBounds | null {
  const strokeBounds = maskStrokeBounds(documentState)
  const bitmapBounds = maskPaintBounds(maskCanvas)
  if (strokeBounds && bitmapBounds) {
    return unionBounds(strokeBounds, bitmapBounds)
  }
  return strokeBounds ?? bitmapBounds
}

function maskPaintBounds(maskCanvas: HTMLCanvasElement): DocumentBounds | null {
  const context = requireCanvasContext(maskCanvas)
  const imageData = context.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (let y = 0; y < maskCanvas.height; y += 1) {
    for (let x = 0; x < maskCanvas.width; x += 1) {
      const index = (y * maskCanvas.width + x) * 4
      if (
        imageData.data[index] < 128 &&
        imageData.data[index + 1] < 128 &&
        imageData.data[index + 2] < 128
      ) {
        continue
      }
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x + 1)
      maxY = Math.max(maxY, y + 1)
    }
  }
  if (!Number.isFinite(minX)) {
    return null
  }
  return sanitizeBounds({
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  })
}

function boundsIntersect(first: DocumentBounds, second: DocumentBounds): boolean {
  return (
    first.x < second.x + second.width
    && first.y < second.y + second.height
    && first.x + first.width > second.x
    && first.y + first.height > second.y
  )
}

export function inpaintSelectionFromBrushCenter(
  visibleBounds: DocumentBounds,
  maskBounds: DocumentBounds,
): SelectionRect {
  if (
    visibleBounds.width <= MAX_INPAINT_INPUT_SIZE
    && visibleBounds.height <= MAX_INPAINT_INPUT_SIZE
  ) {
    return sanitizeSelection(visibleBounds)
  }

  const centerX = maskBounds.x + maskBounds.width / 2
  const centerY = maskBounds.y + maskBounds.height / 2
  const expandedLeft = Math.max(
    visibleBounds.x,
    Math.floor(maskBounds.x - INPAINT_MASK_CONTEXT_PADDING),
  )
  const expandedTop = Math.max(
    visibleBounds.y,
    Math.floor(maskBounds.y - INPAINT_MASK_CONTEXT_PADDING),
  )
  const expandedRight = Math.min(
    visibleBounds.x + visibleBounds.width,
    Math.ceil(maskBounds.x + maskBounds.width + INPAINT_MASK_CONTEXT_PADDING),
  )
  const expandedBottom = Math.min(
    visibleBounds.y + visibleBounds.height,
    Math.ceil(maskBounds.y + maskBounds.height + INPAINT_MASK_CONTEXT_PADDING),
  )
  const width = Math.min(
    MAX_INPAINT_INPUT_SIZE,
    Math.max(1, expandedRight - expandedLeft),
  )
  const height = Math.min(
    MAX_INPAINT_INPUT_SIZE,
    Math.max(1, expandedBottom - expandedTop),
  )
  const x = width < expandedRight - expandedLeft
    ? clamp(
        Math.round(centerX - width / 2),
        visibleBounds.x,
        visibleBounds.x + visibleBounds.width - width,
      )
    : expandedLeft
  const y = height < expandedBottom - expandedTop
    ? clamp(
        Math.round(centerY - height / 2),
        visibleBounds.y,
        visibleBounds.y + visibleBounds.height - height,
      )
    : expandedTop

  return sanitizeSelection({ x, y, width, height })
}

function maskStrokeBounds(documentState: EditorDocument): DocumentBounds | null {
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const stroke of documentState.maskStrokes) {
    const radius = stroke.size / 2
    for (const point of stroke.points) {
      minX = Math.min(minX, point.x - radius)
      minY = Math.min(minY, point.y - radius)
      maxX = Math.max(maxX, point.x + radius)
      maxY = Math.max(maxY, point.y + radius)
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null
  }
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

function controlStrokeBounds(documentState: EditorDocument): DocumentBounds | null {
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const stroke of documentState.controlStrokes) {
    const radius = stroke.size / 2
    for (const point of stroke.points) {
      minX = Math.min(minX, point.x - radius)
      minY = Math.min(minY, point.y - radius)
      maxX = Math.max(maxX, point.x + radius)
      maxY = Math.max(maxY, point.y + radius)
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null
  }
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(value, maximum))
}

function requireCanvasContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d')
  if (!context) {
    throw new AppError('CANVAS_UNAVAILABLE', 'Canvas 2D context is not available.')
  }
  return context
}
