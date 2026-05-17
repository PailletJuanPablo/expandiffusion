import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  CONDITIONING_TYPE_COLOR,
  CONTROL_GUIDE_MASK_MODE_PRESERVE,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  OUTPAINT_DIRECTION_DOWN,
  OUTPAINT_DIRECTION_AROUND,
  OUTPAINT_DIRECTION_LEFT,
  OUTPAINT_DIRECTION_RIGHT,
  OUTPAINT_DIRECTION_UP,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
  OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  OUTPAINT_STRATEGY_SELECTED_FRAME,
  OUTPAINT_STRATEGY_WHOLE_RESIZED,
  STROKE_PAINT,
  type OutpaintDirection,
} from '../constants/domain'
import type { EditorDocument } from '../domain/types'
import {
  clearMaskedControlGuidePixels,
  expandOutpaintSelectionWithContext,
  fillTransparentPixelsFromNearestKnown,
  inpaintSelectionFromBrushCenter,
  planDirectionalOutpaintRender,
  directionalGeneratedBounds,
  composeSelectionResults,
  eraseSemanticMaskFromDocument,
  eraseRasterSelection,
  planHfSpaceFillExpansion,
  planOutpaintRender,
  prepareRasterImport,
  renderDocumentDataUrl,
  renderGenerationInputs,
  renderPluginActionInput,
  renderPluginMaskToDocumentMask,
} from './canvasRender'

class FakeCanvas {
  private canvasWidth = 0
  private canvasHeight = 0
  pixels = new Uint8ClampedArray()
  lastPaintStyle: string | null = null
  lastTransparentPixels: number | null = null

  get width(): number {
    return this.canvasWidth
  }

  set width(value: number) {
    this.canvasWidth = value
    this.resetPixels()
  }

  get height(): number {
    return this.canvasHeight
  }

  set height(value: number) {
    this.canvasHeight = value
    this.resetPixels()
  }

  getContext(type: string): FakeCanvasContext | null {
    return type === '2d' ? new FakeCanvasContext(this) : null
  }

  toDataURL(): string {
    const style = this.lastPaintStyle ? `:${this.lastPaintStyle}` : ''
    const transparent = this.lastTransparentPixels === null
      ? ''
      : `:transparent=${this.lastTransparentPixels}`
    return `data:image/png;base64,${this.width}x${this.height}${style}${transparent}`
  }

  private resetPixels(): void {
    this.pixels = new Uint8ClampedArray(this.canvasWidth * this.canvasHeight * 4)
    this.lastTransparentPixels = null
  }
}

class FakeCanvasContext {
  fillStyle = '#000000'
  strokeStyle = '#000000'
  lineWidth = 1
  lineCap = 'butt'
  lineJoin = 'miter'
  globalAlpha = 1
  globalCompositeOperation = 'source-over'
  private readonly canvas: FakeCanvas

  constructor(canvas: FakeCanvas) {
    this.canvas = canvas
  }

  clearRect(): void {
    this.canvas.pixels.fill(0)
  }

  fillRect(x = 0, y = 0, width = this.canvas.width, height = this.canvas.height): void {
    const left = Math.max(0, Math.round(x))
    const top = Math.max(0, Math.round(y))
    const right = Math.min(this.canvas.width, Math.round(x + width))
    const bottom = Math.min(this.canvas.height, Math.round(y + height))
    let transparentPixels = 0
    for (let row = top; row < bottom; row += 1) {
      for (let column = left; column < right; column += 1) {
        const index = (row * this.canvas.width + column) * 4
        if (this.globalCompositeOperation === 'destination-out') {
          this.canvas.pixels[index + 3] = 0
          transparentPixels += 1
        } else {
          this.canvas.pixels[index + 3] = 255
        }
      }
    }
    if (this.globalCompositeOperation === 'destination-out') {
      this.canvas.lastTransparentPixels = transparentPixels
    }
  }

  drawImage(_image: unknown, ...args: number[]): void {
    const paintStyle =
      typeof _image === 'object' &&
      _image !== null &&
      'paintStyle' in _image &&
      typeof _image.paintStyle === 'string'
        ? hexToRgb(_image.paintStyle)
        : [32, 32, 36]
    const destination = destinationRectFromDrawImageArgs(
      args,
      this.canvas.width,
      this.canvas.height,
    )
    const left = Math.max(0, Math.round(destination.x))
    const top = Math.max(0, Math.round(destination.y))
    const right = Math.min(this.canvas.width, Math.round(destination.x + destination.width))
    const bottom = Math.min(this.canvas.height, Math.round(destination.y + destination.height))
    let transparentPixels = 0
    for (let y = top; y < bottom; y += 1) {
      for (let x = left; x < right; x += 1) {
        const index = (y * this.canvas.width + x) * 4
        if (this.globalCompositeOperation === 'destination-out') {
          const sourceAlpha = sourceAlphaAt(_image, x - left, y - top, destination)
          if (sourceAlpha > 0) {
            this.canvas.pixels[index + 3] = 0
            transparentPixels += 1
          }
          continue
        }
        this.canvas.pixels[index] = paintStyle[0]
        this.canvas.pixels[index + 1] = paintStyle[1]
        this.canvas.pixels[index + 2] = paintStyle[2]
        this.canvas.pixels[index + 3] = 255
      }
    }
    if (this.globalCompositeOperation === 'destination-out') {
      this.canvas.lastTransparentPixels = transparentPixels
    }
  }

  getImageData(): ImageData {
    return {
      width: this.canvas.width,
      height: this.canvas.height,
      data: this.canvas.pixels,
      colorSpace: 'srgb',
    } as ImageData
  }

  createImageData(width: number, height: number): ImageData {
    return {
      width,
      height,
      data: new Uint8ClampedArray(width * height * 4),
      colorSpace: 'srgb',
    } as ImageData
  }

  putImageData(imageData: ImageData): void {
    this.canvas.pixels.set(imageData.data)
  }
  save(): void {}
  restore(): void {}
  beginPath(): void {}
  arc(): void {}
  fill(): void {
    this.canvas.lastPaintStyle = String(this.fillStyle)
  }
  moveTo(): void {}
  lineTo(): void {}
  stroke(): void {
    this.canvas.lastPaintStyle = String(this.strokeStyle)
  }
}

function hexToRgb(value: string): [number, number, number] {
  return [
    Number.parseInt(value.slice(1, 3), 16),
    Number.parseInt(value.slice(3, 5), 16),
    Number.parseInt(value.slice(5, 7), 16),
  ]
}

class FakeImage {
  naturalWidth = 128
  naturalHeight = 128
  paintStyle = '#202024'
  onload: (() => void) | null = null
  onerror: (() => void) | null = null

  set src(value: string) {
    const size = value.match(/(\d+)x(\d+)/)
    if (size) {
      this.naturalWidth = Number.parseInt(size[1], 10)
      this.naturalHeight = Number.parseInt(size[2], 10)
    }
    const style = value.match(/:(#[a-fA-F0-9]{6})$/)
    if (style) {
      this.paintStyle = style[1]
    }
    queueMicrotask(() => this.onload?.())
  }
}

function destinationRectFromDrawImageArgs(
  args: number[],
  canvasWidth: number,
  canvasHeight: number,
): { x: number; y: number; width: number; height: number } {
  if (args.length >= 8) {
    return { x: args[4], y: args[5], width: args[6], height: args[7] }
  }
  if (args.length >= 4) {
    return { x: args[0], y: args[1], width: args[2], height: args[3] }
  }
  if (args.length >= 2) {
    return { x: args[0], y: args[1], width: canvasWidth, height: canvasHeight }
  }
  return { x: 0, y: 0, width: canvasWidth, height: canvasHeight }
}

function sourceAlphaAt(
  image: unknown,
  x: number,
  y: number,
  destination: { width: number; height: number },
): number {
  if (image instanceof FakeCanvas) {
    const sourceX = Math.max(0, Math.min(image.width - 1, Math.floor(x * image.width / destination.width)))
    const sourceY = Math.max(0, Math.min(image.height - 1, Math.floor(y * image.height / destination.height)))
    return image.pixels[(sourceY * image.width + sourceX) * 4 + 3]
  }
  return 255
}

describe('canvasRender', () => {
  beforeEach(() => {
    vi.stubGlobal('document', {
      createElement: (tagName: string) => {
        if (tagName !== 'canvas') {
          throw new TypeError(`Unsupported element: ${tagName}`)
        }
        return new FakeCanvas()
      },
    })
    vi.stubGlobal('Image', FakeImage)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses mask context instead of a full 1024 window for large inpaint images', () => {
    const selection = inpaintSelectionFromBrushCenter(
      { x: 100, y: 200, width: 1800, height: 1400 },
      { x: 1300, y: 900, width: 80, height: 60 },
    )

    expect(selection).toEqual({
      x: 1044,
      y: 644,
      width: 592,
      height: 572,
    })
  })

  it('clamps the inpaint crop inside the visible image bounds', () => {
    const selection = inpaintSelectionFromBrushCenter(
      { x: 100, y: 200, width: 1800, height: 1400 },
      { x: 120, y: 230, width: 40, height: 40 },
    )

    expect(selection).toEqual({
      x: 100,
      y: 200,
      width: 316,
      height: 326,
    })
  })

  it('keeps the full visible image when it fits the inpaint limit', () => {
    const selection = inpaintSelectionFromBrushCenter(
      { x: 40, y: 50, width: 900, height: 700 },
      { x: 600, y: 500, width: 100, height: 80 },
    )

    expect(selection).toEqual({
      x: 40,
      y: 50,
      width: 900,
      height: 700,
    })
  })

  it('places the initial outpaint frame beside the imported image edge', async () => {
    const raster = await prepareRasterImport('data:image/png;base64,raster')

    expect(raster.rasterBounds).toEqual({
      x: 512,
      y: 512,
      width: 128,
      height: 128,
    })
    expect(raster.selection).toEqual({
      x: 576,
      y: 320,
      width: 512,
      height: 512,
    })
  })

  it('downscales oversized raster imports before creating the workspace', async () => {
    const raster = await prepareRasterImport('data:image/png;base64,6000x3000')

    expect(raster.dataUrl).toContain('1536x768')
    expect(raster.rasterBounds).toEqual({
      x: 512,
      y: 512,
      width: 1536,
      height: 768,
    })
    expect(raster.width).toBe(2560)
    expect(raster.height).toBe(1792)
  })

  it('upscales tiny raster imports to a usable model input size', async () => {
    const raster = await prepareRasterImport('data:image/png;base64,32x16')

    expect(raster.dataUrl).toContain('128x64')
    expect(raster.rasterBounds).toEqual({
      x: 512,
      y: 512,
      width: 128,
      height: 64,
    })
  })

  it('exports only the visible document pixels', async () => {
    const dataUrl = await renderDocumentDataUrl({
      id: 'document',
      width: 240,
      height: 180,
      rasterDataUrl: 'data:image/png;base64,raster',
      rasterBounds: { x: 70, y: 55, width: 80, height: 40 },
      selection: { x: 0, y: 0, width: 128, height: 128 },
      semanticMaskDataUrl: null,
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    })

    expect(dataUrl).toContain('80x40')
  })

  it('expands a left outpaint frame with adjacent source context', () => {
    const selection = expandOutpaintSelectionWithContext(
      { x: -980, y: 44, width: 824, height: 1024 },
      { x: 0, y: 0, width: 1024, height: 1024 },
    )

    expect(selection).toEqual({
      x: -980,
      y: 44,
      width: 1492,
      height: 1024,
    })
  })

  it('expands a right outpaint frame with adjacent source context', () => {
    const selection = expandOutpaintSelectionWithContext(
      { x: 1024, y: 0, width: 512, height: 1024 },
      { x: 0, y: 0, width: 1024, height: 1024 },
    )

    expect(selection).toEqual({
      x: 512,
      y: 0,
      width: 1024,
      height: 1024,
    })
  })

  it('plans selected-frame outpaint without adding source context', () => {
    const plan = planOutpaintRender(
      { x: -980, y: 44, width: 824, height: 1024 },
      { x: 0, y: 0, width: 1024, height: 1024 },
      OUTPAINT_STRATEGY_SELECTED_FRAME,
    )

    expect(plan).toEqual({
      selection: { x: -980, y: 44, width: 824, height: 1024 },
      renderWidth: 824,
      renderHeight: 1024,
    })
  })

  it('plans local-context outpaint with adjacent source context at full scale', () => {
    const plan = planOutpaintRender(
      { x: -980, y: 44, width: 824, height: 1024 },
      { x: 0, y: 0, width: 1024, height: 1024 },
      OUTPAINT_STRATEGY_LOCAL_CONTEXT,
    )

    expect(plan).toEqual({
      selection: { x: -980, y: 44, width: 1492, height: 1024 },
      renderWidth: 1492,
      renderHeight: 1024,
    })
  })

  it('plans whole-resized outpaint with the full visible source scaled down', () => {
    const plan = planOutpaintRender(
      { x: -980, y: 44, width: 824, height: 1024 },
      { x: 0, y: 0, width: 1024, height: 1024 },
      OUTPAINT_STRATEGY_WHOLE_RESIZED,
    )

    expect(plan).toEqual({
      selection: { x: -980, y: 0, width: 2004, height: 1068 },
      renderWidth: 1024,
      renderHeight: 546,
    })
  })

  it('plans full-context crop outpaint by preserving the full left mask at source scale', () => {
    const plan = planOutpaintRender(
      { x: -1024, y: 0, width: 1024, height: 1024 },
      { x: 0, y: 0, width: 3239, height: 1024 },
      OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
      { maxWidth: 1536, maxHeight: 1024 },
    )

    expect(plan).toEqual({
      selection: { x: -1024, y: 0, width: 1536, height: 1024 },
      renderWidth: 1536,
      renderHeight: 1024,
    })
  })

  it('plans full-context crop outpaint by preserving the full right mask at source scale', () => {
    const plan = planOutpaintRender(
      { x: 3239, y: 0, width: 1024, height: 1024 },
      { x: 0, y: 0, width: 3239, height: 1024 },
      OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
      { maxWidth: 1536, maxHeight: 1024 },
    )

    expect(plan).toEqual({
      selection: { x: 2727, y: 0, width: 1536, height: 1024 },
      renderWidth: 1536,
      renderHeight: 1024,
    })
  })

  it('plans full-context crop outpaint by preserving the full top mask at source scale', () => {
    const plan = planOutpaintRender(
      { x: 0, y: -768, width: 1024, height: 768 },
      { x: 0, y: 0, width: 1024, height: 2400 },
      OUTPAINT_STRATEGY_FULL_CONTEXT_CROP,
      { maxWidth: 1024, maxHeight: 1280 },
    )

    expect(plan).toEqual({
      selection: { x: 0, y: -768, width: 1024, height: 1280 },
      renderWidth: 1024,
      renderHeight: 1280,
    })
  })

  it('clamps directional outpaint generation and cross sizes to the SDXL minimum', () => {
    const plan = planDirectionalOutpaintRender(
      { x: 200, y: 100, width: 800, height: 400 },
      OUTPAINT_DIRECTION_RIGHT,
      {
        generatedSize: 512,
        contextSize: 512,
        crossSize: 768,
      },
    )

    expect(plan.width).toBe(1536)
    expect(plan.height).toBe(1024)
    expect(plan.generatedRect).toEqual({ x: 512, y: 0, width: 1024, height: 1024 })
    expect(plan.contextRect).toEqual({ x: 0, y: 0, width: 512, height: 1024 })
  })

  const directionalOutpaintCases: Array<[
    OutpaintDirection,
    {
      width: number
      height: number
      contextRect: { x: number; y: number; width: number; height: number }
      generatedRect: { x: number; y: number; width: number; height: number }
    },
  ]> = [
    [
      OUTPAINT_DIRECTION_RIGHT,
      {
        width: 1536,
        height: 1024,
        contextRect: { x: 0, y: 0, width: 512, height: 1024 },
        generatedRect: { x: 512, y: 0, width: 1024, height: 1024 },
      },
    ],
    [
      OUTPAINT_DIRECTION_LEFT,
      {
        width: 1536,
        height: 1024,
        contextRect: { x: 1024, y: 0, width: 512, height: 1024 },
        generatedRect: { x: 0, y: 0, width: 1024, height: 1024 },
      },
    ],
    [
      OUTPAINT_DIRECTION_DOWN,
      {
        width: 1024,
        height: 1536,
        contextRect: { x: 0, y: 0, width: 1024, height: 512 },
        generatedRect: { x: 0, y: 512, width: 1024, height: 1024 },
      },
    ],
    [
      OUTPAINT_DIRECTION_UP,
      {
        width: 1024,
        height: 1536,
        contextRect: { x: 0, y: 1024, width: 1024, height: 512 },
        generatedRect: { x: 0, y: 0, width: 1024, height: 1024 },
      },
    ],
  ]

  it.each(directionalOutpaintCases)('plans fixed directional context and mask bboxes for %s', (direction, expected) => {
    const plan = planDirectionalOutpaintRender(
      { x: 200, y: 100, width: 800, height: 400 },
      direction,
      {
        generatedSize: 1024,
        contextSize: 512,
        crossSize: 1024,
      },
    )

    expect(plan.width).toBe(expected.width)
    expect(plan.height).toBe(expected.height)
    expect(plan.contextRect).toEqual(expected.contextRect)
    expect(plan.generatedRect).toEqual(expected.generatedRect)
  })

  it('renders non-fixed HF Space fill as a normal free-frame outpaint input', async () => {
    const inputs = await renderGenerationInputs(documentWithGuide(), GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_RIGHT,
      hfSpaceOverlapPercentage: 10,
      hfSpaceFixedExpansion: false,
    })

    expect(inputs.selection).toEqual({ x: 0, y: 0, width: 128, height: 128 })
    expect(inputs.previewSelection).toEqual({ x: 0, y: 0, width: 128, height: 128 })
    expect(inputs.replaceDocument).toBe(false)
    expect(inputs.compositionMask).toContain('128x128')
    expect(inputs.directionalPlan).toBeNull()
    expect(inputs.image).toContain('128x128')
    expect(inputs.mask).toContain('128x128')
  })

  it('renders HF Space fill from the adjusted raster instead of the clipped document canvas', async () => {
    const documentState: EditorDocument = {
      id: 'document',
      width: 35,
      height: 600,
      rasterDataUrl: 'data:image/png;base64,raster',
      rasterBounds: { x: -665, y: 0, width: 700, height: 600 },
      selection: { x: -398, y: 0, width: 1500, height: 600 },
      semanticMaskDataUrl: null,
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_RIGHT,
      hfSpaceFixedExpansion: true,
    })

    expect(inputs.image).toContain('700x600')
    expect(inputs.selection).toEqual({ x: -665, y: -212, width: 1720, height: 1024 })
    expect(inputs.previewSelection).toEqual(inputs.selection)
  })

  it('plans fixed HF Space around expansion as a centered output frame', async () => {
    const documentState: EditorDocument = {
      id: 'document',
      width: 1200,
      height: 900,
      rasterDataUrl: 'data:image/png;base64,raster',
      rasterBounds: { x: 100, y: 50, width: 700, height: 600 },
      selection: { x: 0, y: 0, width: 128, height: 128 },
      semanticMaskDataUrl: null,
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_AROUND,
      outpaintGeneratedSize: 1536,
      outpaintCrossSize: 1024,
      hfSpaceFixedExpansion: true,
      hfSpaceResizeOption: '50%',
      hfSpaceOverlapPercentage: 12,
    })

    expect(inputs.selection).toEqual({ x: -318, y: -162, width: 1536, height: 1024 })
    expect(inputs.previewSelection).toEqual(inputs.selection)
    expect(inputs.replaceDocument).toBe(true)
    expect(inputs.directionalPlan).toBeNull()
  })

  it('plans fixed HF Space side expansion from a percentage', () => {
    const plan = planHfSpaceFillExpansion(
      { x: 100, y: 50, width: 800, height: 600 },
      {
        direction: OUTPAINT_DIRECTION_RIGHT,
        expansionPercent: 50,
      },
    )

    expect(plan.selection).toEqual({ x: 100, y: 50, width: 1200, height: 600 })
    expect(plan.renderSize).toEqual({ width: 1200, height: 600 })
  })

  it('plans fixed HF Space around expansion with separate width and height percentages', () => {
    const plan = planHfSpaceFillExpansion(
      { x: 100, y: 50, width: 800, height: 600 },
      {
        direction: OUTPAINT_DIRECTION_AROUND,
        widthExpansionPercent: 50,
        heightExpansionPercent: 25,
      },
    )

    expect(plan.selection).toEqual({ x: -100, y: -22, width: 1200, height: 744 })
    expect(plan.renderSize).toEqual({ width: 1200, height: 744 })
  })

  it('keeps the visual fixed frame while rendering HF Space fill at a lower scale', async () => {
    const documentState: EditorDocument = {
      id: 'document',
      width: 1200,
      height: 900,
      rasterDataUrl: 'data:image/png;base64,raster',
      rasterBounds: { x: 100, y: 50, width: 800, height: 600 },
      selection: { x: 0, y: 0, width: 128, height: 128 },
      semanticMaskDataUrl: null,
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_RIGHT,
      hfSpaceFixedExpansion: true,
      fixedExpandPercent: 50,
      fixedExpandOutputScalePercent: 50,
    })

    expect(inputs.selection).toEqual({ x: 100, y: 50, width: 1200, height: 600 })
    expect(inputs.renderSize).toEqual({ width: 600, height: 296 })
    expect(inputs.previewSelection).toEqual(inputs.selection)
  })

  it('places full replacement results at the selected output frame instead of the origin', async () => {
    const composed = await composeSelectionResults(
      documentWithGuide(),
      { x: -398, y: 24, width: 1496, height: 600 },
      ['data:image/png;base64,result'],
      null,
      { replaceDocument: true },
    )

    expect(composed.bounds).toEqual({ x: -398, y: 24, width: 1496, height: 600 })
    expect(composed.images[0]).toBe('data:image/png;base64,1496x600')
  })

  it('maps a right directional result to the real right edge instead of the document origin', () => {
    const plan = planDirectionalOutpaintRender(
      { x: 512, y: 512, width: 1920, height: 1080 },
      OUTPAINT_DIRECTION_RIGHT,
      {
        generatedSize: 1024,
        contextSize: 512,
        crossSize: 1024,
      },
    )

    expect(directionalGeneratedBounds(plan)).toEqual({
      x: 2432,
      y: 512,
      width: 1080,
      height: 1080,
    })
  })

  it('composes a right directional result beside the original canvas', async () => {
    const plan = planDirectionalOutpaintRender(
      { x: 512, y: 512, width: 1920, height: 1080 },
      OUTPAINT_DIRECTION_RIGHT,
      {
        generatedSize: 1024,
        contextSize: 512,
        crossSize: 1024,
      },
    )
    const documentState: EditorDocument = {
      id: 'document',
      width: 2944,
      height: 1800,
      rasterDataUrl: 'data:image/png;base64,raster',
      rasterBounds: { x: 512, y: 512, width: 1920, height: 1080 },
      selection: { x: 0, y: 0, width: 512, height: 512 },
      semanticMaskDataUrl: null,
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    }

    const composed = await composeSelectionResults(
      documentState,
      { x: 0, y: 0, width: 1536, height: 1024 },
      ['data:image/png;base64,result'],
      null,
      { directionalPlan: plan },
    )

    expect(composed.bounds).toEqual({
      x: 0,
      y: 0,
      width: 3512,
      height: 1800,
    })
    expect(composed.images[0]).toBe('data:image/png;base64,3512x1800')
  })

  it('fills transparent control guide pixels from nearby known color instead of white', () => {
    const imageData = {
      width: 3,
      height: 1,
      colorSpace: 'srgb',
      data: new Uint8ClampedArray([
        20, 30, 40, 255,
        0, 0, 0, 0,
        0, 0, 0, 0,
      ]),
    } as ImageData

    fillTransparentPixelsFromNearestKnown(imageData)

    expect(Array.from(imageData.data.slice(4, 8))).toEqual([20, 30, 40, 255])
    expect(Array.from(imageData.data.slice(8, 12))).toEqual([20, 30, 40, 255])
  })

  it('removes original masked pixels from the inpaint control guide before edge fill', () => {
    const guideImage = {
      width: 3,
      height: 1,
      colorSpace: 'srgb',
      data: new Uint8ClampedArray([
        20, 30, 40, 255,
        200, 210, 220, 255,
        60, 70, 80, 255,
      ]),
    } as ImageData
    const maskImage = {
      width: 3,
      height: 1,
      colorSpace: 'srgb',
      data: new Uint8ClampedArray([
        0, 0, 0, 255,
        255, 255, 255, 255,
        0, 0, 0, 255,
      ]),
    } as ImageData

    clearMaskedControlGuidePixels(guideImage, maskImage)
    fillTransparentPixelsFromNearestKnown(guideImage)

    expect(Array.from(guideImage.data.slice(4, 8))).toEqual([20, 30, 40, 255])
  })

  it('erases a rectangular raster selection to transparency', async () => {
    const dataUrl = await eraseRasterSelection(documentWithGuide(), {
      x: 16,
      y: 20,
      width: 24,
      height: 12,
    })

    expect(dataUrl).toBe('data:image/png;base64,128x128:transparent=288')
  })

  it('omits the control guide when ControlNet sketch is disabled', async () => {
    const documentState = documentWithGuide()

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_INPAINT)

    expect(inputs.conditioning).toBeNull()
  })

  it('includes the control guide when rendering inpaint inputs with ControlNet sketch enabled', async () => {
    const documentState = documentWithGuide()

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_INPAINT, {
      includeControlGuide: true,
    })

    expect(inputs.conditioning).toEqual({
      type: CONDITIONING_TYPE_COLOR,
      image: 'data:image/png;base64,128x128:#1e88e5',
    })
  })

  it('uses the original masked area as the inpaint control guide when preserve mode is selected', async () => {
    const documentState = { ...documentWithGuide(), controlStrokes: [] }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_INPAINT, {
      includeControlGuide: true,
      controlGuideMaskMode: CONTROL_GUIDE_MASK_MODE_PRESERVE,
    })

    expect(inputs.conditioning).toEqual({
      type: CONDITIONING_TYPE_COLOR,
      image: 'data:image/png;base64,128x128',
    })
  })

  it('includes the control guide when rendering outpaint inputs with ControlNet sketch enabled', async () => {
    const documentState = documentWithGuide()

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT, {
      includeControlGuide: true,
    })

    expect(inputs.conditioning).toEqual({
      type: CONDITIONING_TYPE_COLOR,
      image: 'data:image/png;base64,128x128:#1e88e5',
    })
  })

  it('renders the visible canvas as a plugin canvas target with click metadata', async () => {
    const input = await renderPluginActionInput(documentWithGuide(), {
      kind: 'canvas',
      point: { x: 64, y: 68 },
      points: [{ x: 64, y: 68 }, { x: 70, y: 72 }],
    })

    expect(input.image).toBe('data:image/png;base64,128x128')
    expect(input.target).toMatchObject({
      kind: 'canvas',
      bounds: { x: 0, y: 0, width: 128, height: 128 },
      scale: 1,
      point: { x: 64, y: 68 },
      points: [{ x: 64, y: 68 }, { x: 70, y: 72 }],
    })
    expect(input.target.visible_mask).toBe('data:image/png;base64,128x128')
  })

  it('uses a semantic object mask as an inpaint mask without brush strokes', async () => {
    const documentState = {
      ...documentWithGuide(),
      maskStrokes: [],
      semanticMaskDataUrl: 'data:image/png;base64,128x128:#ffffff',
    }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_INPAINT)

    expect(inputs.selection).toEqual({ x: 0, y: 0, width: 128, height: 128 })
    expect(inputs.mask).toBe('data:image/png;base64,128x128')
  })

  it('maps a plugin canvas mask back to document dimensions', async () => {
    const mask = await renderPluginMaskToDocumentMask(
      documentWithGuide(),
      'data:image/png;base64,64x64:#ffffff',
      {
        kind: 'canvas',
        bounds: { x: 32, y: 16, width: 64, height: 64 },
        scale: 1,
      },
    )

    expect(mask).toBe('data:image/png;base64,128x128')
  })

  it('does not erase document pixels from black semantic mask areas', async () => {
    const erased = await eraseSemanticMaskFromDocument(
      documentWithGuide(),
      'data:image/png;base64,128x128:#000000',
    )

    expect(erased).toBe('data:image/png;base64,128x128:transparent=0')
  })
})

function documentWithGuide(): EditorDocument {
  return {
    id: 'document',
    width: 128,
    height: 128,
    rasterDataUrl: 'data:image/png;base64,raster',
    rasterBounds: { x: 0, y: 0, width: 128, height: 128 },
    selection: { x: 0, y: 0, width: 128, height: 128 },
    semanticMaskDataUrl: null,
    maskStrokes: [
      {
        id: 'mask',
        mode: STROKE_PAINT,
        size: 24,
        points: [
          { x: 58, y: 58 },
          { x: 72, y: 72 },
        ],
      },
    ],
    controlStrokes: [
      {
        id: 'guide',
        size: 16,
        color: '#1e88e5',
        points: [
          { x: 56, y: 48 },
          { x: 92, y: 68 },
        ],
      },
    ],
    references: [],
  }
}
