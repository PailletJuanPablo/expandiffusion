import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  CONDITIONING_TYPE_COLOR,
  CONTROL_GUIDE_MASK_MODE_PRESERVE,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
  OUTPAINT_DIRECTION_DOWN,
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
  planOutpaintRender,
  renderGenerationInputs,
} from './canvasRender'

class FakeCanvas {
  private canvasWidth = 0
  private canvasHeight = 0
  pixels = new Uint8ClampedArray()
  lastPaintStyle: string | null = null

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
    return `data:image/png;base64,${this.width}x${this.height}${style}`
  }

  private resetPixels(): void {
    this.pixels = new Uint8ClampedArray(this.canvasWidth * this.canvasHeight * 4)
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

  fillRect(): void {
    for (let index = 0; index < this.canvas.pixels.length; index += 4) {
      this.canvas.pixels[index + 3] = 255
    }
  }

  drawImage(): void {
    for (let index = 0; index < this.canvas.pixels.length; index += 4) {
      this.canvas.pixels[index] = 32
      this.canvas.pixels[index + 1] = 32
      this.canvas.pixels[index + 2] = 36
      this.canvas.pixels[index + 3] = 255
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

  putImageData(): void {}
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

class FakeImage {
  naturalWidth = 128
  naturalHeight = 128
  onload: (() => void) | null = null
  onerror: (() => void) | null = null

  set src(_value: string) {
    queueMicrotask(() => this.onload?.())
  }
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

  it('renders HF Space fill as a full-frame replacement input', async () => {
    const inputs = await renderGenerationInputs(documentWithGuide(), GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_RIGHT,
      hfSpaceOverlapPercentage: 10,
    })

    expect(inputs.selection).toEqual({ x: 0, y: 0, width: 128, height: 128 })
    expect(inputs.previewSelection).toEqual({ x: 0, y: 0, width: 128, height: 128 })
    expect(inputs.replaceDocument).toBe(true)
    expect(inputs.compositionMask).toBeNull()
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
      maskStrokes: [],
      controlStrokes: [],
      references: [],
    }

    const inputs = await renderGenerationInputs(documentState, GENERATION_MODE_OUTPAINT, {
      outpaintStrategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaintDirection: OUTPAINT_DIRECTION_RIGHT,
    })

    expect(inputs.image).toContain('700x600')
    expect(inputs.selection).toEqual({ x: -398, y: 0, width: 1496, height: 600 })
    expect(inputs.previewSelection).toEqual({ x: -398, y: 0, width: 1496, height: 600 })
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
})

function documentWithGuide(): EditorDocument {
  return {
    id: 'document',
    width: 128,
    height: 128,
    rasterDataUrl: 'data:image/png;base64,raster',
    rasterBounds: { x: 0, y: 0, width: 128, height: 128 },
    selection: { x: 0, y: 0, width: 128, height: 128 },
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
