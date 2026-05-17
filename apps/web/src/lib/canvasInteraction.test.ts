import { describe, expect, it } from 'vitest'
import type { EditorDocument } from '../domain/types'
import { canvasCursorForPoint } from './canvasInteraction'

describe('canvas interactions', () => {
  it('uses an add-point cursor only over visible image content for canvas plugin tools', () => {
    const documentState = documentFixture()

    expect(
      canvasCursorForPoint({
        documentState,
        point: { x: 20, y: 30 },
        pluginCanvasToolActive: true,
        panning: false,
      }),
    ).toBe('crosshair')
    expect(
      canvasCursorForPoint({
        documentState,
        point: { x: 125, y: 35 },
        pluginCanvasToolActive: true,
        panning: false,
      }),
    ).toBe('crosshair')
    expect(
      canvasCursorForPoint({
        documentState,
        point: { x: 4, y: 4 },
        pluginCanvasToolActive: true,
        panning: false,
      }),
    ).toBe('default')
  })

  it('keeps the default cursor when the canvas plugin point tool is inactive', () => {
    expect(
      canvasCursorForPoint({
        documentState: documentFixture(),
        point: { x: 20, y: 30 },
        pluginCanvasToolActive: false,
        panning: false,
      }),
    ).toBe('default')
    expect(
      canvasCursorForPoint({
        documentState: documentFixture(),
        point: { x: 20, y: 30 },
        pluginCanvasToolActive: true,
        panning: true,
      }),
    ).toBe('default')
  })
})

function documentFixture(): EditorDocument {
  return {
    id: 'test',
    width: 200,
    height: 120,
    rasterDataUrl: 'data:image/png;base64,test',
    rasterBounds: { x: 10, y: 20, width: 80, height: 50 },
    selection: { x: 0, y: 0, width: 64, height: 64 },
    semanticMaskDataUrl: null,
    maskStrokes: [],
    controlStrokes: [],
    references: [
      {
        id: 'ref',
        dataUrl: 'data:image/png;base64,ref',
        x: 120,
        y: 20,
        width: 40,
        height: 40,
        opacity: 1,
      },
    ],
  }
}
