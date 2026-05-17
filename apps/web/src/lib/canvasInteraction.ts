import type { DocumentBounds, EditorDocument, Point } from '../domain/types'

interface CanvasCursorInput {
  documentState: EditorDocument
  point: Point | null
  pluginCanvasToolActive: boolean
  panning: boolean
}

export function canvasCursorForPoint({
  documentState,
  point,
  pluginCanvasToolActive,
  panning,
}: CanvasCursorInput): string {
  if (!pluginCanvasToolActive || panning || !point) {
    return 'default'
  }
  return pointHitsVisibleImageContent(documentState, point) ? 'crosshair' : 'default'
}

function pointHitsVisibleImageContent(documentState: EditorDocument, point: Point): boolean {
  if (documentState.rasterDataUrl) {
    const rasterBounds = documentState.rasterBounds ?? {
      x: 0,
      y: 0,
      width: documentState.width,
      height: documentState.height,
    }
    if (boundsContainPoint(rasterBounds, point)) {
      return true
    }
  }
  return documentState.references.some((reference) =>
    boundsContainPoint(
      {
        x: reference.x,
        y: reference.y,
        width: reference.width,
        height: reference.height,
      },
      point,
    ),
  )
}

function boundsContainPoint(bounds: DocumentBounds, point: Point): boolean {
  return (
    point.x >= bounds.x &&
    point.y >= bounds.y &&
    point.x <= bounds.x + bounds.width &&
    point.y <= bounds.y + bounds.height
  )
}
