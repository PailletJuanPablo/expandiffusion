import { MAX_ZOOM, MIN_ZOOM } from '../constants/domain'
import type { DocumentBounds, ViewportState } from '../domain/types'

const VIEWPORT_FIT_PADDING = 80

/**
 * Plan a viewport that centers document bounds inside the visible canvas area.
 *
 * @param bounds - Document bounds that should be visible.
 * @param container - Canvas container size in screen pixels.
 * @returns Pan and zoom values for the editor viewport.
 */
export function planCenteredDocumentViewport(
  bounds: DocumentBounds,
  container: { width: number; height: number },
): ViewportState {
  const availableWidth = Math.max(1, container.width - VIEWPORT_FIT_PADDING * 2)
  const availableHeight = Math.max(1, container.height - VIEWPORT_FIT_PADDING * 2)
  const zoom = clamp(
    Math.min(1, availableWidth / bounds.width, availableHeight / bounds.height),
    MIN_ZOOM,
    MAX_ZOOM,
  )
  return {
    x: Math.round((container.width - bounds.width * zoom) / 2 - bounds.x * zoom),
    y: Math.round((container.height - bounds.height * zoom) / 2 - bounds.y * zoom),
    zoom,
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}
