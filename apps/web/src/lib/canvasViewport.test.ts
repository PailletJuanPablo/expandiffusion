import { describe, expect, it } from 'vitest'
import { planCenteredDocumentViewport } from './canvasViewport'

describe('canvasViewport', () => {
  it('centers imported image and outpaint frame bounds in the viewport', () => {
    expect(
      planCenteredDocumentViewport(
        { x: 512, y: 320, width: 576, height: 512 },
        { width: 1000, height: 700 },
      ),
    ).toEqual({
      x: -300,
      y: -226,
      zoom: 1,
    })
  })
})
