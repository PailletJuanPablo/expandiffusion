import { describe, expect, it } from 'vitest'
import { POSTPROCESSOR_CATEGORY_CORRECTION } from '../constants/domain'
import type { PostprocessorInfo } from '../domain/types'
import {
  activeCorrectionItems,
  appendCorrection,
  correctionPostprocessors,
  moveCorrection,
  removeCorrection,
} from './correctionPipeline'

describe('correctionPipeline', () => {
  it('filters and sorts correction postprocessors', () => {
    const corrections = correctionPostprocessors([
      postprocessor('detailer', 'Detailer', 'generation', 1),
      postprocessor('late', 'Late', POSTPROCESSOR_CATEGORY_CORRECTION, 20),
      postprocessor('early', 'Early', POSTPROCESSOR_CATEGORY_CORRECTION, 10),
    ])

    expect(corrections.map((item) => item.id)).toEqual(['early', 'late'])
  })

  it('appends and removes correction ids without duplicates', () => {
    expect(appendCorrection(['a'], 'a')).toEqual(['a'])
    expect(appendCorrection(['a'], 'b')).toEqual(['a', 'b'])
    expect(removeCorrection(['a', 'b'], 'a')).toEqual(['b'])
  })

  it('moves active corrections by one position', () => {
    expect(moveCorrection(['a', 'b', 'c'], 'b', -1)).toEqual(['b', 'a', 'c'])
    expect(moveCorrection(['a', 'b', 'c'], 'b', 1)).toEqual(['a', 'c', 'b'])
    expect(moveCorrection(['a', 'b', 'c'], 'a', -1)).toEqual(['a', 'b', 'c'])
  })

  it('marks unavailable active corrections', () => {
    const items = activeCorrectionItems(['known', 'missing'], [
      postprocessor('known', 'Known', POSTPROCESSOR_CATEGORY_CORRECTION, 10),
    ])

    expect(items).toMatchObject([
      { id: 'known', available: true },
      { id: 'missing', available: false },
    ])
  })
})

function postprocessor(
  id: string,
  label: string,
  category: string,
  defaultOrder: number,
): PostprocessorInfo {
  return {
    id,
    label,
    category,
    default_order: defaultOrder,
    description: '',
    plugin_id: `${id}-plugin`,
  }
}
