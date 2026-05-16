import { describe, expect, it } from 'vitest'
import { FILL_OPTIONS } from '../constants/domain'
import { PREPROCESSOR_DETAILS, preprocessorDetailsFor } from './preprocessorDetails'

describe('preprocessorDetails', () => {
  it('covers every built-in fill option', () => {
    for (const option of FILL_OPTIONS) {
      expect(PREPROCESSOR_DETAILS[option.id]?.title).toBeTruthy()
      expect(PREPROCESSOR_DETAILS[option.id]?.bestFor).toBeTruthy()
      expect(PREPROCESSOR_DETAILS[option.id]?.caution).toBeTruthy()
    }
  })

  it('returns null for custom adapter fill modes', () => {
    expect(preprocessorDetailsFor('custom_fill')).toBeNull()
  })
})
