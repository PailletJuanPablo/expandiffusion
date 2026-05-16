import { describe, expect, it } from 'vitest'
import { LOCALE_ES, createTranslator } from '../i18n/i18n'
import { controlHelpFor, controlOptionDetailsFor } from './controlHelp'

describe('controlHelp', () => {
  it('localizes control descriptions and rich option details', () => {
    const t = createTranslator(LOCALE_ES)
    const details = controlOptionDetailsFor('outpaint_strategy', t)

    expect(controlHelpFor('prompt', t)).toBe('Describe que debe aparecer en el area generada.')
    expect(details?.hf_space_fill.title).toBe('Relleno HF Space')
    expect(details?.hf_space_fill.badge).toBe('Recomendado')
    expect(details?.directional.description).toContain('Expansion de un lado')
  })
})
