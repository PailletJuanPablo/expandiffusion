import { describe, expect, it } from 'vitest'
import {
  LOCALE_EN,
  LOCALE_ES,
  createTranslator,
  detectPreferredLocale,
  interpolateMessage,
  normalizeLocale,
} from './i18n'

describe('i18n', () => {
  it('normalizes supported locale tags', () => {
    expect(normalizeLocale('es-AR')).toBe(LOCALE_ES)
    expect(normalizeLocale('en-US')).toBe(LOCALE_EN)
    expect(normalizeLocale('fr-FR')).toBeNull()
  })

  it('prefers a stored locale before browser languages', () => {
    expect(detectPreferredLocale('es', ['en-US'])).toBe(LOCALE_ES)
    expect(detectPreferredLocale(null, ['es-AR'])).toBe(LOCALE_ES)
    expect(detectPreferredLocale('fr', ['fr-FR'])).toBe(LOCALE_EN)
  })

  it('interpolates message parameters', () => {
    expect(interpolateMessage('{count} / {total} enabled', { count: 2, total: 5 })).toBe(
      '2 / 5 enabled',
    )
  })

  it('translates with locale fallback and key fallback', () => {
    const t = createTranslator(LOCALE_ES)

    expect(t('toolbar.select')).toBe('Seleccionar')
    expect(t('test.onlyInEnglish')).toBe('English fallback')
    expect(t('missing.key')).toBe('missing.key')
  })
})
