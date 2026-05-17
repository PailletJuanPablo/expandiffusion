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

  it('translates object selector panel labels', () => {
    const t = createTranslator(LOCALE_ES)

    expect(t('inspector.visibleCanvas')).toBe('Canvas visible')
    expect(t('inspector.objectPromptOnly')).toBe(
      'Escribe un prompt o agrega uno o mas puntos sobre el canvas.',
    )
    expect(t('inspector.objectPointsSelected', { count: 2 })).toBe(
      '2 puntos seleccionados. Puedes agregar mas puntos o procesar.',
    )
    expect(t('inspector.autoProcessClick')).toBe('Procesar clicks automaticamente')
    expect(t('inspector.clearPoints')).toBe('Limpiar puntos')
    expect(t('inspector.useAsMask')).toBe('Usar como mascara')
    expect(t('inspector.eraseObject')).toBe('Borrar')
  })
})
