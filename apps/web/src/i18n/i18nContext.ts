import { createContext } from 'react'
import { LOCALE_EN, type Locale, type TranslateFunction, createTranslator } from './i18n'

export interface I18nContextValue {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: TranslateFunction
}

const FALLBACK_I18N_CONTEXT: I18nContextValue = Object.freeze({
  locale: LOCALE_EN,
  setLocale: () => undefined,
  t: createTranslator(LOCALE_EN),
})

export const I18nContext = createContext<I18nContextValue>(FALLBACK_I18N_CONTEXT)
