import { useCallback, useMemo, useState, type ReactNode } from 'react'
import {
  LOCALE_EN,
  LOCALE_STORAGE_KEY,
  type Locale,
  detectPreferredLocale,
  createTranslator,
} from './i18n'
import { I18nContext } from './i18nContext'

/**
 * Provide the active locale and translator to the React tree.
 *
 * @param props - Child nodes rendered below the provider.
 * @returns I18n context provider.
 */
export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readInitialLocale)
  const setLocale = useCallback((nextLocale: Locale) => {
    setLocaleState(nextLocale)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, nextLocale)
    }
  }, [])
  const value = useMemo(
    () => ({
      locale,
      setLocale,
      t: createTranslator(locale),
    }),
    [locale, setLocale],
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

function readInitialLocale(): Locale {
  if (typeof window === 'undefined') {
    return LOCALE_EN
  }
  return detectPreferredLocale(
    window.localStorage.getItem(LOCALE_STORAGE_KEY),
    window.navigator.languages,
  )
}
