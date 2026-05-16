import { useContext } from 'react'
import { I18nContext, type I18nContextValue } from './i18nContext'

/**
 * Read i18n helpers from context.
 *
 * @returns Active locale, setter and translation function.
 */
export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
