import { EN_MESSAGES, ES_MESSAGES } from './messages'

export const LOCALE_EN = 'en'
export const LOCALE_ES = 'es'
export const LOCALE_STORAGE_KEY = 'expandiffusion.locale'

export type Locale = typeof LOCALE_EN | typeof LOCALE_ES
export type TranslationValue = string | number | boolean | null | undefined
export type TranslationParams = Record<string, TranslationValue>
export type LocaleMessages = Record<string, string>
export type TranslateFunction = (
  key: string,
  params?: TranslationParams,
  fallback?: string,
) => string

const SUPPORTED_LOCALES: Locale[] = [LOCALE_EN, LOCALE_ES]
const MESSAGES_BY_LOCALE: Record<Locale, LocaleMessages> = Object.freeze({
  [LOCALE_EN]: EN_MESSAGES,
  [LOCALE_ES]: ES_MESSAGES,
})

/**
 * Normalize a browser or stored locale tag into the supported app locale.
 *
 * @param value - Locale tag such as `en-US` or `es-AR`.
 * @returns Supported locale or null when the tag is unsupported.
 */
export function normalizeLocale(value: string | null | undefined): Locale | null {
  if (!value) {
    return null
  }
  const code = value.trim().toLowerCase().split('-')[0]
  return SUPPORTED_LOCALES.find((locale) => locale === code) ?? null
}

/**
 * Resolve the initial locale from persisted preference and browser languages.
 *
 * @param storedLocale - Locale saved in localStorage.
 * @param browserLanguages - Browser language preference list.
 * @returns Supported app locale.
 */
export function detectPreferredLocale(
  storedLocale: string | null | undefined,
  browserLanguages: readonly string[],
): Locale {
  const normalizedStoredLocale = normalizeLocale(storedLocale)
  if (normalizedStoredLocale) {
    return normalizedStoredLocale
  }
  for (const language of browserLanguages) {
    const normalizedLanguage = normalizeLocale(language)
    if (normalizedLanguage) {
      return normalizedLanguage
    }
  }
  return LOCALE_EN
}

/**
 * Replace `{name}` placeholders in a message with display values.
 *
 * @param message - Message template.
 * @param params - Replacement values.
 * @returns Interpolated message.
 */
export function interpolateMessage(message: string, params: TranslationParams = {}): string {
  return message.replace(/\{([a-zA-Z0-9_]+)\}/g, (placeholder, key: string) => {
    const value = params[key]
    return value === null || value === undefined ? placeholder : String(value)
  })
}

/**
 * Create a locale-bound translator with English fallback.
 *
 * @param locale - Active locale.
 * @returns Translation function.
 */
export function createTranslator(locale: Locale): TranslateFunction {
  return (key, params = {}, fallback) => {
    const localizedMessage = MESSAGES_BY_LOCALE[locale][key]
    const englishMessage = EN_MESSAGES[key]
    const message = localizedMessage ?? englishMessage ?? fallback ?? key
    return interpolateMessage(message, params)
  }
}

/**
 * Return whether a translation key exists in the current locale or English fallback.
 *
 * @param key - Translation key.
 * @param locale - Active locale.
 * @returns True when a message is available.
 */
export function hasTranslation(key: string, locale: Locale): boolean {
  return MESSAGES_BY_LOCALE[locale][key] !== undefined || EN_MESSAGES[key] !== undefined
}
