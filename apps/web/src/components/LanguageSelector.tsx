import { LOCALE_EN, LOCALE_ES, normalizeLocale } from '../i18n/i18n'
import { useI18n } from '../i18n/useI18n'

/**
 * Render the topbar locale switcher.
 *
 * @returns Locale selector.
 */
export function LanguageSelector() {
  const { locale, setLocale, t } = useI18n()

  return (
    <label className="language-selector">
      <span>{t('app.language')}</span>
      <select
        value={locale}
        aria-label={t('app.language')}
        onChange={(event) => {
          const nextLocale = normalizeLocale(event.target.value)
          if (nextLocale) {
            setLocale(nextLocale)
          }
        }}
      >
        <option value={LOCALE_EN}>{t('app.languageEnglish')}</option>
        <option value={LOCALE_ES}>{t('app.languageSpanish')}</option>
      </select>
    </label>
  )
}
