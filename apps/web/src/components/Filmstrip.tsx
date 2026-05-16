import { ChevronDown, ChevronUp, Download, History, RotateCcw } from 'lucide-react'
import { EXPORT_FILE_NAME } from '../constants/domain'
import type { DocumentBounds } from '../domain/types'
import { localizeAdapterLabel } from '../i18n/metadata'
import { useI18n } from '../i18n/useI18n'
import { renderDocumentDataUrl } from '../lib/canvasRender'
import { ONBOARDING_TARGET_FILMSTRIP } from '../lib/onboardingTour'
import { downloadDataUrl } from '../lib/projectArchive'
import { useEditorStore } from '../store/editorStore'
import { Button } from './ui/button'

interface FilmstripProps {
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
}

export function Filmstrip({ collapsed, onCollapsedChange }: FilmstripProps) {
  const { t } = useI18n()
  const pendingResults = useEditorStore((state) => state.pendingResults)
  const selectedResultIndex = useEditorStore((state) => state.selectedResultIndex)
  const selectResult = useEditorStore((state) => state.selectResult)
  const setPendingResults = useEditorStore((state) => state.setPendingResults)
  const history = useEditorStore((state) => state.history)
  const documentState = useEditorStore((state) => state.document)
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage)
  const visibleCount = pendingResults.length || history.length

  return (
    <footer
      className={collapsed ? 'filmstrip filmstrip-collapsed' : 'filmstrip'}
      data-tour-id={ONBOARDING_TARGET_FILMSTRIP}
    >
      <div className="filmstrip-actions">
        <Button
          type="button"
          variant="secondary"
          size="compact"
          disabled={!documentState.rasterDataUrl}
          onClick={() => {
            if (documentState.rasterDataUrl) {
              renderDocumentDataUrl(documentState)
                .then((dataUrl) => downloadDataUrl(dataUrl, EXPORT_FILE_NAME))
                .catch((error) =>
                  setErrorMessage(error instanceof Error ? error.message : t('filmstrip.exportFailed')),
                )
            }
          }}
        >
          <Download size={16} />
          PNG
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="compact"
          className="filmstrip-toggle"
          aria-label={collapsed ? t('filmstrip.expandPreviews') : t('filmstrip.collapsePreviews')}
          onClick={() => onCollapsedChange(!collapsed)}
        >
          {collapsed ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          {collapsed ? t('common.show') : t('common.hide')}
        </Button>
      </div>
      {collapsed ? (
        <div className="filmstrip-summary">
          <span>{t('filmstrip.previews')}</span>
          <strong>{visibleCount}</strong>
        </div>
      ) : null}
      {!collapsed ? (
      <div className="filmstrip-content">
        {pendingResults.length > 0 ? (
          <div className="result-strip" aria-label={t('filmstrip.generationPreviews')}>
            <span className="strip-label">{t('filmstrip.previews')}</span>
            {pendingResults.map((result, index) => (
              <button
                type="button"
                key={result}
                className={index === selectedResultIndex ? 'thumb-button thumb-active' : 'thumb-button'}
                onClick={() => selectResult(index)}
              >
                <img src={result} alt={t('filmstrip.resultAlt', { index: index + 1 })} />
              </button>
            ))}
          </div>
        ) : null}
        {pendingResults.length === 0 && history.length > 0 ? (
          <div className="history-strip" aria-label={t('filmstrip.generationHistory')}>
            <div className="strip-label">
              <History size={14} />
              {t('common.history')}
            </div>
            {history.slice(0, 8).map((item) => (
              <HistoryThumb
                key={item.id}
                item={item}
                onOpen={() => setPendingResults(item.images, item.resultBounds ?? null)}
              />
            ))}
          </div>
        ) : null}
        {pendingResults.length === 0 && history.length === 0 ? (
          <div className="empty-strip">{t('filmstrip.empty')}</div>
        ) : null}
      </div>
      ) : null}
    </footer>
  )
}

function HistoryThumb({
  item,
  onOpen,
}: {
  item: {
    acceptedImage: string | null
    images: string[]
    prompt: string
    adapterId: string
    createdAt: string
    resultBounds?: DocumentBounds | null
  }
  onOpen: () => void
}) {
  const { t } = useI18n()
  const image = item.acceptedImage ?? item.images[0]
  if (!image) {
    return null
  }
  return (
    <button
      type="button"
      className="history-card"
      onClick={onOpen}
      title={t('filmstrip.reviewGeneration')}
    >
      <img src={image} alt={item.prompt || t('filmstrip.generationAlt')} />
      <span className="history-card-body">
        <strong>{item.prompt || t('common.noPrompt')}</strong>
        <span>{localizeAdapterLabel(item.adapterId, t)} / {formatHistoryTime(item.createdAt)}</span>
      </span>
      <span className="history-card-action" aria-hidden="true">
        <RotateCcw size={14} />
      </span>
    </button>
  )
}

function formatHistoryTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
}
