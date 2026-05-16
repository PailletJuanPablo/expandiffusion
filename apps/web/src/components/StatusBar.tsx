import { useI18n } from '../i18n/useI18n'
import {
  localizeAdapterLabel,
  localizeErrorMessage,
  localizeJobMessage,
  localizeJobStatus,
} from '../i18n/metadata'
import { useEditorStore } from '../store/editorStore'

export function StatusBar() {
  const { t } = useI18n()
  const documentState = useEditorStore((state) => state.document)
  const viewport = useEditorStore((state) => state.viewport)
  const currentJob = useEditorStore((state) => state.currentJob)
  const generationNote = useEditorStore((state) => state.generationNote)
  const selectedAdapterId = useEditorStore((state) => state.selectedAdapterId)
  const errorMessage = useEditorStore((state) => state.errorMessage)

  return (
    <div className="status-bar">
      <span>{Math.round(viewport.zoom * 100)}%</span>
      <span>{documentState.width} x {documentState.height}</span>
      <span>{localizeAdapterLabel(selectedAdapterId, t)}</span>
      <span>
        {currentJob
          ? `${localizeJobStatus(currentJob.status, t)}: ${localizeJobMessage(currentJob.message, t)}`
          : t('status.noActiveJob')}
      </span>
      {generationNote ? <span>{localizeJobMessage(generationNote, t)}</span> : null}
      {errorMessage ? <span className="status-error">{localizeErrorMessage(errorMessage, t)}</span> : null}
    </div>
  )
}
