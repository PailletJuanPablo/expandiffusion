import { useEditorStore } from '../store/editorStore'

export function StatusBar() {
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
      <span>{selectedAdapterId}</span>
      <span>{currentJob ? `${currentJob.status}: ${currentJob.message}` : 'No active job'}</span>
      {generationNote ? <span>{generationNote}</span> : null}
      {errorMessage ? <span className="status-error">{errorMessage}</span> : null}
    </div>
  )
}
