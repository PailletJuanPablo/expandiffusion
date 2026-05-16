import { RefreshCw } from 'lucide-react'
import type {
  PersistedGeneration,
  PersistedModelLoad,
  PersistedProject,
  PersistentState,
} from '../domain/types'
import { Button } from './ui/button'
import { Skeleton } from './ui/skeleton'

interface PersistenceSectionProps {
  state: PersistentState | undefined
  loading: boolean
  onRefresh: () => void
}

/**
 * Render local model, project and generation history.
 *
 * @param props - Persisted state and refresh callback.
 * @returns Persistence panel section.
 */
export function PersistenceSection({ state, loading, onRefresh }: PersistenceSectionProps) {
  const currentModel = state?.current_model ?? null
  const projects = state?.projects.slice(0, 3) ?? []
  const generations = state?.generations.slice(0, 4) ?? []

  return (
    <section className="panel-section persistence-section">
      <div className="section-heading">
        <span>Local history</span>
        <Button type="button" variant="ghost" size="smallIcon" onClick={onRefresh} title="Refresh history">
          <RefreshCw size={15} className={loading ? 'spin-icon' : ''} />
        </Button>
      </div>
      {loading && !state ? (
        <div className="persistence-list">
          <Skeleton className="persistence-skeleton" />
          <Skeleton className="persistence-skeleton" />
        </div>
      ) : (
        <div className="persistence-list">
          <PersistedModelSummary model={currentModel} />
          <PersistedProjectList projects={projects} />
          <PersistedGenerationList generations={generations} />
        </div>
      )}
    </section>
  )
}

function PersistedModelSummary({ model }: { model: PersistedModelLoad | null }) {
  if (!model) {
    return <div className="persistence-empty">No model load persisted yet.</div>
  }
  return (
    <div className="persistence-card">
      <strong>{model.adapter_label}</strong>
      <span>{model.model_url ?? model.model_id ?? model.local_path ?? model.single_file_path ?? model.adapter_id}</span>
      <span>{model.device} / {model.dtype} / safety {model.safety_checker ? 'on' : 'off'}</span>
    </div>
  )
}

function PersistedProjectList({ projects }: { projects: PersistedProject[] }) {
  if (projects.length === 0) {
    return <div className="persistence-empty">No project activity persisted yet.</div>
  }
  return (
    <div className="persistence-group">
      <span className="persistence-label">Projects</span>
      {projects.map((project) => (
        <div key={project.project_id} className="persistence-card">
          <strong>{project.project_id.slice(0, 10)}</strong>
          <span>{project.generation_count} generations / {project.last_status || 'new'}</span>
          <span>{project.last_prompt || 'No prompt'} / {formatShortDate(project.updated_at)}</span>
        </div>
      ))}
    </div>
  )
}

function PersistedGenerationList({ generations }: { generations: PersistedGeneration[] }) {
  if (generations.length === 0) {
    return null
  }
  return (
    <div className="persistence-group">
      <span className="persistence-label">Generations</span>
      {generations.map((generation) => (
        <div key={generation.job_id} className="persistence-card">
          <strong>{generation.status} / {generation.width} x {generation.height}</strong>
          <span>{generation.prompt || 'No prompt'}</span>
          <span>{generation.adapter_id} / {formatShortDate(generation.updated_at)}</span>
        </div>
      ))}
    </div>
  )
}

function formatShortDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
