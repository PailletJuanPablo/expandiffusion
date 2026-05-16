import { RefreshCw } from 'lucide-react'
import type {
  PersistedGeneration,
  PersistedModelLoad,
  PersistedProject,
  PersistentState,
} from '../domain/types'
import { localizeAdapterLabel, localizeJobStatus } from '../i18n/metadata'
import { useI18n } from '../i18n/useI18n'
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
  const { t } = useI18n()
  const currentModel = state?.current_model ?? null
  const projects = state?.projects.slice(0, 3) ?? []
  const generations = state?.generations.slice(0, 4) ?? []

  return (
    <section className="panel-section persistence-section">
      <div className="section-heading">
        <span>{t('persistence.localHistory')}</span>
        <Button type="button" variant="ghost" size="smallIcon" onClick={onRefresh} title={t('persistence.refreshHistory')}>
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
  const { t } = useI18n()
  if (!model) {
    return <div className="persistence-empty">{t('persistence.noModel')}</div>
  }
  return (
    <div className="persistence-card">
      <strong>{model.adapter_label}</strong>
      <span>
        {model.model_url ??
          model.model_id ??
          model.local_path ??
          model.single_file_path ??
          localizeAdapterLabel(model.adapter_id, t)}
      </span>
      <span>
        {model.device} / {model.dtype} /{' '}
        {t('persistence.safety', {
          state: model.safety_checker ? t('persistence.safetyOn') : t('persistence.safetyOff'),
        })}
      </span>
    </div>
  )
}

function PersistedProjectList({ projects }: { projects: PersistedProject[] }) {
  const { t } = useI18n()
  if (projects.length === 0) {
    return <div className="persistence-empty">{t('persistence.noProjects')}</div>
  }
  return (
    <div className="persistence-group">
      <span className="persistence-label">{t('persistence.projects')}</span>
      {projects.map((project) => (
        <div key={project.project_id} className="persistence-card">
          <strong>{project.project_id.slice(0, 10)}</strong>
          <span>
            {t('persistence.projectSummary', {
              count: project.generation_count,
              status: project.last_status
                ? localizeJobStatus(project.last_status, t)
                : t('persistence.new'),
            })}
          </span>
          <span>{project.last_prompt || t('common.noPrompt')} / {formatShortDate(project.updated_at)}</span>
        </div>
      ))}
    </div>
  )
}

function PersistedGenerationList({ generations }: { generations: PersistedGeneration[] }) {
  const { t } = useI18n()
  if (generations.length === 0) {
    return null
  }
  return (
    <div className="persistence-group">
      <span className="persistence-label">{t('persistence.generations')}</span>
      {generations.map((generation) => (
        <div key={generation.job_id} className="persistence-card">
          <strong>
            {localizeJobStatus(generation.status, t)} / {generation.width} x {generation.height}
          </strong>
          <span>{generation.prompt || t('common.noPrompt')}</span>
          <span>{localizeAdapterLabel(generation.adapter_id, t)} / {formatShortDate(generation.updated_at)}</span>
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
