import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, Plus, X } from 'lucide-react'
import { useState } from 'react'
import type { ControlSchema, GenerationParameters, PostprocessorInfo } from '../domain/types'
import {
  activeCorrectionItems,
  appendCorrection,
  correctionPostprocessors,
  moveCorrection,
  removeCorrection,
} from '../lib/correctionPipeline'
import { isGenerationControlDisabled } from '../lib/controlSchemas'
import { SchemaControl } from './SchemaControl'
import { Button } from './ui/button'

interface CorrectionPipelineSectionProps {
  postprocessors: PostprocessorInfo[]
  controls: ControlSchema[]
  parameters: GenerationParameters
  onParameterChange: (id: string, value: unknown) => void
}

/**
 * Render the active correction pipeline and correction-owned controls.
 *
 * @param props - Runtime correction metadata and generation parameters.
 * @returns Correction pipeline controls.
 */
export function CorrectionPipelineSection({
  postprocessors,
  controls,
  parameters,
  onParameterChange,
}: CorrectionPipelineSectionProps) {
  const [addOpen, setAddOpen] = useState(false)
  const corrections = correctionPostprocessors(postprocessors)
  const activeItems = activeCorrectionItems(parameters.correction_pipeline, corrections)
  const activeIds = new Set(parameters.correction_pipeline)
  const availableItems = corrections.filter((correction) => !activeIds.has(correction.id))

  if (corrections.length === 0 && activeItems.length === 0) {
    return (
      <section className="panel-section panel-section-compact">
        <div className="persistence-empty">No correction plugins loaded.</div>
      </section>
    )
  }

  return (
    <section className="panel-section panel-section-compact correction-pipeline-section">
      {activeItems.length > 0 ? (
        <div className="correction-pipeline-list">
          {activeItems.map((item, index) => (
            <article
              key={`${item.id}-${index}`}
              className={item.available ? 'correction-pipeline-item' : 'correction-pipeline-item correction-pipeline-item-disabled'}
            >
              <div className="correction-pipeline-row">
                <div className="correction-pipeline-main">
                  <strong>{item.label}</strong>
                  <span>{item.description}</span>
                </div>
                <div className="correction-pipeline-actions">
                  <Button
                    type="button"
                    variant="ghost"
                    size="smallIcon"
                    title="Move correction up"
                    aria-label="Move correction up"
                    disabled={index === 0}
                    onClick={() =>
                      onParameterChange(
                        'correction_pipeline',
                        moveCorrection(parameters.correction_pipeline, item.id, -1),
                      )
                    }
                  >
                    <ArrowUp size={14} />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="smallIcon"
                    title="Move correction down"
                    aria-label="Move correction down"
                    disabled={index === activeItems.length - 1}
                    onClick={() =>
                      onParameterChange(
                        'correction_pipeline',
                        moveCorrection(parameters.correction_pipeline, item.id, 1),
                      )
                    }
                  >
                    <ArrowDown size={14} />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="smallIcon"
                    title="Remove correction"
                    aria-label="Remove correction"
                    onClick={() =>
                      onParameterChange(
                        'correction_pipeline',
                        removeCorrection(parameters.correction_pipeline, item.id),
                      )
                    }
                  >
                    <X size={14} />
                  </Button>
                </div>
              </div>
              <CorrectionControls
                controls={controlsForPlugin(controls, item.pluginId)}
                parameters={parameters}
                disabled={!item.available}
                onChange={onParameterChange}
              />
            </article>
          ))}
        </div>
      ) : null}

      {availableItems.length > 0 ? (
        <div className="correction-add-group">
          <button
            type="button"
            className="generation-section-trigger correction-section-trigger"
            aria-expanded={addOpen}
            onClick={() => setAddOpen((current) => !current)}
          >
            <span className="generation-section-title">
              <Plus size={16} />
              <span>Add correction</span>
            </span>
            <span className="generation-section-meta">
              <span>{availableItems.length}</span>
              {addOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
            </span>
          </button>
          {addOpen ? (
            <div className="correction-add-list">
              {availableItems.map((item) => (
                <article key={item.id} className="correction-add-item">
                  <div className="correction-pipeline-main">
                    <strong>{item.label}</strong>
                    <span>{item.description}</span>
                  </div>
                  <Button
                    type="button"
                    variant="secondary"
                    size="smallIcon"
                    className="correction-add-button"
                    title={`Add ${item.label}`}
                    aria-label={`Add ${item.label}`}
                    onClick={() =>
                      onParameterChange(
                        'correction_pipeline',
                        appendCorrection(parameters.correction_pipeline, item.id),
                      )
                    }
                  >
                    <Plus size={14} />
                  </Button>
                </article>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

function controlsForPlugin(controls: ControlSchema[], pluginId: string | null): ControlSchema[] {
  if (pluginId === null) {
    return []
  }
  return controls.filter((control) => control.plugin_id === pluginId)
}

function CorrectionControls({
  controls,
  parameters,
  disabled,
  onChange,
}: {
  controls: ControlSchema[]
  parameters: GenerationParameters
  disabled: boolean
  onChange: (id: string, value: unknown) => void
}) {
  if (controls.length === 0) {
    return null
  }
  return (
    <div className="correction-control-list">
      {controls.map((control) => (
        <SchemaControl
          key={control.id}
          control={control}
          value={parameters[control.id]}
          disabled={disabled || isGenerationControlDisabled(control, parameters)}
          onChange={onChange}
        />
      ))}
    </div>
  )
}
