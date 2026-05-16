import {
  ChevronDown,
  ChevronRight,
  Frame,
  SlidersHorizontal,
  WandSparkles,
  type LucideIcon,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import {
  CONTROL_SECTION_ADVANCED,
  CONTROL_SECTION_BASIC,
  CONTROL_SECTION_EXTENSIONS,
  type GenerationMode,
} from '../constants/domain'
import type { ControlSchema, GenerationParameters } from '../domain/types'
import {
  controlsForGenerationMode,
  controlsForSection,
  isGenerationControlDisabled,
} from '../lib/controlSchemas'
import { useI18n } from '../i18n/useI18n'
import { SchemaControl } from './SchemaControl'

interface GenerationControlsProps {
  controls: ControlSchema[]
  loadControls: ControlSchema[]
  generationMode: GenerationMode
  parameters: GenerationParameters
  loraText: string
  textualInversionText: string
  hiddenControlIds?: Set<string>
  onParameterChange: (id: string, value: unknown) => void
  onLoraTextChange: (value: string) => void
  onTextualInversionTextChange: (value: string) => void
}

/**
 * Render adapter-provided generation and extension controls.
 *
 * @param props - Control schemas and current form state.
 * @returns Inspector disclosure groups.
 */
export function GenerationControls({
  controls,
  loadControls,
  generationMode,
  parameters,
  loraText,
  textualInversionText,
  hiddenControlIds,
  onParameterChange,
  onLoraTextChange,
  onTextualInversionTextChange,
}: GenerationControlsProps) {
  const { t } = useI18n()
  const [openSections, setOpenSections] = useState({
    generation: true,
    advanced: false,
    personalization: false,
  })
  const adapterControls = controlsForGenerationMode(
    controls.filter(
      (control) => !control.plugin_id && !hiddenControlIds?.has(control.id),
    ),
    generationMode,
  )
  const basicControls = controlsForSection(adapterControls, CONTROL_SECTION_BASIC)
  const advancedControls = controlsForSection(adapterControls, CONTROL_SECTION_ADVANCED)
  const extensionControls = controlsForSection(loadControls, CONTROL_SECTION_EXTENSIONS)
    .filter((control) => control.id !== 'loras')
  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections((current) => ({
      ...current,
      [section]: !current[section],
    }))
  }

  return (
    <div className="generation-panel-stack">
      <CollapsibleSection
        title={t('generationControls.generation')}
        icon={WandSparkles}
        count={basicControls.length}
        open={openSections.generation}
        onToggle={() => toggleSection('generation')}
      >
        <SchemaControlList
          controls={basicControls}
          values={parameters}
          onChange={onParameterChange}
        />
      </CollapsibleSection>

      <CollapsibleSection
        title={t('generationControls.advanced')}
        icon={SlidersHorizontal}
        count={advancedControls.length}
        open={openSections.advanced}
        onToggle={() => toggleSection('advanced')}
      >
        <SchemaControlList
          controls={advancedControls}
          values={parameters}
          onChange={onParameterChange}
        />
      </CollapsibleSection>

      {extensionControls.length > 0 ? (
        <CollapsibleSection
          title={t('generationControls.personalization')}
          icon={Frame}
          count={extensionControls.length}
          open={openSections.personalization}
          onToggle={() => toggleSection('personalization')}
        >
          {extensionControls.map((control) => (
            <SchemaControl
              key={control.id}
              control={control}
              value={extensionValue(control.id, loraText, textualInversionText)}
              onChange={(id, value) =>
                updateExtensionValue(
                  id,
                  value,
                  onLoraTextChange,
                  onTextualInversionTextChange,
                )
              }
            />
          ))}
        </CollapsibleSection>
      ) : null}
    </div>
  )
}

function CollapsibleSection({
  title,
  icon: Icon,
  count,
  open,
  onToggle,
  children,
}: {
  title: string
  icon: LucideIcon
  count: number
  open: boolean
  onToggle: () => void
  children: ReactNode
}) {
  return (
    <section className={open ? 'panel-section generation-control-section generation-control-section-open' : 'panel-section generation-control-section'}>
      <button
        type="button"
        className="generation-section-trigger"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span className="generation-section-title">
          <Icon size={16} />
          <span>{title}</span>
        </span>
        <span className="generation-section-meta">
          <span>{count}</span>
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </span>
      </button>
      {open ? <div className="generation-section-body">{children}</div> : null}
    </section>
  )
}

function SchemaControlList({
  controls,
  values,
  onChange,
}: {
  controls: ControlSchema[]
  values: GenerationParameters
  onChange: (id: string, value: unknown) => void
}) {
  return (
    <div className="schema-control-list">
      {controls.map((control) => (
        <SchemaControl
          key={control.id}
          control={control}
          value={values[control.id]}
          disabled={isGenerationControlDisabled(control, values)}
          onChange={onChange}
        />
      ))}
      {controls.length === 0 ? <EmptyControlGroup /> : null}
    </div>
  )
}

function EmptyControlGroup() {
  const { t } = useI18n()
  return <div className="empty-state">{t('generationControls.noControls')}</div>
}

function extensionValue(
  id: string,
  loraText: string,
  textualInversionText: string,
): string {
  if (id === 'textual_inversions') {
    return textualInversionText
  }
  return id === 'loras' ? loraText : ''
}

function updateExtensionValue(
  id: string,
  value: unknown,
  onLoraTextChange: (value: string) => void,
  onTextualInversionTextChange: (value: string) => void,
): void {
  const text = typeof value === 'string' ? value : ''
  if (id === 'textual_inversions') {
    onTextualInversionTextChange(text)
    return
  }
  if (id === 'loras') {
    onLoraTextChange(text)
  }
}
