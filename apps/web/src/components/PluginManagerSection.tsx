import { ChevronDown, ChevronRight, Power, PowerOff, SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'
import type { ControlSchema, GenerationParameters, PluginInfo, PostprocessorInfo } from '../domain/types'
import { correctionPostprocessors } from '../lib/correctionPipeline'
import { isGenerationControlDisabled } from '../lib/controlSchemas'
import { SchemaControl } from './SchemaControl'
import { Button } from './ui/button'

interface PluginManagerSectionProps {
  plugins: PluginInfo[]
  controls: ControlSchema[]
  postprocessors: PostprocessorInfo[]
  parameters: GenerationParameters
  pendingPluginId: string | null
  onToggle: (plugin: PluginInfo) => void
  onParameterChange: (id: string, value: unknown) => void
}

export function PluginManagerSection({
  plugins,
  controls,
  postprocessors,
  parameters,
  pendingPluginId,
  onToggle,
  onParameterChange,
}: PluginManagerSectionProps) {
  const [expandedPluginId, setExpandedPluginId] = useState<string | null>(null)
  const correctionPluginIds = new Set(
    correctionPostprocessors(postprocessors)
      .map((postprocessor) => postprocessor.plugin_id)
      .filter((pluginId) => pluginId !== null),
  )
  if (plugins.length === 0) {
    return (
      <section className="panel-section panel-section-compact">
        <div className="persistence-empty">No plugins found.</div>
      </section>
    )
  }

  return (
    <section className="panel-section panel-section-compact">
      <div className="plugin-list">
        {plugins.map((plugin) => {
          const pluginControls = controls.filter(
            (control) =>
              control.plugin_id === plugin.id && !correctionPluginIds.has(control.plugin_id),
          )
          const expanded = expandedPluginId === plugin.id
          return (
            <article key={plugin.id} className={pluginCardClass(plugin)}>
              <div className="plugin-card-header">
                <div className="plugin-card-main">
                  <strong>{plugin.label}</strong>
                  <span>{plugin.description || plugin.id}</span>
                  {plugin.error ? <span className="plugin-error">{plugin.error}</span> : null}
                </div>
                <div className="plugin-card-actions">
                  <span className={pluginStateClass(plugin)}>
                    <span />
                    {pluginStatus(plugin)}
                  </span>
                  <button
                    type="button"
                    className={plugin.enabled ? 'plugin-switch plugin-switch-on' : 'plugin-switch'}
                    disabled={pendingPluginId === plugin.id}
                    aria-pressed={plugin.enabled}
                    aria-label={plugin.enabled ? `Disable ${plugin.label}` : `Enable ${plugin.label}`}
                    title={plugin.enabled ? 'Disable plugin' : 'Enable plugin'}
                    onClick={() => onToggle(plugin)}
                  >
                    {plugin.enabled ? <PowerOff size={14} /> : <Power size={14} />}
                  </button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="smallIcon"
                    disabled={pluginControls.length === 0}
                    aria-label={expanded ? `Hide ${plugin.label} controls` : `Show ${plugin.label} controls`}
                    title={pluginControls.length === 0 ? 'No controls' : 'Plugin controls'}
                    onClick={() =>
                      setExpandedPluginId((current) => current === plugin.id ? null : plugin.id)
                    }
                  >
                    {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                  </Button>
                </div>
              </div>
              {expanded ? (
                <div className="plugin-control-panel">
                  <div className="plugin-control-heading">
                    <SlidersHorizontal size={15} />
                    <span>Controls</span>
                  </div>
                  <PluginControls
                    controls={pluginControls}
                    parameters={parameters}
                    disabled={!plugin.enabled || !plugin.loaded}
                    onChange={onParameterChange}
                  />
                </div>
              ) : null}
            </article>
          )
        })}
      </div>
    </section>
  )
}

function pluginStatus(plugin: PluginInfo): string {
  if (plugin.enabled && plugin.loaded) {
    return 'Enabled'
  }
  if (plugin.enabled) {
    return 'Load error'
  }
  return 'Disabled'
}

function pluginCardClass(plugin: PluginInfo): string {
  if (plugin.enabled && plugin.loaded) {
    return 'plugin-card plugin-card-enabled'
  }
  if (plugin.enabled) {
    return 'plugin-card plugin-card-error'
  }
  return 'plugin-card'
}

function pluginStateClass(plugin: PluginInfo): string {
  if (plugin.enabled && plugin.loaded) {
    return 'plugin-state plugin-state-enabled'
  }
  if (plugin.enabled) {
    return 'plugin-state plugin-state-error'
  }
  return 'plugin-state'
}

function PluginControls({
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
    <div className="plugin-control-list">
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
