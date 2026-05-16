import { ChevronDown, ChevronRight, Power, PowerOff, Square, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  CONTROL_SECTION_EXTENSIONS,
  CONTROL_SECTION_RUNTIME,
  MODEL_SOURCE_FIELD_LOCAL_PATH,
  MODEL_SOURCE_FIELD_MODEL_ID,
  MODEL_SOURCE_FIELD_MODEL_URL,
  MODEL_SOURCE_FIELD_SINGLE_FILE_PATH,
} from '../constants/domain'
import type {
  AdapterInfo,
  ControlSchema,
  GenerationParameters,
  ModelInfo,
  ModelLoadProgress,
  PluginInfo,
  PostprocessorInfo,
  RuntimeInfo,
} from '../domain/types'
import { controlsForSection } from '../lib/controlSchemas'
import {
  getActiveModelSource,
  getModelSources,
  getModelSourceValue,
  type ModelSourceValues,
} from '../lib/modelSources'
import {
  getModelSetupAdapterGroups,
  getModelSetupDetails,
} from '../lib/modelSetupExperience'
import { ONBOARDING_TARGET_SETUP_DIALOG } from '../lib/onboardingTour'
import { useI18n } from '../i18n/useI18n'
import { localizeAdapterLabel, localizeJobMessage } from '../i18n/metadata'
import { PluginManagerSection } from './PluginManagerSection'
import { SchemaControl } from './SchemaControl'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Progress } from './ui/progress'

interface ModelSetupDialogProps {
  adapters: AdapterInfo[]
  selectedAdapter: AdapterInfo | undefined
  runtime: RuntimeInfo | undefined
  selectedAdapterId: string
  modelSource: string
  sourceValues: ModelSourceValues
  device: string
  dtype: string
  controlnetModelId: string
  safetyChecker: boolean
  loraText: string
  textualInversionText: string
  activeModel: ModelInfo | undefined
  loading: boolean
  loadProgress: ModelLoadProgress | null
  loadDisabled: boolean
  unloading: boolean
  cancelPending: boolean
  plugins: PluginInfo[]
  pluginControls: ControlSchema[]
  pluginPostprocessors: PostprocessorInfo[]
  parameters: GenerationParameters
  pendingPluginId: string | null
  onClose: () => void
  onAdapterChange: (adapterId: string) => void
  onModelSourceChange: (value: string) => void
  onModelIdChange: (value: string) => void
  onLocalPathChange: (value: string) => void
  onSingleFilePathChange: (value: string) => void
  onModelUrlChange: (value: string) => void
  onControlnetModelIdChange: (value: string) => void
  onDeviceChange: (value: string) => void
  onDtypeChange: (value: string) => void
  onSafetyCheckerChange: (value: boolean) => void
  onLoraTextChange: (value: string) => void
  onTextualInversionTextChange: (value: string) => void
  onLoad: () => void
  onUnload: () => void
  onCancelLoad: () => void
  onPluginToggle: (plugin: PluginInfo) => void
  onParameterChange: (id: string, value: unknown) => void
}

/**
 * Render the adapter and model loading dialog.
 *
 * @param props - Runtime metadata, selected adapter and form callbacks.
 * @returns Model setup dialog.
 */
export function ModelSetupDialog({
  adapters,
  selectedAdapter,
  runtime,
  selectedAdapterId,
  modelSource,
  sourceValues,
  device,
  dtype,
  controlnetModelId,
  safetyChecker,
  loraText,
  textualInversionText,
  activeModel,
  loading,
  loadProgress,
  loadDisabled,
  unloading,
  cancelPending,
  plugins,
  pluginControls,
  pluginPostprocessors,
  parameters,
  pendingPluginId,
  onClose,
  onAdapterChange,
  onModelSourceChange,
  onModelIdChange,
  onLocalPathChange,
  onSingleFilePathChange,
  onModelUrlChange,
  onControlnetModelIdChange,
  onDeviceChange,
  onDtypeChange,
  onSafetyCheckerChange,
  onLoraTextChange,
  onTextualInversionTextChange,
  onLoad,
  onUnload,
  onCancelLoad,
  onPluginToggle,
  onParameterChange,
}: ModelSetupDialogProps) {
  const { t } = useI18n()
  const [showExperimentalAdapters, setShowExperimentalAdapters] = useState(false)
  const [activeTab, setActiveTab] = useState<'model' | 'plugins'>('model')
  const adapterGroups = useMemo(
    () => getModelSetupAdapterGroups(adapters),
    [adapters],
  )
  const selectedMoreOption = adapterGroups.experimental.some(
    (adapter) => adapter.id === selectedAdapterId,
  )
  const showMoreOptions = showExperimentalAdapters || selectedMoreOption
  const modelSources = getModelSources(selectedAdapter, t)
  const activeSource = getActiveModelSource(selectedAdapter, modelSource, t)
  const runtimeControls = controlsForSection(
    selectedAdapter?.load_controls ?? [],
    CONTROL_SECTION_RUNTIME,
  ).filter((control) => control.id !== 'controlnet_model_id' || control.options.length > 1)
  const extensionControls = controlsForSection(
    selectedAdapter?.load_controls ?? [],
    CONTROL_SECTION_EXTENSIONS,
  ).filter((control) => control.id !== 'loras')

  return (
    <div
      className="setup-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={t('modelSetup.title')}
    >
      <div className="setup-dialog" data-tour-id={ONBOARDING_TARGET_SETUP_DIALOG}>
        <div className="setup-header">
          <div>
            <h2>{t('modelSetup.title')}</h2>
            <p>{runtime?.note ?? t('modelSetup.waitingRuntime')}</p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="smallIcon"
            className="setup-close-button"
            aria-label={t('common.close')}
            title={t('common.close')}
            onClick={onClose}
          >
            <X size={16} />
          </Button>
        </div>
        <div className="setup-tabs" role="tablist" aria-label={t('modelSetup.sections')}>
          <button
            type="button"
            className={activeTab === 'model' ? 'setup-tab setup-tab-active' : 'setup-tab'}
            role="tab"
            aria-selected={activeTab === 'model'}
            onClick={() => setActiveTab('model')}
          >
            {t('modelSetup.model')}
          </button>
          <button
            type="button"
            className={activeTab === 'plugins' ? 'setup-tab setup-tab-active' : 'setup-tab'}
            role="tab"
            aria-selected={activeTab === 'plugins'}
            onClick={() => setActiveTab('plugins')}
          >
            {t('modelSetup.plugins')}
            <span>{plugins.filter((plugin) => plugin.enabled).length}/{plugins.length}</span>
          </button>
        </div>
        {activeTab === 'model' ? (
          <>
            <div className="setup-main" role="tabpanel" aria-label={t('modelSetup.model')}>
              <div className="setup-grid">
                <section className="setup-section">
              <div className="section-heading">
                <span>{t('modelSetup.chooseModelType')}</span>
              </div>
              <p className="setup-helper-text">
                {t('modelSetup.helper')}
              </p>
              <AdapterChoiceGroup
                adapters={adapterGroups.primary}
                selectedAdapterId={selectedAdapterId}
                loading={loading}
                onAdapterChange={onAdapterChange}
              />
              {adapterGroups.experimental.length > 0 ? (
                <div className="experimental-adapter-block">
                  <button
                    type="button"
                    className="experimental-adapter-toggle"
                    aria-expanded={showMoreOptions}
                    onClick={() =>
                      setShowExperimentalAdapters((current) => !current)
                    }
                  >
                    {showMoreOptions ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                    <span>{t('modelSetup.moreOptions')}</span>
                    <small>{t('modelSetup.moreOptionsHint')}</small>
                  </button>
                  {showMoreOptions ? (
                    <AdapterChoiceGroup
                      adapters={adapterGroups.experimental}
                      selectedAdapterId={selectedAdapterId}
                      loading={loading}
                      onAdapterChange={onAdapterChange}
                    />
                  ) : null}
                </div>
              ) : null}
              <label className="field-label">
                {t('modelSetup.modelSource')}
                <select
                  value={activeSource.id}
                  disabled={loading}
                  onChange={(event) => onModelSourceChange(event.target.value)}
                >
                  {modelSources.map((source) => (
                    <option key={source.id} value={source.id}>
                      {source.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field-label">
                {activeSource.label}
                <Input
                  value={getModelSourceValue(activeSource, sourceValues)}
                  disabled={loading}
                  placeholder={activeSource.placeholder ?? undefined}
                  onChange={(event) =>
                    updateModelSourceValue(
                      activeSource.request_field,
                      event.target.value,
                      onModelIdChange,
                      onLocalPathChange,
                      onSingleFilePathChange,
                      onModelUrlChange,
                    )
                  }
                />
              </label>
                </section>
                <section className="setup-section">
              <div className="section-heading">
                <span>{t('modelSetup.runtime')}</span>
              </div>
              {runtimeControls.map((control) => {
                const renderedControl = runtimeControl(control, runtime, t)
                return (
                  <SchemaControl
                    key={control.id}
                    control={renderedControl}
                    value={runtimeControlValue(
                      renderedControl,
                      device,
                      dtype,
                      controlnetModelId,
                      safetyChecker,
                    )}
                    onChange={(id, value) =>
                      updateRuntimeControl(
                        id,
                        value,
                        onDeviceChange,
                        onDtypeChange,
                        onControlnetModelIdChange,
                        onSafetyCheckerChange,
                      )
                    }
                  />
                )
              })}
              <RuntimeDetails runtime={runtime} />
                </section>
                {extensionControls.length > 0 ? (
                  <section className="setup-section setup-section-wide">
                <div className="section-heading">
                  <span>{t('modelSetup.loadTimeExtensions')}</span>
                </div>
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
                  </section>
                ) : null}
              </div>
              <ModelLoadedStatus activeModel={activeModel} />
              {loading ? <ModelLoadProgressBlock progress={loadProgress} /> : null}
            </div>
            <div className="setup-actions">
              <Button
                type="button"
                variant="primary"
                size="large"
                disabled={loading || loadDisabled}
                onClick={onLoad}
              >
                <Power size={16} />
                {loadButtonLabel(loading, activeModel, selectedAdapterId, t)}
              </Button>
              {loading ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="large"
                  disabled={cancelPending}
                  onClick={onCancelLoad}
                >
                  <Square size={15} />
                  {cancelPending ? t('modelSetup.cancelling') : t('modelSetup.cancelLoad')}
                </Button>
              ) : (
                <Button
                  type="button"
                  variant="secondary"
                  size="large"
                  disabled={!activeModel?.loaded || unloading}
                  onClick={onUnload}
                >
                  <PowerOff size={16} />
                  {unloading ? t('modelSetup.unloading') : t('modelSetup.unloadModel')}
                </Button>
              )}
            </div>
          </>
        ) : (
          <div
            className="setup-main setup-plugin-pane"
            role="tabpanel"
            aria-label={t('modelSetup.plugins')}
          >
            <PluginManagerSection
              plugins={plugins}
              controls={pluginControls}
              postprocessors={pluginPostprocessors}
              parameters={parameters}
              pendingPluginId={pendingPluginId}
              onToggle={onPluginToggle}
              onParameterChange={onParameterChange}
            />
          </div>
        )}
      </div>
    </div>
  )
}

function AdapterChoiceGroup({
  adapters,
  selectedAdapterId,
  loading,
  onAdapterChange,
}: {
  adapters: AdapterInfo[]
  selectedAdapterId: string
  loading: boolean
  onAdapterChange: (adapterId: string) => void
}) {
  const { t } = useI18n()
  if (adapters.length === 0) {
    return <div className="adapter-choice-empty">{t('modelSetup.noProfiles')}</div>
  }
  return (
    <div className="adapter-choice-list">
      {adapters.map((adapter) => (
        <AdapterChoice
          key={adapter.id}
          adapter={adapter}
          selected={adapter.id === selectedAdapterId}
          loading={loading}
          onSelect={() => onAdapterChange(adapter.id)}
        />
      ))}
    </div>
  )
}

function AdapterChoice({
  adapter,
  selected,
  loading,
  onSelect,
}: {
  adapter: AdapterInfo
  selected: boolean
  loading: boolean
  onSelect: () => void
}) {
  const { t } = useI18n()
  const details = getModelSetupDetails(adapter, t)
  return (
    <button
      type="button"
      className={selected ? 'adapter-choice adapter-choice-selected' : 'adapter-choice'}
      aria-pressed={selected}
      disabled={loading}
      onClick={onSelect}
    >
      <span className="adapter-choice-header">
        <strong>{adapter.label}</strong>
        {selected ? <span>{t('common.selected')}</span> : null}
      </span>
      <span className="adapter-choice-summary">{details.summary}</span>
      <span className="adapter-choice-best-for">{details.bestFor}</span>
      <span className="adapter-choice-limits">
        <strong>{t('modelSetup.limits')}</strong>
        <span>{details.limitations.join(' ')}</span>
      </span>
    </button>
  )
}

function ModelLoadedStatus({ activeModel }: { activeModel: ModelInfo | undefined }) {
  const { t } = useI18n()
  if (!activeModel?.loaded) {
    return <div className="model-load-status">{t('modelSetup.noModelLoaded')}</div>
  }
  return (
    <div className="model-load-status">
      <span>{t('modelSetup.loadedModel')}</span>
      <strong>{activeModel.adapter_label}</strong>
      <span>
        {activeModel.model_url ??
          activeModel.model_id ??
          activeModel.local_path ??
          activeModel.single_file_path ??
          localizeAdapterLabel(activeModel.adapter_id, t)}
      </span>
    </div>
  )
}

function loadButtonLabel(
  loading: boolean,
  activeModel: ModelInfo | undefined,
  selectedAdapterId: string,
  t: (key: string) => string,
): string {
  if (loading) {
    return t('modelSetup.loadingModel')
  }
  if (!activeModel?.loaded) {
    return t('modelSetup.loadModel')
  }
  return activeModel.adapter_id === selectedAdapterId
    ? t('modelSetup.reloadModel')
    : t('modelSetup.switchModel')
}

function ModelLoadProgressBlock({ progress }: { progress: ModelLoadProgress | null }) {
  const { t } = useI18n()
  const percent = Math.round((progress?.progress ?? 0) * 100)
  const files = progress?.files_total
    ? t('modelSetup.files', {
        done: progress.files_done ?? 0,
        total: progress.files_total,
      })
    : null
  const fileBytes = progress?.file_bytes_total
    ? `${formatBytes(progress.file_bytes_done ?? 0)} / ${formatBytes(progress.file_bytes_total)}`
    : progress?.file_bytes_done
      ? formatBytes(progress.file_bytes_done)
      : null
  const totalBytes = progress?.bytes_total
    ? t('modelSetup.totalBytes', {
        done: formatBytes(progress.bytes_done ?? 0),
        total: formatBytes(progress.bytes_total),
      })
    : null
  return (
    <div className="job-block">
      <div className="job-row">
        <span>
          {progress?.message
            ? localizeJobMessage(progress.message, t)
            : t('modelSetup.startingLoad')}
        </span>
        <span>{percent}%</span>
      </div>
      <Progress value={percent} />
      {files ? <span>{files}</span> : null}
      {progress?.file_name && fileBytes ? <span>{progress.file_name}: {fileBytes}</span> : null}
      {totalBytes ? <span>{totalBytes}</span> : null}
    </div>
  )
}

function formatBytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  for (const unit of units) {
    if (amount < 1024 || unit === units[units.length - 1]) {
      return unit === 'B' ? `${Math.round(amount)} B` : `${amount.toFixed(1)} ${unit}`
    }
    amount /= 1024
  }
  return `${amount.toFixed(1)} TB`
}

function RuntimeDetails({ runtime }: { runtime: RuntimeInfo | undefined }) {
  const { t } = useI18n()
  if (!runtime) {
    return <div className="runtime-details">{t('modelSetup.runtimeUnavailable')}</div>
  }
  return (
    <div className="runtime-details">
      <span>torch {runtime.torch_version ?? t('common.notInstalled')}</span>
      <span>torchvision {runtime.torchvision_version ?? t('common.notInstalled')}</span>
      <span>
        CUDA {runtime.cuda_available ? runtime.cuda_version ?? t('common.available') : t('common.notAvailable')}
      </span>
      {runtime.devices.map((device) => (
        <span key={device.id}>{device.id} / {device.name}</span>
      ))}
    </div>
  )
}

function runtimeControl(
  control: ControlSchema,
  runtime: RuntimeInfo | undefined,
  t: (key: string) => string,
): ControlSchema {
  if (control.id !== 'device') {
    return control
  }
  return {
    ...control,
    options: [
      { id: 'auto', label: t('modelSetup.bestAvailable') },
      ...(runtime?.devices.map((device) => ({
        id: device.id,
        label: `${device.id} / ${device.name}`,
      })) ?? []),
      { id: 'cpu', label: 'CPU' },
    ],
  }
}

function runtimeControlValue(
  control: ControlSchema,
  device: string,
  dtype: string,
  controlnetModelId: string,
  safetyChecker: boolean,
): unknown {
  if (control.id === 'dtype') {
    return dtype
  }
  if (control.id === 'controlnet_model_id') {
    if (control.options.some((option) => option.id === controlnetModelId)) {
      return controlnetModelId
    }
    if (typeof control.default_value === 'string' && control.default_value) {
      return control.default_value
    }
    return control.options[0]?.id ?? ''
  }
  if (control.id === 'safety_checker') {
    return safetyChecker
  }
  return device
}

function updateRuntimeControl(
  id: string,
  value: unknown,
  onDeviceChange: (value: string) => void,
  onDtypeChange: (value: string) => void,
  onControlnetModelIdChange: (value: string) => void,
  onSafetyCheckerChange: (value: boolean) => void,
): void {
  if (id === 'dtype' && typeof value === 'string') {
    onDtypeChange(value)
    return
  }
  if (id === 'controlnet_model_id' && typeof value === 'string') {
    onControlnetModelIdChange(value)
    return
  }
  if (id === 'safety_checker') {
    onSafetyCheckerChange(Boolean(value))
    return
  }
  if (typeof value === 'string') {
    onDeviceChange(value)
  }
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

function updateModelSourceValue(
  field: string,
  value: string,
  onModelIdChange: (value: string) => void,
  onLocalPathChange: (value: string) => void,
  onSingleFilePathChange: (value: string) => void,
  onModelUrlChange: (value: string) => void,
): void {
  if (field === MODEL_SOURCE_FIELD_LOCAL_PATH) {
    onLocalPathChange(value)
    return
  }
  if (field === MODEL_SOURCE_FIELD_SINGLE_FILE_PATH) {
    onSingleFilePathChange(value)
    return
  }
  if (field === MODEL_SOURCE_FIELD_MODEL_URL) {
    onModelUrlChange(value)
    return
  }
  if (field === MODEL_SOURCE_FIELD_MODEL_ID) {
    onModelIdChange(value)
  }
}
