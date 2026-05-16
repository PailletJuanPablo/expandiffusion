import { Power, PowerOff, Square, X } from 'lucide-react'
import {
  CONTROL_SECTION_EXTENSIONS,
  CONTROL_SECTION_RUNTIME,
  MODEL_SOURCE_FIELD_LOCAL_PATH,
  MODEL_SOURCE_FIELD_MODEL_ID,
  MODEL_SOURCE_FIELD_MODEL_URL,
  MODEL_SOURCE_FIELD_SINGLE_FILE_PATH,
} from '../constants/domain'
import type { AdapterInfo, ControlSchema, ModelInfo, ModelLoadProgress, RuntimeInfo } from '../domain/types'
import { controlsForSection } from '../lib/controlSchemas'
import {
  getActiveModelSource,
  getModelSources,
  getModelSourceValue,
  type ModelSourceValues,
} from '../lib/modelSources'
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
}: ModelSetupDialogProps) {
  const modelSources = getModelSources(selectedAdapter)
  const activeSource = getActiveModelSource(selectedAdapter, modelSource)
  const runtimeControls = controlsForSection(
    selectedAdapter?.load_controls ?? [],
    CONTROL_SECTION_RUNTIME,
  )
  const extensionControls = controlsForSection(
    selectedAdapter?.load_controls ?? [],
    CONTROL_SECTION_EXTENSIONS,
  )

  return (
    <div
      className="setup-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Model setup"
    >
      <div className="setup-dialog">
        <div className="setup-header">
          <div>
            <h2>Model setup</h2>
            <p>{runtime?.note ?? 'Waiting for backend runtime.'}</p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="smallIcon"
            className="setup-close-button"
            aria-label="Close dialog"
            title="Close"
            onClick={onClose}
          >
            <X size={16} />
          </Button>
        </div>
        <div className="setup-grid">
          <section className="setup-section">
            <div className="section-heading">
              <span>Adapter</span>
            </div>
            <label className="field-label">
              Adapter
              <select
                value={selectedAdapterId}
                disabled={loading}
                onChange={(event) => onAdapterChange(event.target.value)}
              >
                {adapters.map((adapter) => (
                  <option key={adapter.id} value={adapter.id}>
                    {adapter.label}
                  </option>
                ))}
              </select>
            </label>
            <AdapterCapabilities adapter={selectedAdapter} />
            <label className="field-label">
              Model source
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
              <span>Runtime</span>
            </div>
            {runtimeControls.map((control) => (
              <SchemaControl
                key={control.id}
                control={runtimeControl(control, runtime)}
                value={runtimeControlValue(
                  control.id,
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
            ))}
            <RuntimeDetails runtime={runtime} />
          </section>
          <section className="setup-section setup-section-wide">
            <div className="section-heading">
              <span>Load-time extensions</span>
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
        </div>
        <ModelLoadedStatus activeModel={activeModel} />
        <div className="setup-actions">
          <Button
            type="button"
            variant="primary"
            size="large"
            disabled={loading || loadDisabled}
            onClick={onLoad}
          >
            <Power size={16} />
            {loadButtonLabel(loading, activeModel, selectedAdapterId)}
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
              {cancelPending ? 'Cancelling' : 'Cancel load'}
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
              {unloading ? 'Unloading' : 'Unload model'}
            </Button>
          )}
        </div>
        {loading ? <ModelLoadProgressBlock progress={loadProgress} /> : null}
      </div>
    </div>
  )
}

function ModelLoadedStatus({ activeModel }: { activeModel: ModelInfo | undefined }) {
  if (!activeModel?.loaded) {
    return <div className="model-load-status">No model loaded.</div>
  }
  return (
    <div className="model-load-status">
      <span>Loaded model</span>
      <strong>{activeModel.adapter_label}</strong>
      <span>{activeModel.model_url ?? activeModel.model_id ?? activeModel.local_path ?? activeModel.single_file_path ?? activeModel.adapter_id}</span>
    </div>
  )
}

function loadButtonLabel(
  loading: boolean,
  activeModel: ModelInfo | undefined,
  selectedAdapterId: string,
): string {
  if (loading) {
    return 'Loading model'
  }
  if (!activeModel?.loaded) {
    return 'Load model'
  }
  return activeModel.adapter_id === selectedAdapterId ? 'Reload model' : 'Switch model'
}

function ModelLoadProgressBlock({ progress }: { progress: ModelLoadProgress | null }) {
  const percent = Math.round((progress?.progress ?? 0) * 100)
  const files = progress?.files_total
    ? `${progress.files_done ?? 0}/${progress.files_total} files`
    : null
  const fileBytes = progress?.file_bytes_total
    ? `${formatBytes(progress.file_bytes_done ?? 0)} / ${formatBytes(progress.file_bytes_total)}`
    : progress?.file_bytes_done
      ? formatBytes(progress.file_bytes_done)
      : null
  const totalBytes = progress?.bytes_total
    ? `${formatBytes(progress.bytes_done ?? 0)} / ${formatBytes(progress.bytes_total)} total`
    : null
  return (
    <div className="job-block">
      <div className="job-row">
        <span>{progress?.message ?? 'Starting model load.'}</span>
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

function AdapterCapabilities({ adapter }: { adapter: AdapterInfo | undefined }) {
  if (!adapter) {
    return <div className="capability-row">No adapters reported by backend.</div>
  }
  const labels = [
    adapter.capabilities.inpaint ? 'inpaint' : '',
    adapter.capabilities.outpaint ? 'outpaint' : '',
    adapter.capabilities.controlnet ? 'ControlNet' : '',
    adapter.capabilities.img2img ? 'img2img' : '',
    adapter.capabilities.lora ? 'LoRA' : '',
    adapter.capabilities.textual_inversion ? 'TI' : '',
    adapter.capabilities.from_single_file ? 'single-file' : '',
  ].filter(Boolean)
  return (
    <div className="capability-row">
      {labels.map((label) => (
        <span key={label}>{label}</span>
      ))}
    </div>
  )
}

function RuntimeDetails({ runtime }: { runtime: RuntimeInfo | undefined }) {
  if (!runtime) {
    return <div className="runtime-details">Runtime unavailable.</div>
  }
  return (
    <div className="runtime-details">
      <span>torch {runtime.torch_version ?? 'not installed'}</span>
      <span>torchvision {runtime.torchvision_version ?? 'not installed'}</span>
      <span>CUDA {runtime.cuda_available ? runtime.cuda_version ?? 'available' : 'not available'}</span>
      {runtime.devices.map((device) => (
        <span key={device.id}>{device.id} / {device.name}</span>
      ))}
    </div>
  )
}

function runtimeControl(control: ControlSchema, runtime: RuntimeInfo | undefined): ControlSchema {
  if (control.id !== 'device') {
    return control
  }
  return {
    ...control,
    options: [
      { id: 'auto', label: 'Best available' },
      ...(runtime?.devices.map((device) => ({
        id: device.id,
        label: `${device.id} / ${device.name}`,
      })) ?? []),
      { id: 'cpu', label: 'CPU' },
    ],
  }
}

function runtimeControlValue(
  id: string,
  device: string,
  dtype: string,
  controlnetModelId: string,
  safetyChecker: boolean,
): unknown {
  if (id === 'dtype') {
    return dtype
  }
  if (id === 'controlnet_model_id') {
    return controlnetModelId
  }
  if (id === 'safety_checker') {
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
