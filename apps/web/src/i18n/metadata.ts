import type {
  AdapterInfo,
  ControlOption,
  ControlSchema,
  ModelInfo,
  ModelSourceSchema,
  PersistedModelLoad,
  PersistentState,
  PluginActionInfo,
  PluginInfo,
  PluginToolInfo,
  PostprocessorInfo,
} from '../domain/types'
import type { TranslateFunction } from './i18n'

const FALLBACK_ERROR_KEYS: Record<string, string> = Object.freeze({
  'Model load failed.': 'errors.modelLoadFailed',
  'Model load cancellation failed.': 'errors.modelLoadCancellationFailed',
  'Model unload failed.': 'errors.modelUnloadFailed',
  'Plugin update failed.': 'errors.pluginUpdateFailed',
  'Plugin tool failed.': 'errors.pluginToolFailed',
  'Result load failed.': 'errors.resultLoadFailed',
  'Job event stream failed. Check API logs.': 'errors.eventStreamFailed',
  'Generation failed.': 'errors.generationFailed',
  'Cancellation failed.': 'errors.cancellationFailed',
  'Erase failed.': 'canvas.eraseFailed',
  'Image import failed.': 'project.imageImportFailed',
  'Project load failed.': 'project.projectLoadFailed',
  'Project save failed.': 'project.projectSaveFailed',
  'Select an uploaded image on the canvas first.': 'errors.selectUploadedImage',
  'Draw a sketch guide inside the generated area, or turn the guide off.':
    'errors.controlGuideRequired',
  'Select visible image content first.': 'canvas.pluginImageRequired',
  'Paint an inpaint mask before generating.': 'canvas.maskRequiredBeforeGenerate',
  'Paint an inpaint mask inside the document.': 'canvas.maskRequiredInsideDocument',
  'Import visible image content before inpainting.': 'canvas.imageRequiredBeforeInpainting',
  'Paint an inpaint mask over the visible image.': 'canvas.maskRequiredOverImage',
  'Canvas 2D context is not available.': 'canvas.contextUnavailable',
  'Root element was not found.': 'errors.rootNotFound',
  'The request failed.': 'errors.requestFailed',
  'Image could not be loaded.': 'errors.imageLoadFailed',
  'File could not be read.': 'errors.fileReadFailed',
  'Move the outpaint frame next to visible image content before generating.':
    'errors.outpaintContextRequired',
  'Import visible image content before generating directional outpaint.':
    'errors.directionalOutpaintImageRequired',
  'Import visible image content before generating HF Space outpaint.':
    'errors.hfSpaceOutpaintImageRequired',
  'Load a model before starting generation.': 'errors.modelRequiredBeforeGeneration',
  'Project archive is missing project.json.': 'project.invalidMissingJson',
  'Project archive is missing the raster asset.': 'project.invalidMissingRaster',
  'Project archive is not valid.': 'project.invalidArchive',
  'Project archive version is not supported.': 'project.unsupportedVersion',
  'Project document is not valid.': 'project.invalidDocument',
})

/**
 * Localize an adapter metadata payload without changing technical ids or defaults.
 *
 * @param adapter - Adapter metadata from the API.
 * @param t - Active translator.
 * @returns Adapter metadata with localized display fields.
 */
export function localizeAdapterInfo(adapter: AdapterInfo, t: TranslateFunction): AdapterInfo {
  return {
    ...adapter,
    label: t(`metadata.adapters.${adapter.id}.label`, {}, adapter.label),
    description: t(`metadata.adapters.${adapter.id}.description`, {}, adapter.description),
    model_sources: adapter.model_sources.map((source) => localizeModelSource(source, t)),
    load_controls: adapter.load_controls.map((control) => localizeControlSchema(control, t)),
    generation_controls: adapter.generation_controls.map((control) =>
      localizeControlSchema(control, t),
    ),
    postprocessors: adapter.postprocessors.map((postprocessor) =>
      localizePostprocessorInfo(postprocessor, t),
    ),
  }
}

/**
 * Localize a plugin info payload.
 *
 * @param plugin - Plugin metadata from the API.
 * @param t - Active translator.
 * @returns Plugin metadata with localized display fields.
 */
export function localizePluginInfo(plugin: PluginInfo, t: TranslateFunction): PluginInfo {
  return {
    ...plugin,
    label: t(`metadata.plugins.${plugin.id}.label`, {}, plugin.label),
    description: t(`metadata.plugins.${plugin.id}.description`, {}, plugin.description),
  }
}

/**
 * Localize a plugin action payload.
 *
 * @param action - Plugin action metadata from the API.
 * @param t - Active translator.
 * @returns Plugin action metadata with localized display fields.
 */
export function localizePluginActionInfo(
  action: PluginActionInfo,
  t: TranslateFunction,
): PluginActionInfo {
  return {
    ...action,
    label: t(`metadata.pluginActions.${action.id}.label`, {}, action.label),
    description: t(`metadata.pluginActions.${action.id}.description`, {}, action.description),
    controls: action.controls.map((control) => localizeControlSchema(control, t)),
  }
}

/**
 * Localize a plugin tool payload.
 *
 * @param tool - Plugin tool metadata from the API.
 * @param t - Active translator.
 * @returns Plugin tool metadata with localized display fields.
 */
export function localizePluginToolInfo(
  tool: PluginToolInfo,
  t: TranslateFunction,
): PluginToolInfo {
  return {
    ...tool,
    label: t(`metadata.pluginTools.${tool.id}.label`, {}, tool.label),
    description: t(`metadata.pluginTools.${tool.id}.description`, {}, tool.description),
    result_label: tool.result_label
      ? t(`metadata.pluginTools.${tool.id}.resultLabel`, {}, tool.result_label)
      : tool.result_label,
    controls: tool.controls.map((control) => localizeControlSchema(control, t)),
  }
}

/**
 * Localize a loaded model payload.
 *
 * @param model - Loaded model state.
 * @param t - Active translator.
 * @returns Model state with localized adapter label.
 */
export function localizeModelInfo(model: ModelInfo, t: TranslateFunction): ModelInfo {
  return {
    ...model,
    adapter_label: t(`metadata.adapters.${model.adapter_id}.label`, {}, model.adapter_label),
  }
}

/**
 * Localize a known adapter id for read-only summaries while preserving unknown ids.
 *
 * @param adapterId - Adapter id stored in app state or generation history.
 * @param t - Active translator.
 * @returns Localized adapter label or the original id.
 */
export function localizeAdapterLabel(adapterId: string, t: TranslateFunction): string {
  return t(`metadata.adapters.${adapterId}.label`, {}, adapterId)
}

/**
 * Localize persisted state labels while keeping stored ids and user content unchanged.
 *
 * @param state - Persisted application state.
 * @param t - Active translator.
 * @returns Localized persistent state.
 */
export function localizePersistentState(state: PersistentState, t: TranslateFunction): PersistentState {
  return {
    ...state,
    current_model: state.current_model ? localizePersistedModelLoad(state.current_model, t) : null,
    model_loads: state.model_loads.map((model) => localizePersistedModelLoad(model, t)),
  }
}

/**
 * Localize a known backend or frontend job status.
 *
 * @param status - Raw status value.
 * @param t - Active translator.
 * @returns Localized status or the original status.
 */
export function localizeJobStatus(status: string, t: TranslateFunction): string {
  return t(`metadata.jobStatus.${status}`, {}, status)
}

/**
 * Localize known job and model-load progress messages.
 *
 * @param message - Raw backend message.
 * @param t - Active translator.
 * @returns Localized message or the original message.
 */
export function localizeJobMessage(message: string, t: TranslateFunction): string {
  const huggingFaceMatch = message.match(/^Resolving Hugging Face files for (.+)\.$/)
  if (huggingFaceMatch) {
    return t(
      'metadata.jobMessages.Resolving Hugging Face files for',
      { repoId: huggingFaceMatch[1] },
      message,
    )
  }
  const downloadMatch = message.match(/^Downloading (.+): (.+?)(?: \(file (.+)\))?\.$/)
  if (downloadMatch) {
    const fileName = downloadMatch[1]
    const progress = downloadMatch[2]
    const fileProgress = downloadMatch[3]
    return fileProgress
      ? t(
          'metadata.jobMessages.Downloading file with index',
          { fileName, progress, fileProgress },
          message,
        )
      : t('metadata.jobMessages.Downloading file', { fileName, progress }, message)
  }
  return t(`metadata.jobMessages.${message}`, {}, message)
}

/**
 * Localize known fallback error messages while preserving unknown backend errors.
 *
 * @param message - Error message to display.
 * @param t - Active translator.
 * @returns Localized error message or the original message.
 */
export function localizeErrorMessage(message: string, t: TranslateFunction): string {
  const key = FALLBACK_ERROR_KEYS[message]
  return key ? t(key, {}, message) : message
}

/**
 * Localize a generic control schema by stable id.
 *
 * @param control - API control schema.
 * @param t - Active translator.
 * @returns Localized control schema.
 */
export function localizeControlSchema(
  control: ControlSchema,
  t: TranslateFunction,
): ControlSchema {
  return {
    ...control,
    label: t(`metadata.controls.${control.id}.label`, {}, control.label),
    placeholder: control.placeholder
      ? t(`metadata.controls.${control.id}.placeholder`, {}, control.placeholder)
      : control.placeholder,
    options: control.options.map((option) => localizeControlOption(control.id, option, t)),
  }
}

function localizeControlOption(
  controlId: string,
  option: ControlOption,
  t: TranslateFunction,
): ControlOption {
  return {
    ...option,
    label: t(`metadata.options.${controlId}.${option.id}`, {}, t(`metadata.options.${option.id}`, {}, option.label)),
  }
}

function localizeModelSource(source: ModelSourceSchema, t: TranslateFunction): ModelSourceSchema {
  return {
    ...source,
    label: t(`metadata.modelSources.${source.id}.label`, {}, source.label),
    placeholder: source.placeholder
      ? t(`metadata.modelSources.${source.id}.placeholder`, {}, source.placeholder)
      : source.placeholder,
  }
}

function localizePostprocessorInfo(
  postprocessor: PostprocessorInfo,
  t: TranslateFunction,
): PostprocessorInfo {
  return {
    ...postprocessor,
    label: t(`metadata.postprocessors.${postprocessor.id}.label`, {}, postprocessor.label),
    description: t(
      `metadata.postprocessors.${postprocessor.id}.description`,
      {},
      postprocessor.description,
    ),
  }
}

function localizePersistedModelLoad(
  model: PersistedModelLoad,
  t: TranslateFunction,
): PersistedModelLoad {
  return {
    ...model,
    adapter_label: t(`metadata.adapters.${model.adapter_id}.label`, {}, model.adapter_label),
  }
}
