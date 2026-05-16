import {
  MODEL_SOURCE_DIRECT_URL,
  MODEL_SOURCE_FIELD_LOCAL_PATH,
  MODEL_SOURCE_FIELD_MODEL_ID,
  MODEL_SOURCE_FIELD_MODEL_URL,
  MODEL_SOURCE_FIELD_SINGLE_FILE_PATH,
  MODEL_SOURCE_HUB,
  MODEL_SOURCE_LOCAL_FOLDER,
  MODEL_SOURCE_SINGLE_FILE,
} from '../constants/domain'
import type { AdapterInfo, ModelLoadRequest, ModelSourceSchema } from '../domain/types'
import type { TranslateFunction } from '../i18n/i18n'
import { parseLoras, parseTextualInversions } from './extensionParsers'

export interface ModelSourceValues {
  modelId: string
  localPath: string
  singleFilePath: string
  modelUrl: string
}

interface BuildModelLoadRequestOptions {
  adapterId: string
  adapter: AdapterInfo | undefined
  modelSource: string
  values: ModelSourceValues
  device: string
  dtype: string
  safetyChecker: boolean
  controlnetModelId: string
  loraText: string
  textualInversionText: string
}

/**
 * Return model source schemas for an adapter with a compatibility fallback.
 *
 * @param adapter - Selected adapter metadata.
 * @returns Source schemas.
 */
export function getModelSources(
  adapter: AdapterInfo | undefined,
  t?: TranslateFunction,
): ModelSourceSchema[] {
  if (Array.isArray(adapter?.model_sources) && adapter.model_sources.length > 0) {
    return adapter.model_sources
  }
  return fallbackModelSources(adapter?.default_model_id ?? null, t)
}

/**
 * Return the active source schema, falling back to the first source.
 *
 * @param adapter - Selected adapter metadata.
 * @param modelSource - Current source id.
 * @returns Active source schema.
 */
export function getActiveModelSource(
  adapter: AdapterInfo | undefined,
  modelSource: string,
  t?: TranslateFunction,
): ModelSourceSchema {
  const sources = getModelSources(adapter, t)
  return sources.find((source) => source.id === modelSource) ?? sources[0]
}

/**
 * Check whether the active model source has a usable value.
 *
 * @param adapter - Selected adapter metadata.
 * @param modelSource - Current source id.
 * @param values - Source input values.
 * @returns Whether the model can be loaded.
 */
export function isModelSourceReady(
  adapter: AdapterInfo | undefined,
  modelSource: string,
  values: ModelSourceValues,
): boolean {
  return Boolean(getModelSourceValue(getActiveModelSource(adapter, modelSource), values).trim())
}

/**
 * Build a backend model-load request from dynamic source metadata.
 *
 * @param options - Load form state.
 * @returns Backend request payload.
 */
export function buildModelLoadRequest({
  adapterId,
  adapter,
  modelSource,
  values,
  device,
  dtype,
  safetyChecker,
  controlnetModelId,
  loraText,
  textualInversionText,
}: BuildModelLoadRequestOptions): ModelLoadRequest {
  const source = getActiveModelSource(adapter, modelSource)
  const modelId = source.request_field === MODEL_SOURCE_FIELD_MODEL_ID
    ? getModelSourceValue(source, values)
    : null
  const localPath = source.request_field === MODEL_SOURCE_FIELD_LOCAL_PATH
    ? getModelSourceValue(source, values)
    : null
  const singleFilePath = source.request_field === MODEL_SOURCE_FIELD_SINGLE_FILE_PATH
    ? getModelSourceValue(source, values)
    : null
  const modelUrl = source.request_field === MODEL_SOURCE_FIELD_MODEL_URL
    ? getModelSourceValue(source, values)
    : null

  return {
    adapter_id: adapterId,
    model_id: modelId,
    local_path: localPath,
    single_file_path: singleFilePath,
    model_url: modelUrl,
    device,
    dtype,
    safety_checker: safetyChecker,
    controlnet_model_id: adapter?.capabilities.controlnet
      ? controlnetModelId.trim() || null
      : null,
    controlnet_local_path: null,
    loras: parseLoras(loraText),
    textual_inversions: parseTextualInversions(textualInversionText),
  }
}

/**
 * Read the form value for a source schema.
 *
 * @param source - Source schema.
 * @param values - Source input values.
 * @returns Current string value.
 */
export function getModelSourceValue(source: ModelSourceSchema, values: ModelSourceValues): string {
  const defaultValue = source.default_value ?? ''
  if (source.request_field === MODEL_SOURCE_FIELD_LOCAL_PATH) {
    return values.localPath || defaultValue
  }
  if (source.request_field === MODEL_SOURCE_FIELD_SINGLE_FILE_PATH) {
    return values.singleFilePath || defaultValue
  }
  if (source.request_field === MODEL_SOURCE_FIELD_MODEL_URL) {
    return values.modelUrl || defaultValue
  }
  return values.modelId || defaultValue
}

function fallbackModelSources(
  defaultModelId: string | null,
  t?: TranslateFunction,
): ModelSourceSchema[] {
  return [
    {
      id: MODEL_SOURCE_HUB,
      label: t
        ? t('metadata.modelSources.hub.label', {}, 'Hugging Face model id')
        : 'Hugging Face model id',
      request_field: MODEL_SOURCE_FIELD_MODEL_ID,
      placeholder: null,
      default_value: defaultModelId,
    },
    {
      id: MODEL_SOURCE_LOCAL_FOLDER,
      label: t
        ? t('metadata.modelSources.local_folder.label', {}, 'Local Diffusers folder')
        : 'Local Diffusers folder',
      request_field: MODEL_SOURCE_FIELD_LOCAL_PATH,
      placeholder: 'E:\\models\\stable-diffusion-inpaint',
      default_value: null,
    },
    {
      id: MODEL_SOURCE_SINGLE_FILE,
      label: t
        ? t('metadata.modelSources.single_file.label', {}, 'Local checkpoint file')
        : 'Local checkpoint file',
      request_field: MODEL_SOURCE_FIELD_SINGLE_FILE_PATH,
      placeholder: 'E:\\models\\model.safetensors',
      default_value: null,
    },
    {
      id: MODEL_SOURCE_DIRECT_URL,
      label: t
        ? t('metadata.modelSources.direct_url.label', {}, 'Checkpoint URL')
        : 'Checkpoint URL',
      request_field: MODEL_SOURCE_FIELD_MODEL_URL,
      placeholder: 'https://civitai.com/models/...?... or https://.../model.safetensors',
      default_value: null,
    },
  ]
}
