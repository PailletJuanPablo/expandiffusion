import { describe, expect, it } from 'vitest'
import type { AdapterInfo } from '../domain/types'
import { LOCALE_ES, createTranslator } from './i18n'
import {
  localizeAdapterInfo,
  localizeErrorMessage,
  localizeJobMessage,
  localizeJobStatus,
  localizePluginActionInfo,
  localizePluginInfo,
  localizePluginToolInfo,
} from './metadata'

describe('metadata localization', () => {
  it('localizes adapter controls and options by stable ids', () => {
    const t = createTranslator(LOCALE_ES)
    const adapter = localizeAdapterInfo(adapterFixture(), t)

    expect(adapter.generation_controls[0].label).toBe('Indicacion negativa')
    expect(adapter.generation_controls[0].options[0].label).toBe('Seleccion completa')
    expect(adapter.model_sources[0].label).toBe('Carpeta local de Diffusers')
    expect(adapter.postprocessors[0].label).toBe('Coincidencia de color')
  })

  it('keeps unknown technical metadata unchanged', () => {
    const t = createTranslator(LOCALE_ES)
    const adapter = localizeAdapterInfo(adapterFixture(), t)

    expect(adapter.label).toBe('Custom Adapter')
    expect(adapter.generation_controls[1].label).toBe('Vendor knob')
    expect(adapter.generation_controls[1].options[0].label).toBe('Vendor option')
  })

  it('localizes known statuses, progress messages, and fallback errors', () => {
    const t = createTranslator(LOCALE_ES)

    expect(localizeJobStatus('running', t)).toBe('en ejecucion')
    expect(localizeJobMessage('Generation complete', t)).toBe('Generacion completa')
    expect(localizeJobMessage('Loading Diffusers pipeline.', t)).toBe(
      'Cargando flujo Diffusers.',
    )
    expect(localizeJobMessage('Resolving Hugging Face files for repo/model.', t)).toBe(
      'Resolviendo archivos de Hugging Face para repo/model.',
    )
    expect(localizeJobMessage('Downloading model.safetensors: 1.0 MB / 2.0 MB (file 1/3).', t)).toBe(
      'Descargando model.safetensors: 1.0 MB / 2.0 MB (archivo 1/3).',
    )
    expect(localizeErrorMessage('Model load failed.', t)).toBe('La carga del modelo fallo.')
    expect(localizeErrorMessage('Image could not be loaded.', t)).toBe(
      'No se pudo cargar la imagen.',
    )
    expect(localizeErrorMessage('Vendor backend error', t)).toBe('Vendor backend error')
  })

  it('localizes known plugin metadata by action, tool, and control ids', () => {
    const t = createTranslator(LOCALE_ES)
    const plugin = localizePluginInfo(pluginFixture(), t)
    const action = localizePluginActionInfo(pluginActionFixture(), t)
    const tool = localizePluginToolInfo(pluginToolFixture(), t)

    expect(plugin.label).toBe('Imagen a texto')
    expect(action.label).toBe('Describir seleccion')
    expect(tool.label).toBe('Imagen a texto')
    expect(tool.result_label).toBe('Descripcion de imagen')
    expect(tool.controls[0].label).toBe('Brillo')
  })
})

function adapterFixture(): AdapterInfo {
  return {
    id: 'custom-adapter',
    label: 'Custom Adapter',
    family: 'custom',
    description: 'Vendor adapter',
    default_model_id: null,
    capabilities: {
      inpaint: true,
      outpaint: true,
      img2img: false,
      txt2img: false,
      lora: false,
      controlnet: false,
      ip_adapter: false,
      textual_inversion: false,
      safety_checker: false,
      schedulers: [],
      from_single_file: false,
    },
    loaded: false,
    plugin_id: null,
    model_sources: [
      {
        id: 'local_folder',
        label: 'Local Diffusers folder',
        request_field: 'local_path',
        placeholder: null,
        default_value: null,
      },
    ],
    load_controls: [],
    generation_controls: [
      {
        id: 'negative_prompt',
        label: 'Negative prompt',
        kind: 'select',
        section: 'basic',
        plugin_id: null,
        options: [{ id: 'whole_selection', label: 'whole selection' }],
        default_value: null,
        min: null,
        max: null,
        step: null,
        rows: null,
        placeholder: null,
      },
      {
        id: 'vendor_knob',
        label: 'Vendor knob',
        kind: 'select',
        section: 'basic',
        plugin_id: null,
        options: [{ id: 'vendor_option', label: 'Vendor option' }],
        default_value: null,
        min: null,
        max: null,
        step: null,
        rows: null,
        placeholder: null,
      },
    ],
    generation_defaults: {},
    postprocessors: [
      {
        id: 'correction-color-match',
        label: 'Color match',
        description: 'Matches generated LAB color statistics to the preserved boundary.',
        plugin_id: null,
        category: 'correction',
        default_order: 100,
      },
    ],
  }
}

function pluginFixture() {
  return {
    id: 'image-to-text',
    label: 'Image to Text',
    version: '1.0.0',
    description: 'Generate Stable Diffusion prompts from selected image regions.',
    path: 'plugins/image-to-text',
    adapter_ids: [],
    postprocessor_ids: [],
    action_ids: ['image-to-text'],
    tool_ids: ['image-to-text'],
    enabled: true,
    loaded: true,
    error_code: null,
    error: null,
  }
}

function pluginActionFixture() {
  return {
    id: 'image-to-text',
    label: 'Describe selection',
    description: 'Runs CLIP Interrogator over the selected block.',
    plugin_id: 'image-to-text',
    menu: 'tools',
    controls: [],
    default_values: {},
  }
}

function pluginToolFixture() {
  return {
    id: 'image-to-text',
    label: 'Image to Text',
    description: 'Select an image block and generate a Stable Diffusion prompt.',
    plugin_id: 'image-to-text',
    action_id: 'image-to-text',
    icon: 'captions',
    icon_color: '#4f46e5',
    accent_color: '#4f46e5',
    result_label: 'Image description',
    target: 'image',
    live_preview: false,
    controls: [
      {
        id: 'image_adjustments_brightness',
        label: 'Brightness',
        kind: 'slider',
        section: 'basic',
        plugin_id: 'image-adjustments',
        options: [],
        default_value: 1,
        min: 0,
        max: 2,
        step: 0.05,
        rows: null,
        placeholder: null,
      },
    ],
    default_values: {},
  }
}
