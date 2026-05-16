import { describe, expect, it } from 'vitest'
import {
  DEFAULT_CONTROLNET_MODEL_ID,
  MODEL_SOURCE_FIELD_LOCAL_PATH,
  MODEL_SOURCE_FIELD_MODEL_URL,
  MODEL_SOURCE_DIRECT_URL,
  MODEL_SOURCE_LOCAL_FOLDER,
} from '../constants/domain'
import type { AdapterInfo, ModelSourceSchema } from '../domain/types'
import { buildModelLoadRequest, getActiveModelSource } from './modelSources'

describe('modelSources', () => {
  it('uses adapter-provided source schemas', () => {
    const adapter = adapterWithSources([
      {
        id: MODEL_SOURCE_DIRECT_URL,
        label: 'Direct URL',
        request_field: MODEL_SOURCE_FIELD_MODEL_URL,
        placeholder: null,
        default_value: null,
      },
    ])

    expect(getActiveModelSource(adapter, MODEL_SOURCE_DIRECT_URL).request_field).toBe(
      MODEL_SOURCE_FIELD_MODEL_URL,
    )
  })

  it('builds a load request from the active source schema', () => {
    const adapter = adapterWithSources([
      {
        id: MODEL_SOURCE_DIRECT_URL,
        label: 'Direct URL',
        request_field: MODEL_SOURCE_FIELD_MODEL_URL,
        placeholder: null,
        default_value: null,
      },
    ])

    expect(
      buildModelLoadRequest({
        adapterId: 'adapter-a',
        adapter,
        modelSource: MODEL_SOURCE_DIRECT_URL,
        values: {
          modelId: 'ignored',
          localPath: '',
          singleFilePath: '',
          modelUrl: 'https://example.test/model.safetensors',
        },
        device: 'cpu',
        dtype: 'float32',
        safetyChecker: false,
        controlnetModelId: '',
        loraText: 'lora.safetensors | 0.5',
        textualInversionText: '',
      }),
    ).toMatchObject({
      adapter_id: 'adapter-a',
      model_id: null,
      model_url: 'https://example.test/model.safetensors',
      device: 'cpu',
      dtype: 'float32',
      safety_checker: false,
      loras: [{ path: 'lora.safetensors', scale: 0.5 }],
    })
  })

  it('uses a source default value when the current field is empty', () => {
    const adapter = adapterWithSources([
      {
        id: MODEL_SOURCE_LOCAL_FOLDER,
        label: 'Local Diffusers folder',
        request_field: MODEL_SOURCE_FIELD_LOCAL_PATH,
        placeholder: null,
        default_value: 'E:\\expandiffusion\\models\\diffusers\\FLUX.1-Fill-dev',
      },
    ])

    expect(
      buildModelLoadRequest({
        adapterId: 'flux-fill-fp8',
        adapter,
        modelSource: MODEL_SOURCE_LOCAL_FOLDER,
        values: {
          modelId: '',
          localPath: '',
          singleFilePath: '',
          modelUrl: '',
        },
        device: 'auto',
        dtype: 'auto',
        safetyChecker: true,
        controlnetModelId: '',
        loraText: '',
        textualInversionText: '',
      }),
    ).toMatchObject({
      adapter_id: 'flux-fill-fp8',
      local_path: 'E:\\expandiffusion\\models\\diffusers\\FLUX.1-Fill-dev',
    })
  })

  it('adds the selected ControlNet model for ControlNet adapters', () => {
    const adapter = {
      ...adapterWithSources([]),
      capabilities: {
        ...adapterWithSources([]).capabilities,
        controlnet: true,
      },
    }

    expect(
      buildModelLoadRequest({
        adapterId: 'sd15-controlnet-inpaint',
        adapter,
        modelSource: MODEL_SOURCE_DIRECT_URL,
        values: {
          modelId: '',
          localPath: '',
          singleFilePath: '',
          modelUrl: 'https://example.test/model.safetensors',
        },
        device: 'auto',
        dtype: 'auto',
        safetyChecker: true,
        controlnetModelId: DEFAULT_CONTROLNET_MODEL_ID,
        loraText: '',
        textualInversionText: '',
      }),
    ).toMatchObject({
      controlnet_model_id: DEFAULT_CONTROLNET_MODEL_ID,
    })
  })
})

function adapterWithSources(modelSources: ModelSourceSchema[]): AdapterInfo {
  return {
    id: 'adapter-a',
    label: 'Adapter A',
    family: 'test',
    description: '',
    default_model_id: null,
    loaded: false,
    plugin_id: null,
    model_sources: modelSources,
    load_controls: [],
    generation_controls: [],
    generation_defaults: {},
    postprocessors: [],
    capabilities: {
      inpaint: true,
      outpaint: true,
      img2img: false,
      txt2img: false,
      lora: true,
      controlnet: false,
      ip_adapter: false,
      textual_inversion: false,
      safety_checker: false,
      schedulers: [],
      from_single_file: true,
    },
  }
}
