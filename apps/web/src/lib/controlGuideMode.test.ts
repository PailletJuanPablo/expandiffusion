import { describe, expect, it } from 'vitest'
import {
  ADAPTER_SDXL_CONTROLNET_INPAINT,
  ADAPTER_SDXL_INPAINT,
  ADAPTER_SD15_CONTROLNET_INPAINT,
  ADAPTER_SD15_INPAINT,
} from '../constants/domain'
import type { AdapterInfo } from '../domain/types'
import {
  adapterIdForControlGuideMode,
  controlnetModelIdForAdapter,
} from './controlGuideMode'

describe('controlGuideMode', () => {
  it('does not switch away from the loaded ControlNet adapter when guide is disabled', () => {
    expect(
      adapterIdForControlGuideMode(false, ADAPTER_SD15_CONTROLNET_INPAINT, adapters()),
    ).toBeNull()
  })

  it('switches SD 1.5 standard inpaint to the matching ControlNet adapter', () => {
    expect(
      adapterIdForControlGuideMode(true, ADAPTER_SD15_INPAINT, adapters()),
    ).toBe(ADAPTER_SD15_CONTROLNET_INPAINT)
  })

  it('switches SDXL standard inpaint to the matching ControlNet adapter', () => {
    expect(
      adapterIdForControlGuideMode(true, ADAPTER_SDXL_INPAINT, adapters()),
    ).toBe(ADAPTER_SDXL_CONTROLNET_INPAINT)
  })

  it('reads the default ControlNet model from the selected adapter controls', () => {
    const adapter = adapters().find((item) => item.id === ADAPTER_SDXL_CONTROLNET_INPAINT)

    expect(controlnetModelIdForAdapter(adapter)).toBe('xinsir/controlnet-tile-sdxl-1.0')
  })
})

function adapters(): AdapterInfo[] {
  return [
    {
      id: ADAPTER_SD15_INPAINT,
      label: 'Stable Diffusion 1.5 Inpaint',
      family: 'stable-diffusion-1.5',
      description: '',
      default_model_id: 'stable-diffusion-v1-5/stable-diffusion-inpainting',
      plugin_id: null,
      capabilities: {
        inpaint: true,
        outpaint: true,
        img2img: true,
        txt2img: false,
        lora: true,
        controlnet: false,
        ip_adapter: false,
        textual_inversion: true,
        safety_checker: true,
        schedulers: [],
        from_single_file: true,
      },
      loaded: false,
      model_sources: [],
      load_controls: [],
      generation_controls: [],
      generation_defaults: {},
      postprocessors: [],
    },
    {
      id: ADAPTER_SDXL_INPAINT,
      label: 'Stable Diffusion XL Inpaint',
      family: 'stable-diffusion-xl',
      description: '',
      default_model_id: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
      plugin_id: null,
      capabilities: {
        inpaint: true,
        outpaint: true,
        img2img: true,
        txt2img: false,
        lora: true,
        controlnet: false,
        ip_adapter: false,
        textual_inversion: true,
        safety_checker: false,
        schedulers: [],
        from_single_file: true,
      },
      loaded: false,
      model_sources: [],
      load_controls: [],
      generation_controls: [],
      generation_defaults: {},
      postprocessors: [],
    },
    {
      id: ADAPTER_SDXL_CONTROLNET_INPAINT,
      label: 'Stable Diffusion XL Tile ControlNet',
      family: 'stable-diffusion-xl',
      description: '',
      default_model_id: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
      plugin_id: null,
      capabilities: {
        inpaint: true,
        outpaint: true,
        img2img: true,
        txt2img: false,
        lora: true,
        controlnet: true,
        ip_adapter: false,
        textual_inversion: true,
        safety_checker: false,
        schedulers: [],
        from_single_file: true,
      },
      loaded: false,
      model_sources: [],
      load_controls: [
        {
          id: 'controlnet_model_id',
          label: 'ControlNet model',
          kind: 'select',
          section: 'runtime',
          plugin_id: null,
          options: [
            { id: 'xinsir/controlnet-tile-sdxl-1.0', label: 'Tile / color sketch' },
            { id: 'xinsir/controlnet-scribble-sdxl-1.0', label: 'Scribble / line guide' },
          ],
          default_value: 'xinsir/controlnet-tile-sdxl-1.0',
          min: null,
          max: null,
          step: null,
          rows: null,
          placeholder: null,
        },
      ],
      generation_controls: [],
      generation_defaults: {},
      postprocessors: [],
    },
    {
      id: ADAPTER_SD15_CONTROLNET_INPAINT,
      label: 'Stable Diffusion 1.5 Tile ControlNet',
      family: 'stable-diffusion-1.5',
      description: '',
      default_model_id: 'stable-diffusion-v1-5/stable-diffusion-inpainting',
      plugin_id: null,
      capabilities: {
        inpaint: true,
        outpaint: true,
        img2img: true,
        txt2img: false,
        lora: true,
        controlnet: true,
        ip_adapter: false,
        textual_inversion: true,
        safety_checker: true,
        schedulers: [],
        from_single_file: true,
      },
      loaded: true,
      model_sources: [],
      load_controls: [],
      generation_controls: [],
      generation_defaults: {},
      postprocessors: [],
    },
  ]
}
