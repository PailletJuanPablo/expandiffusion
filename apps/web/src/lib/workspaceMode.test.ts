import { describe, expect, it } from 'vitest'
import {
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
  ADAPTER_SDXL_FILL_IP_REFINE,
  ADAPTER_SD15_INPAINT,
  WORKSPACE_MODE_EXPAND_IMAGE,
  WORKSPACE_MODE_FREE_EDIT,
} from '../constants/domain'
import type { AdapterInfo } from '../domain/types'
import { adapterIdForWorkspaceMode } from './workspaceMode'

describe('workspaceMode', () => {
  it('uses the SDXL fill profile for fixed image expansion', () => {
    expect(
      adapterIdForWorkspaceMode(
        WORKSPACE_MODE_EXPAND_IMAGE,
        ADAPTER_SD15_INPAINT,
        [
          adapter(ADAPTER_SD15_INPAINT),
          adapter(ADAPTER_SDXL_FILL_CONTROLNET_UNION),
          adapter(ADAPTER_SDXL_FILL_IP_REFINE),
        ],
      ),
    ).toBe(ADAPTER_SDXL_FILL_IP_REFINE)
  })

  it('preserves the selected adapter for free edit mode', () => {
    expect(
      adapterIdForWorkspaceMode(
        WORKSPACE_MODE_FREE_EDIT,
        ADAPTER_SD15_INPAINT,
        [
          adapter(ADAPTER_SD15_INPAINT),
          adapter(ADAPTER_SDXL_FILL_IP_REFINE),
        ],
      ),
    ).toBe(ADAPTER_SD15_INPAINT)
  })
})

function adapter(id: string): AdapterInfo {
  return {
    id,
    label: id,
    family: 'stable-diffusion-xl',
    description: '',
    default_model_id: id,
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
    model_sources: [],
    load_controls: [],
    generation_controls: [],
    generation_defaults: {},
    postprocessors: [],
  }
}
