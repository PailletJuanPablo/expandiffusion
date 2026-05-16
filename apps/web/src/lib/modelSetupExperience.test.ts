import { describe, expect, it } from 'vitest'
import type { AdapterInfo } from '../domain/types'
import {
  getModelSetupAdapterGroups,
  getModelSetupDetails,
  shouldOpenInitialModelSetup,
} from './modelSetupExperience'

describe('modelSetupExperience', () => {
  it('prompts for model setup only after models load and no model is active', () => {
    expect(shouldOpenInitialModelSetup(false, false, false)).toBe(false)
    expect(shouldOpenInitialModelSetup(true, true, false)).toBe(false)
    expect(shouldOpenInitialModelSetup(true, false, true)).toBe(false)
    expect(shouldOpenInitialModelSetup(true, false, false)).toBe(true)
  })

  it('hides standard, SD2, and duplicate SDXL inpaint adapters while moving experimental profiles behind more options', () => {
    const groups = getModelSetupAdapterGroups([
      adapter('sd15-inpaint', 'Stable Diffusion 1.5 Inpaint', 'stable-diffusion-1.5'),
      adapter('sd15-img2img', 'Stable Diffusion 1.x Standard', 'stable-diffusion-1.5'),
      adapter('sd15-controlnet-inpaint', 'Stable Diffusion 1.5 Tile ControlNet', 'stable-diffusion-1.5'),
      adapter('sd2-inpaint', 'Stable Diffusion 2 Inpaint', 'stable-diffusion-2'),
      adapter('sdxl-inpaint', 'Stable Diffusion XL Inpaint', 'stable-diffusion-xl'),
      adapter('sdxl-fill-controlnet-union', 'SDXL Fill ControlNet Union', 'stable-diffusion-xl'),
      adapter('sdxl-fill-ip-refine', 'SDXL Fill + IP-Adapter Plus Refine', 'stable-diffusion-xl'),
      adapter('flux-fill', 'FLUX.1 Fill', 'flux'),
      adapter('chroma-inpaint', 'Chroma Inpaint', 'chroma'),
    ])

    expect(groups.primary.map((item) => item.id)).toEqual([
      'sdxl-fill-ip-refine',
      'sd15-inpaint',
    ])
    expect(groups.experimental.map((item) => item.id)).toEqual([
      'sd15-controlnet-inpaint',
      'sdxl-fill-controlnet-union',
      'flux-fill',
      'chroma-inpaint',
    ])
  })

  it('explains Flux limitations without applying them to the SDXL fill profile', () => {
    const sdxlFillDetails = getModelSetupDetails(
      adapter(
        'sdxl-fill-ip-refine',
        'SDXL Fill + IP-Adapter Plus Refine',
        'stable-diffusion-xl',
        {
          controlnet: true,
          ip_adapter: true,
          from_single_file: false,
          textual_inversion: false,
          schedulers: ['auto'],
        },
      ),
    )
    const details = getModelSetupDetails(
      adapter('flux-fill', 'FLUX.1 Fill', 'flux', {
        from_single_file: false,
        textual_inversion: false,
        schedulers: ['auto'],
      }),
    )

    expect(sdxlFillDetails.summary).toContain('two-pass SDXL fill pipeline')
    expect(sdxlFillDetails.capabilities).toContain('Visual refine')
    expect(sdxlFillDetails.limitations).not.toContain('Requires a full Diffusers folder or Hugging Face repo.')
    expect(details.capabilities).toContain('Inpaint')
    expect(details.capabilities).toContain('Outpaint')
    expect(details.limitations).toContain('Requires a full Diffusers folder or Hugging Face repo.')
    expect(details.limitations).toContain('Scheduler is fixed by the pipeline.')
  })
})

function adapter(
  id: string,
  label: string,
  family: string,
  capabilityOverrides: Partial<AdapterInfo['capabilities']> = {},
): AdapterInfo {
  return {
    id,
    label,
    family,
    description: `${label} description.`,
    default_model_id: null,
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
      schedulers: ['auto', 'dpmpp_2m'],
      from_single_file: true,
      ...capabilityOverrides,
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
