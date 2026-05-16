import { describe, expect, it } from 'vitest'
import {
  CONTROL_NUMBER,
  CONTROL_SECTION_ADVANCED,
  CONTROL_SECTION_BASIC,
  GENERATION_MODE_INPAINT,
  GENERATION_MODE_OUTPAINT,
} from '../constants/domain'
import type { ControlSchema } from '../domain/types'
import {
  controlsForGenerationMode,
  controlsForSection,
  isGenerationControlDisabled,
} from './controlSchemas'

describe('controlSchemas', () => {
  it('filters controls by adapter-provided section', () => {
    const controls = [
      control('prompt', CONTROL_SECTION_BASIC),
      control('seed', CONTROL_SECTION_ADVANCED),
    ]

    expect(controlsForSection(controls, CONTROL_SECTION_BASIC)).toEqual([controls[0]])
  })

  it('disables seed when random seed is enabled', () => {
    expect(
      isGenerationControlDisabled(control('seed', CONTROL_SECTION_ADVANCED), {
        random_seed: true,
      }),
    ).toBe(true)
  })

  it('disables HF Space custom resize unless Custom is selected', () => {
    expect(
      isGenerationControlDisabled(control('hf_space_custom_resize_percentage', CONTROL_SECTION_BASIC), {
        hf_space_resize_option: 'Full',
      }),
    ).toBe(true)
    expect(
      isGenerationControlDisabled(control('hf_space_custom_resize_percentage', CONTROL_SECTION_BASIC), {
        hf_space_resize_option: 'Custom',
      }),
    ).toBe(false)
  })

  it('hides HF Space outpaint controls while in inpaint mode', () => {
    const controls = [
      control('prompt', CONTROL_SECTION_BASIC),
      control('outpaint_max_width', CONTROL_SECTION_ADVANCED),
      control('outpaint_direction', CONTROL_SECTION_BASIC),
      control('hf_space_overlap_percentage', CONTROL_SECTION_BASIC),
      control('inpaint_area', CONTROL_SECTION_ADVANCED),
      control('mask_crop_padding', CONTROL_SECTION_ADVANCED),
      control('mask_blur', CONTROL_SECTION_ADVANCED),
    ]

    expect(controlsForGenerationMode(controls, GENERATION_MODE_INPAINT).map((item) => item.id))
      .toEqual(['prompt', 'inpaint_area', 'mask_crop_padding', 'mask_blur'])
  })

  it('hides inpaint crop controls while in outpaint mode', () => {
    const controls = [
      control('prompt', CONTROL_SECTION_BASIC),
      control('inpaint_area', CONTROL_SECTION_ADVANCED),
      control('mask_crop_padding', CONTROL_SECTION_ADVANCED),
      control('mask_blur', CONTROL_SECTION_ADVANCED),
      control('outpaint_max_width', CONTROL_SECTION_ADVANCED),
    ]

    expect(controlsForGenerationMode(controls, GENERATION_MODE_OUTPAINT).map((item) => item.id))
      .toEqual(['prompt', 'mask_blur', 'outpaint_max_width'])
  })

  it('hides ControlNet controls while the guide UI is disabled', () => {
    const controls = [
      control('prompt', CONTROL_SECTION_BASIC),
      control('controlnet_model_id', CONTROL_SECTION_ADVANCED),
      control('controlnet_conditioning_scale', CONTROL_SECTION_ADVANCED),
      control('control_guidance_start', CONTROL_SECTION_ADVANCED),
      control('control_guidance_end', CONTROL_SECTION_ADVANCED),
    ]

    expect(controlsForSection(controls, CONTROL_SECTION_ADVANCED)).toEqual([])
    expect(controlsForGenerationMode(controls, GENERATION_MODE_OUTPAINT).map((item) => item.id))
      .toEqual(['prompt'])
  })
})

function control(id: string, section: string): ControlSchema {
  return {
    id,
    label: id,
    kind: CONTROL_NUMBER,
    section,
    plugin_id: null,
    options: [],
    default_value: null,
    min: null,
    max: null,
    step: null,
    rows: null,
    placeholder: null,
  }
}
