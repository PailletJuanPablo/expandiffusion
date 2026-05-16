import { describe, expect, it } from 'vitest'
import {
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
  FILL_TRANSPARENT,
  GENERATION_MODE_OUTPAINT,
  INPAINT_AREA_WHOLE_SELECTION,
  OUTPAINT_DIRECTION_RIGHT,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  RESULT_MODE_FEATHER_KNOWN,
  RESULT_MODE_PRESERVE_KNOWN,
  SCHEDULER_AUTO,
} from '../constants/domain'
import type { GenerationParameters } from '../domain/types'
import * as outpaintJob from './useOutpaintJob'

describe('useOutpaintJob', () => {
  it('keeps a free outpaint overlap below 20 percent when building request parameters', () => {
    const parametersForGenerationMode = Reflect.get(outpaintJob, 'parametersForGenerationMode')
    expect(typeof parametersForGenerationMode).toBe('function')

    const parameters: GenerationParameters = {
      prompt: '',
      negative_prompt: '',
      width: 1024,
      height: 1024,
      steps: 8,
      guidance_scale: 1.5,
      strength: 1,
      seed: null,
      random_seed: true,
      sample_count: 1,
      scheduler: SCHEDULER_AUTO,
      safety_checker: true,
      img2img: false,
      fill_mode: FILL_TRANSPARENT,
      correction_pipeline: [],
      inpaint_area: INPAINT_AREA_WHOLE_SELECTION,
      mask_crop_padding: 32,
      mask_blur: 0,
      outpaint_max_width: 1536,
      outpaint_max_height: 1024,
      result_mode: RESULT_MODE_PRESERVE_KNOWN,
      outpaint_strategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      outpaint_direction: OUTPAINT_DIRECTION_RIGHT,
      outpaint_generated_size: 1024,
      outpaint_context_size: 512,
      outpaint_cross_size: 1024,
      hf_space_overlap_percentage: 1,
      hf_space_overlap_left: true,
      hf_space_overlap_right: true,
      hf_space_overlap_top: true,
      hf_space_overlap_bottom: true,
      hf_space_resize_option: 'Full',
      hf_space_custom_resize_percentage: 50,
      fixed_expand_percent: 50,
      fixed_expand_width_percent: 50,
      fixed_expand_height_percent: 50,
      fixed_expand_output_scale: 'balanced',
      fixed_expand_custom_output_scale: 75,
      fixed_expand_show_guides: true,
      loras: [],
      textual_inversions: [],
    }

    const requestParameters = parametersForGenerationMode(
      parameters,
      GENERATION_MODE_OUTPAINT,
      false,
      ADAPTER_SDXL_FILL_CONTROLNET_UNION,
    )

    expect(requestParameters.outpaint_strategy).toBe(OUTPAINT_STRATEGY_LOCAL_CONTEXT)
    expect(requestParameters.result_mode).toBe(RESULT_MODE_FEATHER_KNOWN)
    expect(requestParameters.hf_space_overlap_percentage).toBe(1)
  })
})
