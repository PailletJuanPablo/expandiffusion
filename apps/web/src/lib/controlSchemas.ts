import {
  CONTROLNET_GUIDE_UI_ENABLED,
  GENERATION_MODE_INPAINT,
  type GenerationMode,
} from '../constants/domain'
import type { ControlSchema } from '../domain/types'

const CONTROLNET_HIDDEN_CONTROL_IDS = new Set([
  'controlnet_model_id',
  'controlnet_conditioning_scale',
  'control_guidance_start',
  'control_guidance_end',
])

const INPAINT_HIDDEN_CONTROL_IDS = new Set([
  'outpaint_max_width',
  'outpaint_max_height',
  'outpaint_direction',
  'hf_space_resize_option',
  'hf_space_custom_resize_percentage',
  'hf_space_overlap_percentage',
  'hf_space_overlap_left',
  'hf_space_overlap_right',
  'hf_space_overlap_top',
  'hf_space_overlap_bottom',
])

const OUTPAINT_HIDDEN_CONTROL_IDS = new Set([
  'inpaint_area',
  'mask_crop_padding',
])

/**
 * Filter adapter controls by section.
 *
 * @param controls - Adapter-provided control schemas.
 * @param section - Section id to select.
 * @returns Controls belonging to the section.
 */
export function controlsForSection(controls: ControlSchema[], section: string): ControlSchema[] {
  return controls.filter(
    (control) => control.section === section && isControlnetControlVisible(control.id),
  )
}

export function controlsForGenerationMode(
  controls: ControlSchema[],
  generationMode: GenerationMode,
): ControlSchema[] {
  const visibleControls = controls.filter((control) => isControlnetControlVisible(control.id))
  if (generationMode === GENERATION_MODE_INPAINT) {
    return visibleControls.filter((control) => !INPAINT_HIDDEN_CONTROL_IDS.has(control.id))
  }
  return visibleControls.filter((control) => !OUTPAINT_HIDDEN_CONTROL_IDS.has(control.id))
}

function isControlnetControlVisible(controlId: string): boolean {
  return CONTROLNET_GUIDE_UI_ENABLED || !CONTROLNET_HIDDEN_CONTROL_IDS.has(controlId)
}

/**
 * Resolve whether a control should be disabled based on dependent state.
 *
 * @param control - Control schema.
 * @param parameters - Current generation parameter map.
 * @returns Whether the control should be disabled.
 */
export function isGenerationControlDisabled(
  control: ControlSchema,
  parameters: Record<string, unknown>,
): boolean {
  if (control.id === 'seed') {
    return Boolean(parameters.random_seed)
  }
  if (control.id === 'hf_space_custom_resize_percentage') {
    return parameters.hf_space_resize_option !== 'Custom'
  }
  return false
}
