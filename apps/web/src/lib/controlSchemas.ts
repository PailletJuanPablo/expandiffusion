import { GENERATION_MODE_INPAINT, type GenerationMode } from '../constants/domain'
import type { ControlSchema } from '../domain/types'

const INPAINT_HIDDEN_CONTROL_IDS = new Set([
  'outpaint_direction',
  'hf_space_resize_option',
  'hf_space_custom_resize_percentage',
  'hf_space_overlap_percentage',
  'hf_space_overlap_left',
  'hf_space_overlap_right',
  'hf_space_overlap_top',
  'hf_space_overlap_bottom',
])

/**
 * Filter adapter controls by section.
 *
 * @param controls - Adapter-provided control schemas.
 * @param section - Section id to select.
 * @returns Controls belonging to the section.
 */
export function controlsForSection(controls: ControlSchema[], section: string): ControlSchema[] {
  return controls.filter((control) => control.section === section)
}

export function controlsForGenerationMode(
  controls: ControlSchema[],
  generationMode: GenerationMode,
): ControlSchema[] {
  if (generationMode !== GENERATION_MODE_INPAINT) {
    return controls
  }
  return controls.filter((control) => !INPAINT_HIDDEN_CONTROL_IDS.has(control.id))
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
