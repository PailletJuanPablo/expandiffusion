import { POSTPROCESSOR_CATEGORY_CORRECTION } from '../constants/domain'
import type { PostprocessorInfo } from '../domain/types'

export interface CorrectionPipelineItem {
  id: string
  label: string
  description: string
  pluginId: string | null
  available: boolean
}

/**
 * Return correction postprocessors sorted by their default plugin order.
 *
 * @param postprocessors - Runtime postprocessor metadata from the selected adapter.
 * @returns Correction postprocessors available for selection.
 */
export function correctionPostprocessors(
  postprocessors: PostprocessorInfo[],
): PostprocessorInfo[] {
  return [...postprocessors]
    .filter((postprocessor) => postprocessor.category === POSTPROCESSOR_CATEGORY_CORRECTION)
    .sort((left, right) => left.default_order - right.default_order || left.label.localeCompare(right.label))
}

/**
 * Append a correction id if it is not already active.
 *
 * @param pipeline - Current correction pipeline.
 * @param correctionId - Correction id to append.
 * @returns Updated pipeline.
 */
export function appendCorrection(pipeline: string[], correctionId: string): string[] {
  return pipeline.includes(correctionId) ? pipeline : [...pipeline, correctionId]
}

/**
 * Remove a correction id from the pipeline.
 *
 * @param pipeline - Current correction pipeline.
 * @param correctionId - Correction id to remove.
 * @returns Updated pipeline.
 */
export function removeCorrection(pipeline: string[], correctionId: string): string[] {
  return pipeline.filter((item) => item !== correctionId)
}

/**
 * Move one correction by one position inside the pipeline.
 *
 * @param pipeline - Current correction pipeline.
 * @param correctionId - Correction id to move.
 * @param direction - Negative to move up, positive to move down.
 * @returns Updated pipeline.
 */
export function moveCorrection(
  pipeline: string[],
  correctionId: string,
  direction: number,
): string[] {
  const index = pipeline.indexOf(correctionId)
  const nextIndex = index + direction
  if (index < 0 || nextIndex < 0 || nextIndex >= pipeline.length) {
    return pipeline
  }
  const next = [...pipeline]
  const current = next[index]
  next[index] = next[nextIndex]
  next[nextIndex] = current
  return next
}

/**
 * Resolve active pipeline ids into displayable correction rows.
 *
 * @param pipeline - Current correction pipeline.
 * @param postprocessors - Available correction metadata.
 * @returns Active correction rows, including unavailable ids.
 */
export function activeCorrectionItems(
  pipeline: string[],
  postprocessors: PostprocessorInfo[],
): CorrectionPipelineItem[] {
  return pipeline.map((correctionId) => {
    const postprocessor = postprocessors.find((item) => item.id === correctionId)
    if (postprocessor) {
      return {
        id: postprocessor.id,
        label: postprocessor.label,
        description: postprocessor.description,
        pluginId: postprocessor.plugin_id,
        available: true,
      }
    }
    return {
      id: correctionId,
      label: correctionId,
      description: 'Unavailable',
      pluginId: null,
      available: false,
    }
  })
}
