import { useMutation } from '@tanstack/react-query'
import {
  CONTROLNET_GUIDE_UI_ENABLED,
  FILL_TRANSPARENT,
  GENERATION_MODE_INPAINT,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  OUTPAINT_STRATEGY_LOCAL_CONTEXT,
  RESULT_MODE_GENERATED_SELECTION,
  RESULT_MODE_FEATHER_KNOWN,
  RESULT_MODE_PRESERVE_KNOWN,
  JOB_CANCELLED,
  JOB_FAILED,
  JOB_QUEUED,
  JOB_RUNNING,
  JOB_SUCCEEDED,
  WORKSPACE_MODE_EXPAND_IMAGE,
  WORKSPACE_MODE_FREE_EDIT,
} from '../constants/domain'
import {
  cancelJob,
  connectJobEvents,
  getJobResult,
  startInpaint,
  startOutpaint,
} from '../lib/apiClient'
import { composeSelectionResults, renderGenerationInputs } from '../lib/canvasRender'
import { AppError } from '../lib/errors'
import { parseLoras, parseTextualInversions } from '../lib/extensionParsers'
import { isExpandImageAdapter } from '../lib/workspaceMode'
import { useEditorStore } from '../store/editorStore'
import type { AdapterInfo, GenerationParameters } from '../domain/types'
import { useI18n } from '../i18n/useI18n'
import { localizeJobMessage } from '../i18n/metadata'
import type { TranslateFunction } from '../i18n/i18n'

const CONTROL_GUIDE_DISABLED_ADAPTERS = new Set([
  'sdxl-fill-controlnet-union',
  'sdxl-fill-ip-refine',
])

interface UseOutpaintJobOptions {
  selectedAdapter: AdapterInfo | undefined
  loraText: string
  textualInversionText: string
  onPersistentStateRefresh: () => void
}

/**
 * Coordinate generation job creation, websocket updates and cancellation.
 *
 * @param options - Extension text inputs and persistence refresh callback.
 * @returns Job state helpers for the inspector.
 */
export function useOutpaintJob({
  selectedAdapter,
  loraText,
  textualInversionText,
  onPersistentStateRefresh,
}: UseOutpaintJobOptions) {
  const { t } = useI18n()
  const documentState = useEditorStore((state) => state.document)
  const selectedAdapterId = useEditorStore((state) => state.selectedAdapterId)
  const controlGuideEnabled = useEditorStore((state) => state.controlGuideEnabled)
  const controlGuideMaskMode = useEditorStore((state) => state.controlGuideMaskMode)
  const workspaceMode = useEditorStore((state) => state.workspaceMode)
  const generationMode = useEditorStore((state) => state.generationMode)
  const parameters = useEditorStore((state) => state.parameters)
  const currentJob = useEditorStore((state) => state.currentJob)
  const setCurrentJob = useEditorStore((state) => state.setCurrentJob)
  const setCurrentJobSelection = useEditorStore((state) => state.setCurrentJobSelection)
  const setGenerationNote = useEditorStore((state) => state.setGenerationNote)
  const setPendingResults = useEditorStore((state) => state.setPendingResults)
  const pushHistory = useEditorStore((state) => state.pushHistory)
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage)

  const generateMutation = useMutation({
    mutationFn: async () => {
      const directionalOutpaintActive =
        generationMode !== GENERATION_MODE_INPAINT &&
        parameters.outpaint_strategy === OUTPAINT_STRATEGY_DIRECTIONAL
      const hfSpaceFillActive =
        generationMode !== GENERATION_MODE_INPAINT &&
        parameters.outpaint_strategy === OUTPAINT_STRATEGY_HF_SPACE_FILL
      const controlGuideAllowed =
        CONTROLNET_GUIDE_UI_ENABLED &&
        !CONTROL_GUIDE_DISABLED_ADAPTERS.has(selectedAdapterId)
      const expandImageGenerationActive =
        workspaceMode === WORKSPACE_MODE_EXPAND_IMAGE &&
        isExpandImageAdapter(selectedAdapterId)
      const requestParameters = parametersForGenerationMode(
        parameters,
        generationMode,
        expandImageGenerationActive,
        selectedAdapterId,
      )
      const inputs = await renderGenerationInputs(documentState, generationMode, {
        includeControlGuide:
          controlGuideEnabled &&
          controlGuideAllowed &&
          !directionalOutpaintActive &&
          !hfSpaceFillActive,
        controlGuideMaskMode,
        outpaintStrategy: requestParameters.outpaint_strategy,
        outpaintMaxWidth: requestParameters.outpaint_max_width,
        outpaintMaxHeight: requestParameters.outpaint_max_height,
        outpaintDirection: requestParameters.outpaint_direction,
        outpaintGeneratedSize: requestParameters.outpaint_generated_size,
        outpaintContextSize: requestParameters.outpaint_context_size,
        outpaintCrossSize: requestParameters.outpaint_cross_size,
        hfSpaceOverlapPercentage: requestParameters.hf_space_overlap_percentage,
        hfSpaceFixedExpansion: expandImageGenerationActive,
        hfSpaceResizeOption: requestParameters.hf_space_resize_option,
        hfSpaceCustomResizePercentage: requestParameters.hf_space_custom_resize_percentage,
        hfSpaceOverlapLeft: requestParameters.hf_space_overlap_left,
        hfSpaceOverlapRight: requestParameters.hf_space_overlap_right,
        hfSpaceOverlapTop: requestParameters.hf_space_overlap_top,
        hfSpaceOverlapBottom: requestParameters.hf_space_overlap_bottom,
        fixedExpandPercent: requestParameters.fixed_expand_percent,
        fixedExpandWidthPercent: requestParameters.fixed_expand_width_percent,
        fixedExpandHeightPercent: requestParameters.fixed_expand_height_percent,
        fixedExpandOutputScalePercent: fixedExpandOutputScalePercent(requestParameters),
      })
      if (
        controlGuideEnabled &&
        controlGuideAllowed &&
        !directionalOutpaintActive &&
        !hfSpaceFillActive &&
        !inputs.conditioning
      ) {
        throw new AppError(
          'CONTROL_GUIDE_REQUIRED',
          'Draw a sketch guide inside the generated area, or turn the guide off.',
        )
      }
      setCurrentJobSelection(inputs.previewSelection)
      setGenerationNote(null)
      const request = {
        adapter_id: selectedAdapterId,
        image: inputs.image,
        mask: inputs.mask,
        conditioning: inputs.conditioning,
        mode: generationMode,
        parameters: {
          ...requestParameters,
          width: inputs.renderSize?.width ?? inputs.selection.width,
          height: inputs.renderSize?.height ?? inputs.selection.height,
          loras: parseLoras(loraText),
          textual_inversions: parseTextualInversions(textualInversionText),
        },
        project_id: documentState.id,
        metadata: {
          selection: inputs.selection,
          reference_count: documentState.references.length,
          generation_mode: generationMode,
          workspace_mode: expandImageGenerationActive ? workspaceMode : WORKSPACE_MODE_FREE_EDIT,
          adapter_family: selectedAdapter?.family,
          directional_outpaint_plan: inputs.directionalPlan,
        },
      }
      const job = generationMode === GENERATION_MODE_INPAINT
        ? await startInpaint(request)
        : await startOutpaint(request)
      return {
        job,
        documentState,
        selection: inputs.selection,
        compositionMask: inputs.compositionMask,
        replaceDocument: inputs.replaceDocument,
        directionalPlan: inputs.directionalPlan,
        parameters: requestParameters,
        selectedAdapterId,
      }
    },
    onSuccess: ({
      job,
      documentState: generationDocument,
      selection,
      compositionMask,
      replaceDocument,
      directionalPlan,
      parameters,
      selectedAdapterId,
    }) => {
      setErrorMessage(null)
      const socket = connectJobEvents(
        job.job_id,
        (event) => {
          setCurrentJob(event.job)
          if (event.job.status === JOB_SUCCEEDED) {
            getJobResult(event.job.id)
              .then(async (result) => {
                const composed = await composeSelectionResults(
                  generationDocument,
                  selection,
                  result.images,
                  compositionMask,
                  {
                    replaceDocument,
                    directionalPlan,
                    softCompositionMask: parameters.result_mode === RESULT_MODE_FEATHER_KNOWN,
                  },
                )
                setPendingResults(composed.images, composed.bounds, replaceDocument)
                setGenerationNote(formatPostprocessorDiagnostics(result.metadata, t))
                pushHistory({
                  id: result.job_id,
                  adapterId: selectedAdapterId,
                  prompt: String(parameters.prompt ?? ''),
                  createdAt: new Date().toISOString(),
                  images: composed.images,
                  acceptedImage: null,
                  resultBounds: composed.bounds,
                })
                onPersistentStateRefresh()
                socket.close()
              })
              .catch((error) =>
                setErrorMessage(error instanceof Error ? error.message : 'Result load failed.'),
              )
          }
          if (event.job.status === JOB_FAILED || event.job.status === JOB_CANCELLED) {
            setCurrentJobSelection(null)
            setGenerationNote(event.job.error ?? event.job.message)
            onPersistentStateRefresh()
            socket.close()
          }
        },
        () => setErrorMessage('Job event stream failed. Check API logs.'),
      )
    },
    onError: (error) => {
      setCurrentJobSelection(null)
      setGenerationNote(null)
      setErrorMessage(error instanceof Error ? error.message : 'Generation failed.')
    },
  })

  const running = currentJob?.status === JOB_RUNNING || currentJob?.status === JOB_QUEUED

  const cancelRunningJob = () => {
    if (!currentJob) {
      return
    }
    cancelJob(currentJob.id)
      .then((job) => {
        setCurrentJob(job)
        onPersistentStateRefresh()
      })
      .catch((error) => setErrorMessage(error instanceof Error ? error.message : 'Cancellation failed.'))
  }

  return {
    generateMutation,
    running,
    cancelRunningJob,
  }
}

function fixedExpandOutputScalePercent(parameters: GenerationParameters): number {
  if (parameters.fixed_expand_output_scale === 'safe') {
    return 60
  }
  if (parameters.fixed_expand_output_scale === 'balanced') {
    return 75
  }
  if (parameters.fixed_expand_output_scale === 'custom') {
    return Math.max(25, Math.min(100, Math.round(parameters.fixed_expand_custom_output_scale)))
  }
  return 100
}

export function parametersForGenerationMode(
  parameters: ReturnType<typeof useEditorStore.getState>['parameters'],
  generationMode: ReturnType<typeof useEditorStore.getState>['generationMode'],
  expandImageGenerationActive: boolean,
  selectedAdapterId: string,
): GenerationParameters {
  if (expandImageGenerationActive) {
    return {
      ...parameters,
      outpaint_strategy: OUTPAINT_STRATEGY_HF_SPACE_FILL,
      result_mode: RESULT_MODE_PRESERVE_KNOWN,
    }
  }
  if (
    generationMode === GENERATION_MODE_INPAINT &&
    parameters.result_mode === RESULT_MODE_GENERATED_SELECTION
  ) {
    return { ...parameters, result_mode: RESULT_MODE_PRESERVE_KNOWN }
  }
  if (
    generationMode !== GENERATION_MODE_INPAINT &&
    isExpandImageAdapter(selectedAdapterId)
  ) {
    return {
      ...parameters,
      outpaint_strategy: OUTPAINT_STRATEGY_LOCAL_CONTEXT,
      fill_mode: FILL_TRANSPARENT,
      result_mode: RESULT_MODE_FEATHER_KNOWN,
    }
  }
  return parameters
}

function formatPostprocessorDiagnostics(
  metadata: Record<string, unknown>,
  t: TranslateFunction,
): string | null {
  const diagnostics = metadata.postprocessors
  if (!Array.isArray(diagnostics) || diagnostics.length === 0) {
    return null
  }
  const reports = diagnostics.filter(
    (item): item is Record<string, unknown> => typeof item === 'object' && item !== null,
  )
  if (reports.length === 0) {
    return null
  }
  const warningReport = [...reports]
    .reverse()
    .find((item) => item.status !== 'applied' && typeof item.message === 'string')
  const report = warningReport ?? reports[reports.length - 1]
  const processorId = String(report.processor_id ?? 'postprocessor')
  const status = String(report.status ?? '')
  const detected = Number(report.detected_regions ?? 0)
  const refined = Number(report.refined_regions ?? 0)
  if (status === 'applied') {
    return t('generationNote.postprocessorApplied', { processorId, refined, detected })
  }
  const message = typeof report.message === 'string' ? report.message : null
  return message
    ? t('generationNote.postprocessorMessage', {
        processorId,
        message: localizeJobMessage(message, t),
      })
    : null
}
