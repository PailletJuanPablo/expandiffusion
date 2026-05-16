import { useMutation } from '@tanstack/react-query'
import {
  GENERATION_MODE_INPAINT,
  OUTPAINT_STRATEGY_DIRECTIONAL,
  OUTPAINT_STRATEGY_HF_SPACE_FILL,
  RESULT_MODE_GENERATED_SELECTION,
  RESULT_MODE_PRESERVE_KNOWN,
  JOB_CANCELLED,
  JOB_FAILED,
  JOB_QUEUED,
  JOB_RUNNING,
  JOB_SUCCEEDED,
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
import { useEditorStore } from '../store/editorStore'
import type { AdapterInfo } from '../domain/types'

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
  const documentState = useEditorStore((state) => state.document)
  const selectedAdapterId = useEditorStore((state) => state.selectedAdapterId)
  const controlGuideEnabled = useEditorStore((state) => state.controlGuideEnabled)
  const controlGuideMaskMode = useEditorStore((state) => state.controlGuideMaskMode)
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
      const requestParameters = parametersForGenerationMode(parameters, generationMode)
      const inputs = await renderGenerationInputs(documentState, generationMode, {
        includeControlGuide:
          controlGuideEnabled &&
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
      })
      if (
        controlGuideEnabled &&
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
          width: inputs.selection.width,
          height: inputs.selection.height,
          loras: parseLoras(loraText),
          textual_inversions: parseTextualInversions(textualInversionText),
        },
        project_id: documentState.id,
        metadata: {
          selection: inputs.selection,
          reference_count: documentState.references.length,
          generation_mode: generationMode,
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
                  { replaceDocument, directionalPlan },
                )
                setPendingResults(composed.images, composed.bounds, replaceDocument)
                setGenerationNote(formatPostprocessorDiagnostics(result.metadata))
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

function parametersForGenerationMode(
  parameters: ReturnType<typeof useEditorStore.getState>['parameters'],
  generationMode: ReturnType<typeof useEditorStore.getState>['generationMode'],
) {
  if (
    generationMode === GENERATION_MODE_INPAINT &&
    parameters.result_mode === RESULT_MODE_GENERATED_SELECTION
  ) {
    return { ...parameters, result_mode: RESULT_MODE_PRESERVE_KNOWN }
  }
  if (
    generationMode !== GENERATION_MODE_INPAINT &&
    parameters.outpaint_strategy === OUTPAINT_STRATEGY_HF_SPACE_FILL &&
    parameters.result_mode === RESULT_MODE_GENERATED_SELECTION
  ) {
    return { ...parameters, result_mode: RESULT_MODE_PRESERVE_KNOWN }
  }
  return parameters
}

function formatPostprocessorDiagnostics(metadata: Record<string, unknown>): string | null {
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
    return `${processorId} applied: ${refined}/${detected} regions refined.`
  }
  const message = typeof report.message === 'string' ? report.message : null
  return message ? `${processorId}: ${message}` : null
}
