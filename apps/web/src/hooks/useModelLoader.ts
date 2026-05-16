import { useMutation } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { ERROR_MODEL_LOAD_CANCELLED } from '../constants/domain'
import type { AdapterInfo, ModelLoadProgress } from '../domain/types'
import { cancelModelLoad, getModelLoadProgress, loadModel } from '../lib/apiClient'
import { AppError } from '../lib/errors'
import {
  buildModelLoadRequest,
  type ModelSourceValues,
} from '../lib/modelSources'
import { useEditorStore } from '../store/editorStore'

interface UseModelLoaderOptions {
  selectedAdapterId: string
  selectedAdapter: AdapterInfo | undefined
  modelSource: string
  sourceValues: ModelSourceValues
  device: string
  dtype: string
  safetyChecker: boolean
  controlnetModelId: string
  loraText: string
  textualInversionText: string
  onLoaded: () => Promise<void>
  onChanged: () => Promise<void>
}

/**
 * Coordinate model-load request creation and post-load refreshes.
 *
 * @param options - Current model setup state.
 * @returns Model load mutation.
 */
export function useModelLoader({
  selectedAdapterId,
  selectedAdapter,
  modelSource,
  sourceValues,
  device,
  dtype,
  safetyChecker,
  controlnetModelId,
  loraText,
  textualInversionText,
  onLoaded,
  onChanged,
}: UseModelLoaderOptions) {
  const setErrorMessage = useEditorStore((state) => state.setErrorMessage)
  const [loadProgress, setLoadProgress] = useState<ModelLoadProgress | null>(null)

  const refreshLoadProgress = async () => {
    try {
      setLoadProgress(await getModelLoadProgress())
    } catch {
      // The load error itself is more important than a failed progress refresh.
    }
  }

  const mutation = useMutation({
    mutationFn: () =>
      loadModel(
        buildModelLoadRequest({
          adapterId: selectedAdapterId,
          adapter: selectedAdapter,
          modelSource,
          values: sourceValues,
          device,
          dtype,
          safetyChecker,
          controlnetModelId,
          loraText,
          textualInversionText,
        }),
      ),
    onSuccess: async () => {
      await refreshLoadProgress()
      await onLoaded()
      setErrorMessage(null)
    },
    onError: async (error) => {
      await refreshLoadProgress()
      await onChanged()
      if (error instanceof AppError && error.code === ERROR_MODEL_LOAD_CANCELLED) {
        setErrorMessage(null)
        return
      }
      setErrorMessage(error instanceof Error ? error.message : 'Model load failed.')
    },
  })

  const cancelMutation = useMutation({
    mutationFn: cancelModelLoad,
    onSuccess: async (progress) => {
      setLoadProgress(progress)
      await onChanged()
    },
    onError: (error) => {
      setErrorMessage(error instanceof Error ? error.message : 'Model load cancellation failed.')
    },
  })

  useEffect(() => {
    if (!mutation.isPending) {
      return
    }
    let active = true
    const refresh = () => {
      void getModelLoadProgress()
        .then((progress) => {
          if (active) {
            setLoadProgress(progress)
          }
        })
        .catch(() => undefined)
    }
    refresh()
    const interval = window.setInterval(refresh, 1000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [mutation.isPending])

  return {
    ...mutation,
    loadProgress,
    cancelLoad: () => cancelMutation.mutate(),
    cancelPending: cancelMutation.isPending,
  }
}
