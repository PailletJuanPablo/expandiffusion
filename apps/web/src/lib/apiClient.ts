import { API_BASE_PATH } from '../constants/domain'
import { AppError } from './errors'
import type {
  AdapterInfo,
  JobCreateResponse,
  JobEvent,
  JobInfo,
  JobResult,
  ModelInfo,
  ModelLoadProgress,
  ModelLoadRequest,
  OutpaintRequest,
  PersistentState,
  PluginActionInfo,
  PluginActionResult,
  PluginActionRunRequest,
  PluginInfo,
  PluginToolInfo,
  RuntimeInfo,
} from '../domain/types'

/**
 * Fetch JSON from the backend and normalize API errors.
 *
 * @param path - API path below `/api`.
 * @param init - Fetch options.
 * @returns Parsed response body.
 * @throws {AppError} When the backend returns a structured error.
 */
export async function fetchJson<TResponse>(path: string, init?: RequestInit): Promise<TResponse> {
  const response = await fetch(`${API_BASE_PATH}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  const body = await response.json()
  if (!response.ok) {
    const detail = readErrorDetail(body)
    throw new AppError(detail.code, detail.message)
  }
  return body
}

/**
 * Read registered model adapters.
 *
 * @returns Adapter metadata.
 */
export function listAdapters(): Promise<AdapterInfo[]> {
  return fetchJson<AdapterInfo[]>('/adapters')
}

/**
 * Read local plugin load status.
 *
 * @returns Plugin metadata and load errors.
 */
export function listPlugins(): Promise<PluginInfo[]> {
  return fetchJson<PluginInfo[]>('/plugins')
}

/**
 * Read enabled plugin actions.
 *
 * @returns Plugin action metadata and controls.
 */
export function listPluginActions(): Promise<PluginActionInfo[]> {
  return fetchJson<PluginActionInfo[]>('/plugins/actions')
}

/**
 * Read enabled plugin editor tools.
 *
 * @returns Plugin tool metadata and controls.
 */
export function listPluginTools(): Promise<PluginToolInfo[]> {
  return fetchJson<PluginToolInfo[]>('/plugins/tools')
}

/**
 * Run a plugin action on a selected image block.
 *
 * @param actionId - Plugin action id.
 * @param request - Selected image payload and control values.
 * @returns Generic plugin action result.
 */
export function runPluginAction(
  actionId: string,
  request: PluginActionRunRequest,
): Promise<PluginActionResult> {
  return fetchJson<PluginActionResult>(`/plugins/actions/${actionId}/run`, {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/**
 * Enable a local backend plugin.
 *
 * @param pluginId - Plugin id.
 * @returns Updated plugin metadata.
 */
export function enablePlugin(pluginId: string): Promise<PluginInfo> {
  return fetchJson<PluginInfo>(`/plugins/${pluginId}/enable`, { method: 'POST' })
}

/**
 * Disable a local backend plugin.
 *
 * @param pluginId - Plugin id.
 * @returns Updated plugin metadata.
 */
export function disablePlugin(pluginId: string): Promise<PluginInfo> {
  return fetchJson<PluginInfo>(`/plugins/${pluginId}/disable`, { method: 'POST' })
}

/**
 * Read model load state.
 *
 * @returns Model states.
 */
export function listModels(): Promise<ModelInfo[]> {
  return fetchJson<ModelInfo[]>('/models')
}

/**
 * Read backend PyTorch and hardware runtime state.
 *
 * @returns Runtime metadata.
 */
export function getRuntime(): Promise<RuntimeInfo> {
  return fetchJson<RuntimeInfo>('/runtime')
}

/**
 * Read persisted local application history.
 *
 * @returns Persisted state.
 */
export function getPersistentState(): Promise<PersistentState> {
  return fetchJson<PersistentState>('/state').catch((error) => {
    if (error instanceof AppError && error.code === 'REQUEST_FAILED') {
      return emptyPersistentState()
    }
    throw error
  })
}

/**
 * Load a model adapter.
 *
 * @param request - Model load payload.
 * @returns Loaded model state.
 */
export function loadModel(request: ModelLoadRequest): Promise<ModelInfo> {
  return fetchJson<ModelInfo>('/models/load', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/**
 * Unload a model adapter.
 *
 * @param adapterId - Adapter to unload.
 * @returns Unloaded model state.
 */
export function unloadModel(adapterId: string): Promise<ModelInfo> {
  return fetchJson<ModelInfo>('/models/unload', {
    method: 'POST',
    body: JSON.stringify({ adapter_id: adapterId }),
  })
}

/**
 * Read current model load progress.
 *
 * @returns Latest backend model load progress.
 */
export function getModelLoadProgress(): Promise<ModelLoadProgress> {
  return fetchJson<ModelLoadProgress>('/models/load/progress')
}

/**
 * Request cancellation of the current model load.
 *
 * @returns Latest backend model load progress.
 */
export function cancelModelLoad(): Promise<ModelLoadProgress> {
  return fetchJson<ModelLoadProgress>('/models/load/cancel', { method: 'POST' })
}

/**
 * Start an outpaint job.
 *
 * @param request - Outpaint payload.
 * @returns Created job.
 */
export function startOutpaint(request: OutpaintRequest): Promise<JobCreateResponse> {
  return fetchJson<JobCreateResponse>('/generations/outpaint', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/**
 * Start an inpaint job.
 *
 * @param request - Inpaint payload.
 * @returns Created job.
 */
export function startInpaint(request: OutpaintRequest): Promise<JobCreateResponse> {
  return fetchJson<JobCreateResponse>('/generations/inpaint', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/**
 * Cancel a running job.
 *
 * @param jobId - Job id.
 * @returns Latest job state.
 */
export function cancelJob(jobId: string): Promise<JobInfo> {
  return fetchJson<JobInfo>(`/jobs/${jobId}/cancel`, { method: 'POST' })
}

/**
 * Get a job result.
 *
 * @param jobId - Job id.
 * @returns Generated images.
 */
export function getJobResult(jobId: string): Promise<JobResult> {
  return fetchJson<JobResult>(`/jobs/${jobId}/result`)
}

/**
 * Open a websocket stream for job events.
 *
 * @param jobId - Job id.
 * @param onEvent - Event callback.
 * @returns Open websocket.
 */
export function connectJobEvents(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onError: () => void,
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const socket = new WebSocket(`${protocol}//${window.location.host}${API_BASE_PATH}/jobs/${jobId}/events`)
  socket.onmessage = (message) => {
    onEvent(JSON.parse(message.data))
  }
  socket.onerror = onError
  return socket
}

function readErrorDetail(body: unknown): { code: string; message: string } {
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const detail = body.detail
    if (typeof detail === 'object' && detail !== null && 'code' in detail && 'message' in detail) {
      return {
        code: String(detail.code),
        message: String(detail.message),
      }
    }
  }
  return { code: 'REQUEST_FAILED', message: 'The request failed.' }
}

function emptyPersistentState(): PersistentState {
  return {
    version: 1,
    updated_at: new Date(0).toISOString(),
    current_model: null,
    model_loads: [],
    projects: [],
    generations: [],
    plugin_states: [],
  }
}
