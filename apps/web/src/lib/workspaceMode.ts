import {
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
  ADAPTER_SDXL_FILL_IP_REFINE,
  WORKSPACE_MODE_EXPAND_IMAGE,
  type WorkspaceMode,
} from '../constants/domain'
import type { AdapterInfo } from '../domain/types'

const EXPAND_IMAGE_ADAPTER_ORDER = [
  ADAPTER_SDXL_FILL_IP_REFINE,
  ADAPTER_SDXL_FILL_CONTROLNET_UNION,
]

export function adapterIdForWorkspaceMode(
  workspaceMode: WorkspaceMode,
  selectedAdapterId: string,
  adapters: AdapterInfo[],
): string {
  if (workspaceMode !== WORKSPACE_MODE_EXPAND_IMAGE) {
    return selectedAdapterId
  }
  if (isExpandImageAdapter(selectedAdapterId)) {
    return selectedAdapterId
  }
  const adapterIds = new Set(adapters.map((adapter) => adapter.id))
  return (
    EXPAND_IMAGE_ADAPTER_ORDER.find((adapterId) => adapterIds.has(adapterId)) ??
    selectedAdapterId
  )
}

export function isExpandImageAdapter(adapterId: string): boolean {
  return EXPAND_IMAGE_ADAPTER_ORDER.includes(adapterId)
}
