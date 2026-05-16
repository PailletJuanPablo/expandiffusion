import type { AdapterInfo } from '../domain/types'
import { CONTROLNET_GUIDE_UI_ENABLED } from '../constants/domain'

const ADAPTER_SDXL_FILL_IP_REFINE = 'sdxl-fill-ip-refine'
const ADAPTER_SDXL_FILL_CONTROLNET_UNION = 'sdxl-fill-controlnet-union'
const HIDDEN_ADAPTER_IDS = new Set([
  'sd15-img2img',
  'sdxl-img2img',
  'sd2-inpaint',
  'sdxl-inpaint',
])
const PRIMARY_ADAPTER_ORDER = [
  ADAPTER_SDXL_FILL_IP_REFINE,
  'sd15-inpaint',
]

export interface ModelSetupAdapterGroups {
  primary: AdapterInfo[]
  experimental: AdapterInfo[]
}

export interface ModelSetupDetails {
  summary: string
  bestFor: string
  capabilities: string[]
  limitations: string[]
}

export function shouldOpenInitialModelSetup(
  modelsResolved: boolean,
  hasLoadedModel: boolean,
  alreadyPrompted: boolean,
): boolean {
  return modelsResolved && !hasLoadedModel && !alreadyPrompted
}

export function getModelSetupAdapterGroups(adapters: AdapterInfo[]): ModelSetupAdapterGroups {
  const hasVisualRefine = adapters.some((adapter) => adapter.id === ADAPTER_SDXL_FILL_IP_REFINE)
  const visibleAdapters = adapters.filter((adapter) => !isHiddenAdapter(adapter))
  const primary = visibleAdapters.filter((adapter) =>
    isPrimaryAdapter(adapter, hasVisualRefine),
  )
  return {
    primary: sortPrimaryAdapters(primary),
    experimental: visibleAdapters.filter((adapter) =>
      !isPrimaryAdapter(adapter, hasVisualRefine),
    ),
  }
}

export function getModelSetupDetails(adapter: AdapterInfo): ModelSetupDetails {
  return {
    summary: adapterSummary(adapter),
    bestFor: adapterBestFor(adapter),
    capabilities: capabilityLabels(adapter),
    limitations: limitationLabels(adapter),
  }
}

function isHiddenAdapter(adapter: AdapterInfo): boolean {
  return (
    HIDDEN_ADAPTER_IDS.has(adapter.id) ||
    adapter.label.toLowerCase().includes('standard')
  )
}

function isPrimaryAdapter(adapter: AdapterInfo, hasVisualRefine: boolean): boolean {
  if (PRIMARY_ADAPTER_ORDER.includes(adapter.id)) {
    return true
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION && !hasVisualRefine) {
    return true
  }
  return false
}

function sortPrimaryAdapters(adapters: AdapterInfo[]): AdapterInfo[] {
  return [...adapters].sort((left, right) => {
    const leftIndex = primaryOrderIndex(left.id)
    const rightIndex = primaryOrderIndex(right.id)
    return leftIndex - rightIndex
  })
}

function primaryOrderIndex(adapterId: string): number {
  const index = PRIMARY_ADAPTER_ORDER.indexOf(adapterId)
  return index === -1 ? PRIMARY_ADAPTER_ORDER.length : index
}

function adapterSummary(adapter: AdapterInfo): string {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return 'Recommended two-pass SDXL fill pipeline with optional visual refine.'
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return 'SDXL fill pipeline used by the recommended profile, without the refine pass.'
  }
  if (adapter.family === 'flux') {
    return 'High-quality fill model for difficult inpaint and outpaint jobs.'
  }
  if (adapter.family === 'chroma') {
    return 'Newer inpaint and outpaint profile for testing Chroma results.'
  }
  if (adapter.family === 'z-image') {
    return 'Newer Z-Image profile for local inpaint and outpaint experiments.'
  }
  if (CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet) {
    return 'Adds sketch or color guidance when you need more control over the edit.'
  }
  if (adapter.family === 'stable-diffusion-xl') {
    return 'Recommended quality option for most local inpaint and outpaint work.'
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    return 'Fast local option for quick edits and common community checkpoints.'
  }
  return adapter.description
}

function adapterBestFor(adapter: AdapterInfo): string {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return 'Best for: the most reliable outpaint results with fewer manual model choices.'
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return 'Best for: testing the base fill pass without IP-Adapter refine.'
  }
  if (adapter.family === 'flux') {
    return 'Best for: large fills, composition-sensitive edits, and high quality output.'
  }
  if (adapter.family === 'chroma') {
    return 'Best for: comparing newer model behavior before making it your default.'
  }
  if (adapter.family === 'z-image') {
    return 'Best for: testing Z-Image checkpoints already prepared for this app.'
  }
  if (CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet) {
    return 'Best for: guided edits where a sketch, tile, or color map should steer the result.'
  }
  if (adapter.family === 'stable-diffusion-xl') {
    return 'Best for: cleaner detail and stronger prompts when your GPU has enough memory.'
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    return 'Best for: faster iteration, lower VRAM, and broad checkpoint compatibility.'
  }
  return 'Best for: the workflow described by this adapter.'
}

function capabilityLabels(adapter: AdapterInfo): string[] {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return ['Inpaint', 'Outpaint', 'Visual refine', 'Pipeline profile']
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return ['Inpaint', 'Outpaint', 'Pipeline profile']
  }
  return [
    adapter.capabilities.inpaint ? 'Inpaint' : '',
    adapter.capabilities.outpaint ? 'Outpaint' : '',
    CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet ? 'Guide support' : '',
    adapter.capabilities.textual_inversion ? 'Textual inversion' : '',
    adapter.capabilities.from_single_file ? 'Single-file checkpoints' : '',
  ].filter(Boolean)
}

function limitationLabels(adapter: AdapterInfo): string[] {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return [
      'Uses more VRAM than SD 1.5.',
      'Visual refine adds a second pass and can be slower.',
    ]
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return [
      'Technical base profile.',
      'Usually prefer the visual refine profile unless you are comparing passes.',
    ]
  }
  const limitations = [
    adapter.capabilities.from_single_file
      ? ''
      : 'Requires a full Diffusers folder or Hugging Face repo.',
    adapter.capabilities.textual_inversion ? '' : 'No textual inversion support.',
    adapter.capabilities.safety_checker ? '' : 'No built-in safety checker.',
    adapter.capabilities.schedulers.length === 1 &&
    adapter.capabilities.schedulers[0] === 'auto'
      ? 'Scheduler is fixed by the pipeline.'
      : '',
  ].filter(Boolean)

  if (adapter.family === 'stable-diffusion-xl') {
    limitations.push('Uses more VRAM than SD 1.5.')
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    limitations.push('Lower native detail than SDXL or newer models.')
  }
  return limitations
}
