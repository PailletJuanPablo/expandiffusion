import type { AdapterInfo } from '../domain/types'
import { CONTROLNET_GUIDE_UI_ENABLED } from '../constants/domain'
import type { TranslateFunction } from '../i18n/i18n'

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

export function getModelSetupDetails(adapter: AdapterInfo, t?: TranslateFunction): ModelSetupDetails {
  return {
    summary: adapterSummary(adapter, t),
    bestFor: adapterBestFor(adapter, t),
    capabilities: capabilityLabels(adapter, t),
    limitations: limitationLabels(adapter, t),
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

function adapterSummary(adapter: AdapterInfo, t?: TranslateFunction): string {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return translate(t, 'modelDetails.sdxlFillRefine.summary', 'Recommended two-pass SDXL fill pipeline with optional visual refine.')
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return translate(t, 'modelDetails.sdxlFill.summary', 'SDXL fill pipeline used by the recommended profile, without the refine pass.')
  }
  if (adapter.family === 'flux') {
    return translate(t, 'modelDetails.flux.summary', 'High-quality fill model for difficult inpaint and outpaint jobs.')
  }
  if (adapter.family === 'chroma') {
    return translate(t, 'modelDetails.chroma.summary', 'Newer inpaint and outpaint profile for testing Chroma results.')
  }
  if (adapter.family === 'z-image') {
    return translate(t, 'modelDetails.zImage.summary', 'Newer Z-Image profile for local inpaint and outpaint experiments.')
  }
  if (CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet) {
    return translate(t, 'modelDetails.controlnet.summary', 'Adds sketch or color guidance when you need more control over the edit.')
  }
  if (adapter.family === 'stable-diffusion-xl') {
    return translate(t, 'modelDetails.sdxl.summary', 'Recommended quality option for most local inpaint and outpaint work.')
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    return translate(t, 'modelDetails.sd15.summary', 'Fast local option for quick edits and common community checkpoints.')
  }
  return adapter.description
}

function adapterBestFor(adapter: AdapterInfo, t?: TranslateFunction): string {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return translate(t, 'modelDetails.sdxlFillRefine.bestFor', 'Best for: the most reliable outpaint results with fewer manual model choices.')
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return translate(t, 'modelDetails.sdxlFill.bestFor', 'Best for: testing the base fill pass without IP-Adapter refine.')
  }
  if (adapter.family === 'flux') {
    return translate(t, 'modelDetails.flux.bestFor', 'Best for: large fills, composition-sensitive edits, and high quality output.')
  }
  if (adapter.family === 'chroma') {
    return translate(t, 'modelDetails.chroma.bestFor', 'Best for: comparing newer model behavior before making it your default.')
  }
  if (adapter.family === 'z-image') {
    return translate(t, 'modelDetails.zImage.bestFor', 'Best for: testing Z-Image checkpoints already prepared for this app.')
  }
  if (CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet) {
    return translate(t, 'modelDetails.controlnet.bestFor', 'Best for: guided edits where a sketch, tile, or color map should steer the result.')
  }
  if (adapter.family === 'stable-diffusion-xl') {
    return translate(t, 'modelDetails.sdxl.bestFor', 'Best for: cleaner detail and stronger prompts when your GPU has enough memory.')
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    return translate(t, 'modelDetails.sd15.bestFor', 'Best for: faster iteration, lower VRAM, and broad checkpoint compatibility.')
  }
  return translate(t, 'modelDetails.defaultBestFor', 'Best for: the workflow described by this adapter.')
}

function capabilityLabels(adapter: AdapterInfo, t?: TranslateFunction): string[] {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return [
      translate(t, 'modelDetails.capability.inpaint', 'Inpaint'),
      translate(t, 'modelDetails.capability.outpaint', 'Outpaint'),
      translate(t, 'modelDetails.capability.visualRefine', 'Visual refine'),
      translate(t, 'modelDetails.capability.pipelineProfile', 'Pipeline profile'),
    ]
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return [
      translate(t, 'modelDetails.capability.inpaint', 'Inpaint'),
      translate(t, 'modelDetails.capability.outpaint', 'Outpaint'),
      translate(t, 'modelDetails.capability.pipelineProfile', 'Pipeline profile'),
    ]
  }
  return [
    adapter.capabilities.inpaint ? translate(t, 'modelDetails.capability.inpaint', 'Inpaint') : '',
    adapter.capabilities.outpaint ? translate(t, 'modelDetails.capability.outpaint', 'Outpaint') : '',
    CONTROLNET_GUIDE_UI_ENABLED && adapter.capabilities.controlnet ? translate(t, 'modelDetails.capability.guideSupport', 'Guide support') : '',
    adapter.capabilities.textual_inversion ? translate(t, 'modelDetails.capability.textualInversion', 'Textual inversion') : '',
    adapter.capabilities.from_single_file ? translate(t, 'modelDetails.capability.singleFile', 'Single-file checkpoints') : '',
  ].filter(Boolean)
}

function limitationLabels(adapter: AdapterInfo, t?: TranslateFunction): string[] {
  if (adapter.id === ADAPTER_SDXL_FILL_IP_REFINE) {
    return [
      translate(t, 'modelDetails.sdxlFillRefine.limit.vram', 'Uses more VRAM than SD 1.5.'),
      translate(t, 'modelDetails.sdxlFillRefine.limit.refine', 'Visual refine adds a second pass and can be slower.'),
    ]
  }
  if (adapter.id === ADAPTER_SDXL_FILL_CONTROLNET_UNION) {
    return [
      translate(t, 'modelDetails.sdxlFill.limit.technical', 'Technical base profile.'),
      translate(t, 'modelDetails.sdxlFill.limit.preferRefine', 'Usually prefer the visual refine profile unless you are comparing passes.'),
    ]
  }
  const limitations = [
    adapter.capabilities.from_single_file
      ? ''
      : translate(t, 'modelDetails.limit.diffusersFolder', 'Requires a full Diffusers folder or Hugging Face repo.'),
    adapter.capabilities.textual_inversion ? '' : translate(t, 'modelDetails.limit.noTextualInversion', 'No textual inversion support.'),
    adapter.capabilities.safety_checker ? '' : translate(t, 'modelDetails.limit.noSafetyChecker', 'No built-in safety checker.'),
    adapter.capabilities.schedulers.length === 1 &&
    adapter.capabilities.schedulers[0] === 'auto'
      ? translate(t, 'modelDetails.limit.fixedScheduler', 'Scheduler is fixed by the pipeline.')
      : '',
  ].filter(Boolean)

  if (adapter.family === 'stable-diffusion-xl') {
    limitations.push(translate(t, 'modelDetails.limit.sdxlVram', 'Uses more VRAM than SD 1.5.'))
  }
  if (adapter.family === 'stable-diffusion-1.5') {
    limitations.push(translate(t, 'modelDetails.limit.sd15Detail', 'Lower native detail than SDXL or newer models.'))
  }
  return limitations
}

function translate(t: TranslateFunction | undefined, key: string, fallback: string): string {
  return t ? t(key, {}, fallback) : fallback
}
