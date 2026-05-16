import type { DocumentBounds, SelectionRect, ViewportState } from '../domain/types'

export const ONBOARDING_STEP_WELCOME = 'welcome'
export const ONBOARDING_STEP_LOAD_MODEL = 'load-model'
export const ONBOARDING_STEP_TOOLS = 'tools'
export const ONBOARDING_STEP_UPLOAD_IMAGE = 'upload-image'
export const ONBOARDING_STEP_IMAGE_FOCUS = 'image-focus'
export const ONBOARDING_STEP_PREPARE_OUTPAINT = 'prepare-outpaint'
export const ONBOARDING_STEP_PROMPT = 'prompt'
export const ONBOARDING_STEP_GENERATE = 'generate'

export const ONBOARDING_TARGET_TOPBAR_MODEL = 'topbar-model'
export const ONBOARDING_TARGET_SETUP_BUTTON = 'setup-button'
export const ONBOARDING_TARGET_SETUP_DIALOG = 'setup-dialog'
export const ONBOARDING_TARGET_TOOLBAR = 'tool-rail'
export const ONBOARDING_TARGET_UPLOAD_BUTTON = 'upload-image-button'
export const ONBOARDING_TARGET_CANVAS = 'canvas-workspace'
export const ONBOARDING_TARGET_PROMPT = 'prompt-controls'
export const ONBOARDING_TARGET_PROMPT_INPUT = 'prompt-input'
export const ONBOARDING_TARGET_GENERATE_BUTTON = 'generate-button'
export const ONBOARDING_TARGET_FILMSTRIP = 'filmstrip'

export const ONBOARDING_ACTION_NONE = 'none'
export const ONBOARDING_ACTION_OPEN_SETUP = 'open-setup'
export const ONBOARDING_ACTION_LOAD_MODEL = 'load-model'
export const ONBOARDING_ACTION_CHOOSE_IMAGE = 'choose-image'
export const ONBOARDING_ACTION_USE_OUTPAINT = 'use-outpaint'
export const ONBOARDING_ACTION_FOCUS_PROMPT = 'focus-prompt'
export const ONBOARDING_ACTION_GENERATE = 'generate'

export type OnboardingStepId =
  | typeof ONBOARDING_STEP_WELCOME
  | typeof ONBOARDING_STEP_LOAD_MODEL
  | typeof ONBOARDING_STEP_TOOLS
  | typeof ONBOARDING_STEP_UPLOAD_IMAGE
  | typeof ONBOARDING_STEP_IMAGE_FOCUS
  | typeof ONBOARDING_STEP_PREPARE_OUTPAINT
  | typeof ONBOARDING_STEP_PROMPT
  | typeof ONBOARDING_STEP_GENERATE

export type OnboardingTargetId =
  | typeof ONBOARDING_TARGET_TOPBAR_MODEL
  | typeof ONBOARDING_TARGET_SETUP_BUTTON
  | typeof ONBOARDING_TARGET_SETUP_DIALOG
  | typeof ONBOARDING_TARGET_TOOLBAR
  | typeof ONBOARDING_TARGET_UPLOAD_BUTTON
  | typeof ONBOARDING_TARGET_CANVAS
  | typeof ONBOARDING_TARGET_PROMPT
  | typeof ONBOARDING_TARGET_PROMPT_INPUT
  | typeof ONBOARDING_TARGET_GENERATE_BUTTON
  | typeof ONBOARDING_TARGET_FILMSTRIP

export type OnboardingActionId =
  | typeof ONBOARDING_ACTION_NONE
  | typeof ONBOARDING_ACTION_OPEN_SETUP
  | typeof ONBOARDING_ACTION_LOAD_MODEL
  | typeof ONBOARDING_ACTION_CHOOSE_IMAGE
  | typeof ONBOARDING_ACTION_USE_OUTPAINT
  | typeof ONBOARDING_ACTION_FOCUS_PROMPT
  | typeof ONBOARDING_ACTION_GENERATE

export interface OnboardingProgressState {
  modelLoaded: boolean
  modelLoading: boolean
  imageLoaded: boolean
  outpaintReady: boolean
  generationStarted: boolean
}

export interface OnboardingStepDefinition {
  id: OnboardingStepId
  targetId: OnboardingTargetId
  kicker: string
  title: string
  body: string
  waitingBody: string
  primaryAction: OnboardingActionId
  primaryLabel: string
}

export interface OnboardingStepState {
  complete: boolean
  waiting: boolean
  canContinue: boolean
}

export interface OnboardingScreenRect {
  top: number
  left: number
  right: number
  bottom: number
  width: number
  height: number
}

export const ONBOARDING_STEPS: OnboardingStepDefinition[] = [
  {
    id: ONBOARDING_STEP_WELCOME,
    targetId: ONBOARDING_TARGET_TOPBAR_MODEL,
    kicker: 'Welcome',
    title: 'Explore what diffusion models can imagine',
    body: 'This workspace is a hands-on way to test image generation with diffusion models. We will warm up a model, bring in an image, mark where the canvas should grow or change, write a prompt, and review the first result together.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_NONE,
    primaryLabel: 'Begin',
  },
  {
    id: ONBOARDING_STEP_LOAD_MODEL,
    targetId: ONBOARDING_TARGET_SETUP_BUTTON,
    kicker: 'Warm up',
    title: 'Prepare the image engine',
    body: 'A model needs to be loaded before it can interpret prompts and pixels. Start with the recommended profile first; smaller models are faster, while larger families can take longer and use more VRAM.',
    waitingBody: 'The first load can take a few minutes. Stay here and the guide will continue automatically as soon as the model is ready.',
    primaryAction: ONBOARDING_ACTION_LOAD_MODEL,
    primaryLabel: 'Load model',
  },
  {
    id: ONBOARDING_STEP_TOOLS,
    targetId: ONBOARDING_TARGET_TOOLBAR,
    kicker: 'Creative tools',
    title: 'Choose how the image will change',
    body: 'Outpaint extends the scene beyond an edge. Inpaint edits a painted area inside the image. The rail also gives you selection, sketch guidance, pan, erase, upload, and project actions.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_NONE,
    primaryLabel: 'Continue',
  },
  {
    id: ONBOARDING_STEP_UPLOAD_IMAGE,
    targetId: ONBOARDING_TARGET_UPLOAD_BUTTON,
    kicker: 'Source image',
    title: 'Bring in the image you want to transform',
    body: 'Choose a source image. The first one becomes the base canvas; any extra images become references you can position later.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_CHOOSE_IMAGE,
    primaryLabel: 'Choose image',
  },
  {
    id: ONBOARDING_STEP_IMAGE_FOCUS,
    targetId: ONBOARDING_TARGET_CANVAS,
    kicker: 'Canvas',
    title: 'Now the image becomes editable space',
    body: 'The canvas is where generations are previewed and composed. Hold Shift and drag anytime to pan around without switching tools.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_NONE,
    primaryLabel: 'Continue',
  },
  {
    id: ONBOARDING_STEP_PREPARE_OUTPAINT,
    targetId: ONBOARDING_TARGET_CANVAS,
    kicker: 'Generation area',
    title: 'Show the model where to continue',
    body: 'The cyan frame marks the area that will be generated. Place it next to the image and overlap part of the original so the model can read edges, lighting, and texture.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_USE_OUTPAINT,
    primaryLabel: 'Use outpaint',
  },
  {
    id: ONBOARDING_STEP_PROMPT,
    targetId: ONBOARDING_TARGET_PROMPT_INPUT,
    kicker: 'Prompt',
    title: 'Give the model a creative direction',
    body: 'Describe what should appear in the generated area. Scene, style, materials, mood, and lighting all help the model make a more coherent choice.',
    waitingBody: '',
    primaryAction: ONBOARDING_ACTION_FOCUS_PROMPT,
    primaryLabel: 'Write prompt',
  },
  {
    id: ONBOARDING_STEP_GENERATE,
    targetId: ONBOARDING_TARGET_GENERATE_BUTTON,
    kicker: 'Generate',
    title: 'Create the first variation',
    body: 'Start generation, then use the preview strip to compare samples and accept the strongest continuation.',
    waitingBody: 'Generation is running. When previews arrive, review them in the strip and keep the strongest result.',
    primaryAction: ONBOARDING_ACTION_GENERATE,
    primaryLabel: 'Generate when ready',
  },
]

/**
 * Return completion and navigation availability for one onboarding step.
 *
 * @param stepId - Step to evaluate.
 * @param progress - Current editor and generation progress.
 * @returns Step state for rendering and navigation.
 */
export function getOnboardingStepState(
  stepId: OnboardingStepId,
  progress: OnboardingProgressState,
): OnboardingStepState {
  const complete = getOnboardingStepComplete(stepId, progress)
  return {
    complete,
    waiting: getOnboardingStepWaiting(stepId, progress),
    canContinue: stepId === ONBOARDING_STEP_LOAD_MODEL ? complete : true,
  }
}

/**
 * Return the next tour index, keeping required setup steps active until complete.
 *
 * @param currentIndex - Current index in `ONBOARDING_STEPS`.
 * @param progress - Current editor and generation progress.
 * @returns The next index or the current one when setup is incomplete.
 */
export function getNextOnboardingStepIndex(
  currentIndex: number,
  progress: OnboardingProgressState,
): number {
  const step = ONBOARDING_STEPS[currentIndex]
  if (!step) {
    return 0
  }
  if (!getOnboardingStepState(step.id, progress).canContinue) {
    return currentIndex
  }
  return Math.min(currentIndex + 1, ONBOARDING_STEPS.length - 1)
}

/**
 * Return whether a live state change should move the tour forward automatically.
 *
 * @param stepId - Active onboarding step.
 * @param progress - Current editor and generation progress.
 * @returns True when the step was completed by a live user action.
 */
export function shouldAutoAdvanceOnboardingStep(
  stepId: OnboardingStepId,
  progress: OnboardingProgressState,
): boolean {
  if (stepId === ONBOARDING_STEP_LOAD_MODEL) {
    return progress.modelLoaded
  }
  if (stepId === ONBOARDING_STEP_UPLOAD_IMAGE) {
    return progress.imageLoaded
  }
  return false
}

/**
 * Map document-space bounds to browser viewport coordinates for canvas spotlight targets.
 *
 * @param bounds - Bounds in document coordinates.
 * @param viewport - Current canvas pan and zoom.
 * @param canvasRect - Browser-space origin of the canvas container.
 * @returns Screen coordinates for the spotlight.
 */
export function getOnboardingDocumentTargetRect(
  bounds: DocumentBounds | SelectionRect,
  viewport: ViewportState,
  canvasRect: { left: number; top: number },
): OnboardingScreenRect {
  const left = Math.round(canvasRect.left + viewport.x + bounds.x * viewport.zoom)
  const top = Math.round(canvasRect.top + viewport.y + bounds.y * viewport.zoom)
  const width = Math.round(bounds.width * viewport.zoom)
  const height = Math.round(bounds.height * viewport.zoom)
  return {
    left,
    top,
    right: left + width,
    bottom: top + height,
    width,
    height,
  }
}

function getOnboardingStepComplete(
  stepId: OnboardingStepId,
  progress: OnboardingProgressState,
): boolean {
  if (stepId === ONBOARDING_STEP_LOAD_MODEL) {
    return progress.modelLoaded
  }
  if (stepId === ONBOARDING_STEP_UPLOAD_IMAGE) {
    return progress.imageLoaded
  }
  if (stepId === ONBOARDING_STEP_PREPARE_OUTPAINT) {
    return progress.outpaintReady
  }
  if (stepId === ONBOARDING_STEP_GENERATE) {
    return progress.generationStarted
  }
  return true
}

function getOnboardingStepWaiting(
  stepId: OnboardingStepId,
  progress: OnboardingProgressState,
): boolean {
  if (stepId === ONBOARDING_STEP_LOAD_MODEL) {
    return progress.modelLoading && !progress.modelLoaded
  }
  if (stepId === ONBOARDING_STEP_GENERATE) {
    return progress.generationStarted
  }
  return false
}
