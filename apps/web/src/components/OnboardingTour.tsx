import { Check, ChevronLeft, ChevronRight, Loader2, Sparkles, X } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from 'react'
import type { DocumentBounds, ModelLoadProgress, SelectionRect, ViewportState } from '../domain/types'
import {
  ONBOARDING_ACTION_CHOOSE_IMAGE,
  ONBOARDING_ACTION_FOCUS_PROMPT,
  ONBOARDING_ACTION_GENERATE,
  ONBOARDING_ACTION_LOAD_MODEL,
  ONBOARDING_ACTION_NONE,
  ONBOARDING_ACTION_USE_OUTPAINT,
  ONBOARDING_STEP_GENERATE,
  ONBOARDING_STEP_IMAGE_FOCUS,
  ONBOARDING_STEP_LOAD_MODEL,
  ONBOARDING_STEP_PREPARE_OUTPAINT,
  ONBOARDING_STEPS,
  ONBOARDING_TARGET_CANVAS,
  ONBOARDING_TARGET_PROMPT_INPUT,
  ONBOARDING_TARGET_SETUP_DIALOG,
  ONBOARDING_TARGET_UPLOAD_BUTTON,
  getOnboardingDocumentTargetRect,
  getNextOnboardingStepIndex,
  getOnboardingStepState,
  shouldAutoAdvanceOnboardingStep,
  type OnboardingScreenRect,
  type OnboardingProgressState,
  type OnboardingStepDefinition,
  type OnboardingTargetId,
} from '../lib/onboardingTour'
import { Button } from './ui/button'
import { Progress } from './ui/progress'

const TARGET_SELECTOR_PREFIX = '[data-tour-id="'
const TARGET_SELECTOR_SUFFIX = '"]'
const SPOTLIGHT_PADDING = 10
const SPOTLIGHT_RADIUS = 12
const POPOVER_WIDTH = 340
const POPOVER_GAP = 18
const VIEWPORT_MARGIN = 18
const MODEL_READY_ADVANCE_DELAY_MS = 900
const IMAGE_READY_ADVANCE_DELAY_MS = 520
const MODEL_LOADING_MESSAGE_INTERVAL_MS = 4200
const MODEL_LOADING_MESSAGES = Object.freeze([
  {
    title: 'Warming the model',
    body: 'The image engine is moving into memory so it can read your prompt and the pixels around the edit area.',
  },
  {
    title: 'Reading visual context',
    body: 'Generation works best when the model can borrow nearby edges, light, texture, and composition from the original image.',
  },
  {
    title: 'Getting faster after this',
    body: 'The first run is usually the slowest because files and GPU memory are being prepared. Once ready, the creative loop feels much quicker.',
  },
  {
    title: 'Next up',
    body: 'After the model is ready, you will choose an image, mark where it should grow or change, and guide the result with a short prompt.',
  },
])

interface OnboardingTourProps {
  progress: OnboardingProgressState
  modelSetupOpen: boolean
  modelLoadProgress: ModelLoadProgress | null
  imageBounds: DocumentBounds | null
  outpaintFrame: SelectionRect
  viewport: ViewportState
  modelLoadDisabled: boolean
  generationDisabled: boolean
  onOpenSetup: () => void
  onLoadModel: () => void
  onUseOutpaint: () => void
  onGenerate: () => void
}

type TourTargetRect = OnboardingScreenRect

/**
 * Render the first-run guided tour over the live editor surface.
 *
 * @param props - Current workflow state and callbacks for guided actions.
 * @returns Spotlight onboarding overlay.
 */
export function OnboardingTour({
  progress,
  modelSetupOpen,
  modelLoadProgress,
  imageBounds,
  outpaintFrame,
  viewport,
  modelLoadDisabled,
  generationDisabled,
  onOpenSetup,
  onLoadModel,
  onUseOutpaint,
  onGenerate,
}: OnboardingTourProps) {
  const [open, setOpen] = useState(true)
  const [stepIndex, setStepIndex] = useState(0)
  const [targetRect, setTargetRect] = useState<TourTargetRect | null>(null)
  const prefersReducedMotion = useReducedMotion()
  const step = ONBOARDING_STEPS[stepIndex] ?? ONBOARDING_STEPS[0]
  const stepState = getOnboardingStepState(step.id, progress)
  const targetId = activeTargetId(step, modelSetupOpen)
  const lastStep = stepIndex === ONBOARDING_STEPS.length - 1
  const percent = Math.round((modelLoadProgress?.progress ?? 0) * 100)

  const refreshTargetRect = useCallback(() => {
    const documentTargetRect = getDocumentStepTargetRect(
      step.id,
      imageBounds,
      outpaintFrame,
      viewport,
    )
    if (documentTargetRect) {
      setTargetRect(documentTargetRect)
      return
    }
    const target = document.querySelector<HTMLElement>(tourSelector(targetId))
    if (!target) {
      setTargetRect(null)
      return
    }
    const rect = target.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) {
      setTargetRect(null)
      return
    }
    setTargetRect({
      top: rect.top,
      left: rect.left,
      right: rect.right,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
    })
  }, [imageBounds, outpaintFrame, step.id, targetId, viewport])

  useEffect(() => {
    if (!open) {
      return
    }
    const resizeObserver = new ResizeObserver(refreshTargetRect)
    resizeObserver.observe(document.body)
    const animationFrame = window.requestAnimationFrame(refreshTargetRect)
    const retryTimer = window.setInterval(refreshTargetRect, 600)
    window.addEventListener('resize', refreshTargetRect)
    return () => {
      resizeObserver.disconnect()
      window.cancelAnimationFrame(animationFrame)
      window.clearInterval(retryTimer)
      window.removeEventListener('resize', refreshTargetRect)
    }
  }, [open, refreshTargetRect])

  useEffect(() => {
    if (!open || !shouldAutoAdvanceOnboardingStep(step.id, progress)) {
      return
    }
    const timer = window.setTimeout(() => {
      setStepIndex((current) => getNextOnboardingStepIndex(current, progress))
    }, autoAdvanceDelayForStep(step.id))
    return () => window.clearTimeout(timer)
  }, [open, progress, step.id])

  useEffect(() => {
    if (!open || step.primaryAction !== ONBOARDING_ACTION_FOCUS_PROMPT) {
      return
    }
    const target = document.querySelector<HTMLElement>(tourSelector(ONBOARDING_TARGET_PROMPT_INPUT))
    if (!target) {
      return
    }
    target.scrollIntoView({
      behavior: prefersReducedMotion ? 'auto' : 'smooth',
      block: 'center',
      inline: 'nearest',
    })
    const animationFrame = window.requestAnimationFrame(() => {
      target.focus({ preventScroll: true })
      refreshTargetRect()
    })
    const refreshTimer = window.setTimeout(refreshTargetRect, prefersReducedMotion ? 80 : 360)
    return () => {
      window.cancelAnimationFrame(animationFrame)
      window.clearTimeout(refreshTimer)
    }
  }, [open, prefersReducedMotion, refreshTargetRect, step.primaryAction])

  const popoverStyle = useMemo(
    () => getPopoverStyle(targetRect),
    [targetRect],
  )
  const spotlightStyle = useMemo(
    () => getSpotlightStyle(targetRect),
    [targetRect],
  )

  const closeTour = () => setOpen(false)
  const previousStep = () => setStepIndex((current) => Math.max(0, current - 1))
  const nextStep = () => {
    if (lastStep) {
      closeTour()
      return
    }
    setStepIndex((current) => getNextOnboardingStepIndex(current, progress))
  }

  const runPrimaryAction = () => {
    if (step.id === ONBOARDING_STEP_LOAD_MODEL) {
      if (progress.modelLoaded) {
        nextStep()
        return
      }
      onOpenSetup()
      if (!progress.modelLoading && !modelLoadDisabled) {
        onLoadModel()
      }
      return
    }
    if (step.primaryAction === ONBOARDING_ACTION_CHOOSE_IMAGE) {
      clickTourTarget(ONBOARDING_TARGET_UPLOAD_BUTTON)
      return
    }
    if (step.primaryAction === ONBOARDING_ACTION_USE_OUTPAINT) {
      onUseOutpaint()
      return
    }
    if (step.primaryAction === ONBOARDING_ACTION_FOCUS_PROMPT) {
      scrollAndFocusTourTarget(ONBOARDING_TARGET_PROMPT_INPUT)
      return
    }
    if (step.primaryAction === ONBOARDING_ACTION_GENERATE) {
      onGenerate()
      return
    }
    nextStep()
  }

  if (!open) {
    return null
  }

  return (
    <AnimatePresence>
      <div className="onboarding-tour-layer" aria-live="polite">
        {!targetRect ? (
          <motion.div
            className="onboarding-tour-scrim"
            initial={prefersReducedMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={prefersReducedMotion ? undefined : { opacity: 0 }}
          />
        ) : null}
        {targetRect ? (
          <motion.div
            className="onboarding-spotlight"
            style={spotlightStyle}
            initial={prefersReducedMotion ? false : { opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.18 }}
          />
        ) : null}
        <motion.section
          className="onboarding-popover"
          role="dialog"
          aria-modal="false"
          aria-label="Guided onboarding"
          style={popoverStyle}
          initial={prefersReducedMotion ? false : { opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
          transition={{ duration: 0.2 }}
        >
          <div className="onboarding-popover-header">
            <span>{step.kicker}</span>
            <Button
              type="button"
              variant="ghost"
              size="smallIcon"
              aria-label="Skip tour"
              title="Skip tour"
              onClick={closeTour}
            >
              <X size={15} />
            </Button>
          </div>
          <div className="onboarding-title-row">
            <Sparkles size={18} />
            <h2>{step.title}</h2>
          </div>
          <p>{stepState.waiting ? step.waitingBody : step.body}</p>
          <StepStatus
            step={step}
            stepState={stepState}
            modelLoadProgress={modelLoadProgress}
            percent={percent}
          />
          <div className="onboarding-progress-dots" aria-label="Tour progress">
            {ONBOARDING_STEPS.map((item, index) => (
              <span
                key={item.id}
                className={index === stepIndex ? 'onboarding-progress-dot-active' : ''}
              />
            ))}
          </div>
          <div className="onboarding-actions">
            <Button
              type="button"
              variant="secondary"
              size="compact"
              disabled={stepIndex === 0}
              onClick={previousStep}
            >
              <ChevronLeft size={15} />
              Back
            </Button>
            {step.primaryAction !== ONBOARDING_ACTION_NONE ? (
              <Button
                type="button"
                variant="secondary"
                size="compact"
                disabled={primaryActionDisabled(
                  step,
                  progress,
                  modelLoadDisabled,
                  generationDisabled,
                )}
                onClick={runPrimaryAction}
              >
                {progress.modelLoading && step.id === ONBOARDING_STEP_LOAD_MODEL ? (
                  <Loader2 className="spin-icon" size={15} />
                ) : null}
                {primaryActionLabel(step, progress, modelSetupOpen, modelLoadDisabled)}
              </Button>
            ) : null}
            <Button
              type="button"
              variant="primary"
              size="compact"
              disabled={!stepState.canContinue}
              onClick={nextStep}
            >
              {lastStep ? 'Finish' : 'Next'}
              {!lastStep ? <ChevronRight size={15} /> : null}
            </Button>
          </div>
        </motion.section>
      </div>
    </AnimatePresence>
  )
}

function StepStatus({
  step,
  stepState,
  modelLoadProgress,
  percent,
}: {
  step: OnboardingStepDefinition
  stepState: ReturnType<typeof getOnboardingStepState>
  modelLoadProgress: ModelLoadProgress | null
  percent: number
}) {
  if (step.id === ONBOARDING_STEP_LOAD_MODEL && stepState.waiting) {
    return (
      <div className="onboarding-wait-block">
        <div className="job-row">
          <span>{modelLoadProgress?.message ?? 'Preparing the model.'}</span>
          <span>{percent}%</span>
        </div>
        <Progress value={percent} />
        {modelLoadProgress?.file_name ? (
          <span>{modelLoadProgress.file_name}</span>
        ) : null}
        <ModelLoadingLessons />
      </div>
    )
  }
  if (!stepState.complete) {
    return <div className="onboarding-status">Required before continuing</div>
  }
  return (
    <div className="onboarding-status onboarding-status-complete">
      <Check size={14} />
      Ready
    </div>
  )
}

function activeTargetId(
  step: OnboardingStepDefinition,
  modelSetupOpen: boolean,
): OnboardingTargetId {
  if (step.id === ONBOARDING_STEP_LOAD_MODEL && modelSetupOpen) {
    return ONBOARDING_TARGET_SETUP_DIALOG
  }
  return step.targetId
}

function primaryActionLabel(
  step: OnboardingStepDefinition,
  progress: OnboardingProgressState,
  modelSetupOpen: boolean,
  modelLoadDisabled: boolean,
): string {
  if (step.id !== ONBOARDING_STEP_LOAD_MODEL) {
    return step.primaryLabel
  }
  if (progress.modelLoaded) {
    return 'Continue'
  }
  if (progress.modelLoading) {
    return 'Loading model'
  }
  if (modelLoadDisabled) {
    return 'Open setup'
  }
  return modelSetupOpen ? 'Load model' : 'Load recommended model'
}

function primaryActionDisabled(
  step: OnboardingStepDefinition,
  progress: OnboardingProgressState,
  modelLoadDisabled: boolean,
  generationDisabled: boolean,
): boolean {
  if (step.id === ONBOARDING_STEP_LOAD_MODEL) {
    return progress.modelLoading
  }
  if (step.id === ONBOARDING_STEP_GENERATE) {
    return generationDisabled
  }
  return modelLoadDisabled && step.primaryAction === ONBOARDING_ACTION_LOAD_MODEL
}

function clickTourTarget(targetId: OnboardingTargetId): void {
  document.querySelector<HTMLElement>(tourSelector(targetId))?.click()
}

function scrollAndFocusTourTarget(targetId: OnboardingTargetId): void {
  const target = document.querySelector<HTMLElement>(tourSelector(targetId))
  if (!target) {
    return
  }
  target.scrollIntoView({ block: 'center', inline: 'nearest' })
  target.focus({ preventScroll: true })
}

function tourSelector(targetId: OnboardingTargetId): string {
  return `${TARGET_SELECTOR_PREFIX}${targetId}${TARGET_SELECTOR_SUFFIX}`
}

function getDocumentStepTargetRect(
  stepId: OnboardingStepDefinition['id'],
  imageBounds: DocumentBounds | null,
  outpaintFrame: SelectionRect,
  viewport: ViewportState,
): TourTargetRect | null {
  const bounds = documentBoundsForStep(stepId, imageBounds, outpaintFrame)
  if (!bounds) {
    return null
  }
  const canvas = document.querySelector<HTMLElement>(tourSelector(ONBOARDING_TARGET_CANVAS))
  if (!canvas) {
    return null
  }
  const canvasRect = canvas.getBoundingClientRect()
  return getOnboardingDocumentTargetRect(bounds, viewport, canvasRect)
}

function documentBoundsForStep(
  stepId: OnboardingStepDefinition['id'],
  imageBounds: DocumentBounds | null,
  outpaintFrame: SelectionRect,
): DocumentBounds | SelectionRect | null {
  if (stepId === ONBOARDING_STEP_IMAGE_FOCUS) {
    return imageBounds
  }
  if (stepId === ONBOARDING_STEP_PREPARE_OUTPAINT) {
    return outpaintFrame
  }
  return null
}

function ModelLoadingLessons() {
  const [messageIndex, setMessageIndex] = useState(0)
  const prefersReducedMotion = useReducedMotion()
  const message = MODEL_LOADING_MESSAGES[messageIndex] ?? MODEL_LOADING_MESSAGES[0]
  useEffect(() => {
    const timer = window.setInterval(() => {
      setMessageIndex((current) => (current + 1) % MODEL_LOADING_MESSAGES.length)
    }, MODEL_LOADING_MESSAGE_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <div className="onboarding-lesson-block" aria-live="polite">
      <div className="onboarding-message-header">
        <span>{message.title}</span>
        <span>{messageIndex + 1}/{MODEL_LOADING_MESSAGES.length}</span>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.p
          key={message.title}
          className="onboarding-lesson-body"
          initial={prefersReducedMotion ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={prefersReducedMotion ? undefined : { opacity: 0, y: -6 }}
          transition={{ duration: 0.22 }}
        >
          {message.body}
        </motion.p>
      </AnimatePresence>
      <div className="onboarding-message-dots" aria-label="Loading message progress">
        {MODEL_LOADING_MESSAGES.map((item, index) => (
          <span
            key={item.title}
            className={index === messageIndex ? 'onboarding-message-dot-active' : ''}
          />
        ))}
      </div>
    </div>
  )
}

function getSpotlightStyle(targetRect: TourTargetRect | null): CSSProperties {
  if (!targetRect) {
    return {}
  }
  return {
    top: targetRect.top - SPOTLIGHT_PADDING,
    left: targetRect.left - SPOTLIGHT_PADDING,
    width: targetRect.width + SPOTLIGHT_PADDING * 2,
    height: targetRect.height + SPOTLIGHT_PADDING * 2,
    borderRadius: SPOTLIGHT_RADIUS,
  }
}

function getPopoverStyle(targetRect: TourTargetRect | null): CSSProperties {
  if (!targetRect) {
    return {
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    }
  }
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  const rightSideLeft = targetRect.right + POPOVER_GAP
  const leftSideLeft = targetRect.left - POPOVER_WIDTH - POPOVER_GAP
  const centeredLeft = targetRect.left + targetRect.width / 2 - POPOVER_WIDTH / 2
  const left =
    rightSideLeft + POPOVER_WIDTH + VIEWPORT_MARGIN <= viewportWidth
      ? rightSideLeft
      : leftSideLeft >= VIEWPORT_MARGIN
        ? leftSideLeft
        : clamp(centeredLeft, VIEWPORT_MARGIN, viewportWidth - POPOVER_WIDTH - VIEWPORT_MARGIN)
  const preferredTop = targetRect.top + targetRect.height / 2 - 120
  return {
    top: clamp(preferredTop, VIEWPORT_MARGIN, viewportHeight - 280),
    left,
    width: POPOVER_WIDTH,
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}

function autoAdvanceDelayForStep(stepId: OnboardingStepDefinition['id']): number {
  if (stepId === ONBOARDING_STEP_LOAD_MODEL) {
    return MODEL_READY_ADVANCE_DELAY_MS
  }
  return IMAGE_READY_ADVANCE_DELAY_MS
}
