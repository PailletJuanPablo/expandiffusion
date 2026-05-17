import { Check, ChevronLeft, ChevronRight, ListChecks, Loader2, Sparkles, X } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
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
  ONBOARDING_TARGET_CANVAS,
  ONBOARDING_TARGET_PROMPT_INPUT,
  ONBOARDING_TARGET_SETUP_DIALOG,
  ONBOARDING_TARGET_UPLOAD_BUTTON,
  getOnboardingDocumentTargetRect,
  getLocalizedOnboardingSteps,
  getNextOnboardingStepIndex,
  getOnboardingStepState,
  shouldAutoAdvanceOnboardingStep,
  type OnboardingScreenRect,
  type OnboardingProgressState,
  type OnboardingStepDefinition,
  type OnboardingTargetId,
} from '../lib/onboardingTour'
import { useI18n } from '../i18n/useI18n'
import { localizeJobMessage } from '../i18n/metadata'
import { Button } from './ui/button'
import { Progress } from './ui/progress'

const TARGET_SELECTOR_PREFIX = '[data-tour-id="'
const TARGET_SELECTOR_SUFFIX = '"]'
const SPOTLIGHT_PADDING = 10
const SPOTLIGHT_RADIUS = 12
const POPOVER_WIDTH = 340
const POPOVER_WAITING_WIDTH = 420
const POPOVER_GAP = 18
const VIEWPORT_MARGIN = 18
const POPOVER_ESTIMATED_HEIGHT = 280
const POPOVER_WAITING_ESTIMATED_HEIGHT = 470
const MODEL_READY_ADVANCE_DELAY_MS = 900
const IMAGE_READY_ADVANCE_DELAY_MS = 520
const MODEL_LOADING_MESSAGE_INTERVAL_MS = 4200
const MODEL_LOADING_STAGES = Object.freeze([
  { key: 'source', start: 0 },
  { key: 'download', start: 3 },
  { key: 'pipeline', start: 20 },
  { key: 'device', start: 74 },
  { key: 'extensions', start: 90 },
  { key: 'ready', start: 98 },
])
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
  const { t } = useI18n()
  const [open, setOpen] = useState(true)
  const [stepIndex, setStepIndex] = useState(0)
  const [targetRect, setTargetRect] = useState<TourTargetRect | null>(null)
  const prefersReducedMotion = useReducedMotion()
  const localizedSteps = useMemo(() => getLocalizedOnboardingSteps(t), [t])
  const step = localizedSteps[stepIndex] ?? localizedSteps[0]
  const stepState = getOnboardingStepState(step.id, progress)
  const targetId = activeTargetId(step, modelSetupOpen)
  const lastStep = stepIndex === localizedSteps.length - 1
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
    () => getPopoverStyle(targetRect, stepState.waiting),
    [stepState.waiting, targetRect],
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
          className={`onboarding-popover${stepState.waiting ? ' onboarding-popover-waiting' : ''}`}
          role="dialog"
          aria-modal="false"
          aria-label={t('onboarding.dialogLabel')}
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
              aria-label={t('common.skipTour')}
              title={t('common.skipTour')}
              onClick={closeTour}
            >
              <X size={15} />
            </Button>
          </div>
          <div className="onboarding-popover-main">
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
          </div>
          <div className="onboarding-progress-dots" aria-label={t('onboarding.tourProgress')}>
            {localizedSteps.map((item, index) => (
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
              {t('common.back')}
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
                {primaryActionLabel(step, progress, modelSetupOpen, modelLoadDisabled, t)}
              </Button>
            ) : null}
            <Button
              type="button"
              variant="primary"
              size="compact"
              disabled={!stepState.canContinue}
              onClick={nextStep}
            >
              {lastStep ? t('common.finish') : t('common.next')}
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
  const { t } = useI18n()
  const elapsedMs = useElapsedMilliseconds(step.id === ONBOARDING_STEP_LOAD_MODEL && stepState.waiting)
  if (step.id === ONBOARDING_STEP_LOAD_MODEL && stepState.waiting) {
    return (
      <ModelLoadWaitPanel
        progress={modelLoadProgress}
        percent={percent}
        elapsedMs={elapsedMs}
      />
    )
  }
  if (!stepState.complete) {
    return <div className="onboarding-status">{t('common.requiredBeforeContinuing')}</div>
  }
  return (
    <div className="onboarding-status onboarding-status-complete">
      <Check size={14} />
      {t('common.ready')}
    </div>
  )
}

function ModelLoadWaitPanel({
  progress,
  percent,
  elapsedMs,
}: {
  progress: ModelLoadProgress | null
  percent: number
  elapsedMs: number
}) {
  const { t } = useI18n()
  const message = progress?.message
    ? localizeJobMessage(progress.message, t)
    : t('onboarding.preparingModel')
  const files = progress?.files_total
    ? `${progress.files_done ?? 0}/${progress.files_total}`
    : null
  const fileBytes = progress?.file_bytes_total
    ? `${formatBytes(progress.file_bytes_done ?? 0)} / ${formatBytes(progress.file_bytes_total)}`
    : progress?.file_bytes_done
      ? formatBytes(progress.file_bytes_done)
      : null
  const totalBytes = progress?.bytes_total
    ? `${formatBytes(progress.bytes_done ?? 0)} / ${formatBytes(progress.bytes_total)}`
    : null

  return (
    <div className="onboarding-wait-block">
      <div className="onboarding-load-summary">
        <div className="onboarding-load-metric">
          <span>{t('onboarding.loading.currentStatus')}</span>
          <strong title={message}>{message}</strong>
        </div>
        <div className="onboarding-load-metric">
          <span>{t('onboarding.loading.elapsed')}</span>
          <strong>{formatElapsedTime(elapsedMs)}</strong>
        </div>
      </div>
      <div className="onboarding-load-progress-row">
        <span>{t('onboarding.loading.progress')}</span>
        <span>{percent}%</span>
      </div>
      <Progress value={percent} />
      {files || progress?.file_name || totalBytes ? (
        <div className="onboarding-load-details">
          {files ? <span>{t('onboarding.loading.files')}: {files}</span> : null}
          {progress?.file_name ? (
            <span title={progress.file_name}>
              {t('onboarding.loading.currentFile')}: {progress.file_name}
              {fileBytes ? ` - ${fileBytes}` : ''}
            </span>
          ) : null}
          {totalBytes ? <span>{t('onboarding.loading.totalDownload')}: {totalBytes}</span> : null}
        </div>
      ) : null}
      <ModelLoadingStages percent={percent} />
      <ModelLoadingLessons />
    </div>
  )
}

function ModelLoadingStages({ percent }: { percent: number }) {
  const { t } = useI18n()
  return (
    <div className="onboarding-stage-list" aria-label={t('onboarding.loading.stages')}>
      <div className="onboarding-stage-list-header">
        <ListChecks size={14} />
        <span>{t('onboarding.loading.stages')}</span>
      </div>
      {MODEL_LOADING_STAGES.map((stage, index) => {
        const nextStage = MODEL_LOADING_STAGES[index + 1]
        const complete = percent >= (nextStage?.start ?? 100)
        const current = !complete && percent >= stage.start
        return (
          <div
            key={stage.key}
            className={`onboarding-stage-item${
              complete
                ? ' onboarding-stage-item-complete'
                : current
                  ? ' onboarding-stage-item-current'
                  : ''
            }`}
          >
            <span className="onboarding-stage-marker" />
            <span>{t(`onboarding.loading.stage.${stage.key}`)}</span>
          </div>
        )
      })}
    </div>
  )
}

function useElapsedMilliseconds(active: boolean): number {
  const startedAtRef = useRef<number | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    if (!active) {
      startedAtRef.current = null
      return
    }
    const start = Date.now()
    startedAtRef.current = start
    const timer = window.setInterval(() => setElapsedMs(Date.now() - start), 1000)
    return () => window.clearInterval(timer)
  }, [active])

  return active ? elapsedMs : 0
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
  t: (key: string) => string,
): string {
  if (step.id !== ONBOARDING_STEP_LOAD_MODEL) {
    return step.primaryLabel
  }
  if (progress.modelLoaded) {
    return t('common.continue')
  }
  if (progress.modelLoading) {
    return t('onboarding.loadingModel')
  }
  if (modelLoadDisabled) {
    return t('onboarding.openSetup')
  }
  return modelSetupOpen ? step.primaryLabel : t('onboarding.loadRecommendedModel')
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
    return imageBounds ?? outpaintFrame
  }
  return null
}

function ModelLoadingLessons() {
  const { t } = useI18n()
  const [messageIndex, setMessageIndex] = useState(0)
  const prefersReducedMotion = useReducedMotion()
  const message = MODEL_LOADING_MESSAGES[messageIndex] ?? MODEL_LOADING_MESSAGES[0]
  const messageKey = loadingMessageKey(messageIndex)
  useEffect(() => {
    const timer = window.setInterval(() => {
      setMessageIndex((current) => (current + 1) % MODEL_LOADING_MESSAGES.length)
    }, MODEL_LOADING_MESSAGE_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <div className="onboarding-lesson-block" aria-live="polite">
      <div className="onboarding-message-header">
        <span>{t(`onboarding.lessons.${messageKey}.title`, {}, message.title)}</span>
        <span>{messageIndex + 1}/{MODEL_LOADING_MESSAGES.length}</span>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.p
          key={messageKey}
          className="onboarding-lesson-body"
          initial={prefersReducedMotion ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={prefersReducedMotion ? undefined : { opacity: 0, y: -6 }}
          transition={{ duration: 0.22 }}
        >
          {t(`onboarding.lessons.${messageKey}.body`, {}, message.body)}
        </motion.p>
      </AnimatePresence>
      <div className="onboarding-message-dots" aria-label={t('onboarding.loadingMessageProgress')}>
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

function formatElapsedTime(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes <= 0) {
    return `${seconds}s`
  }
  return `${minutes}m ${seconds.toString().padStart(2, '0')}s`
}

function formatBytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  for (const unit of units) {
    if (amount < 1024 || unit === units[units.length - 1]) {
      return unit === 'B' ? `${Math.round(amount)} B` : `${amount.toFixed(1)} ${unit}`
    }
    amount /= 1024
  }
  return `${amount.toFixed(1)} TB`
}

function loadingMessageKey(index: number): string {
  if (index === 1) {
    return 'context'
  }
  if (index === 2) {
    return 'faster'
  }
  if (index === 3) {
    return 'next'
  }
  return 'warming'
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

function getPopoverStyle(targetRect: TourTargetRect | null, expanded = false): CSSProperties {
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  const availableWidth = Math.max(0, viewportWidth - VIEWPORT_MARGIN * 2)
  const popoverWidth = Math.min(
    expanded ? POPOVER_WAITING_WIDTH : POPOVER_WIDTH,
    availableWidth,
  )
  if (!targetRect) {
    return {
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: popoverWidth,
    }
  }
  const rightSideLeft = targetRect.right + POPOVER_GAP
  const leftSideLeft = targetRect.left - popoverWidth - POPOVER_GAP
  const centeredLeft = targetRect.left + targetRect.width / 2 - popoverWidth / 2
  const left =
    rightSideLeft + popoverWidth + VIEWPORT_MARGIN <= viewportWidth
      ? rightSideLeft
      : leftSideLeft >= VIEWPORT_MARGIN
        ? leftSideLeft
        : clamp(centeredLeft, VIEWPORT_MARGIN, viewportWidth - popoverWidth - VIEWPORT_MARGIN)
  const estimatedHeight = expanded ? POPOVER_WAITING_ESTIMATED_HEIGHT : POPOVER_ESTIMATED_HEIGHT
  const preferredTop = targetRect.top + targetRect.height / 2 - estimatedHeight / 2
  return {
    top: clamp(preferredTop, VIEWPORT_MARGIN, viewportHeight - estimatedHeight - VIEWPORT_MARGIN),
    left,
    width: popoverWidth,
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
