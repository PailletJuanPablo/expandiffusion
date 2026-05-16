import { describe, expect, it } from 'vitest'
import {
  ONBOARDING_ACTION_FOCUS_PROMPT,
  ONBOARDING_STEPS,
  ONBOARDING_STEP_WELCOME,
  ONBOARDING_STEP_IMAGE_FOCUS,
  ONBOARDING_STEP_LOAD_MODEL,
  ONBOARDING_STEP_PREPARE_OUTPAINT,
  ONBOARDING_STEP_PROMPT,
  ONBOARDING_STEP_UPLOAD_IMAGE,
  ONBOARDING_TARGET_PROMPT_INPUT,
  getOnboardingDocumentTargetRect,
  getOnboardingStepState,
  getNextOnboardingStepIndex,
  shouldAutoAdvanceOnboardingStep,
} from './onboardingTour'

describe('onboardingTour', () => {
  it('keeps the tour on model loading until a model is ready', () => {
    const loadingStepIndex = 1
    const nextIndex = getNextOnboardingStepIndex(loadingStepIndex, {
      modelLoaded: false,
      modelLoading: false,
      imageLoaded: false,
      outpaintReady: false,
      generationStarted: false,
    })

    expect(nextIndex).toBe(loadingStepIndex)
    expect(
      getOnboardingStepState(ONBOARDING_STEP_LOAD_MODEL, {
        modelLoaded: false,
        modelLoading: true,
        imageLoaded: false,
        outpaintReady: false,
        generationStarted: false,
      }),
    ).toEqual({
      complete: false,
      waiting: true,
      canContinue: false,
    })
  })

  it('allows upload guidance only after the required model is loaded', () => {
    expect(
      getNextOnboardingStepIndex(1, {
        modelLoaded: true,
        modelLoading: false,
        imageLoaded: false,
        outpaintReady: false,
        generationStarted: false,
      }),
    ).toBe(2)
    expect(
      getOnboardingStepState(ONBOARDING_STEP_UPLOAD_IMAGE, {
        modelLoaded: true,
        modelLoading: false,
        imageLoaded: true,
        outpaintReady: false,
        generationStarted: false,
      }),
    ).toEqual({
      complete: true,
      waiting: false,
      canContinue: true,
    })
  })

  it('maps document image and frame bounds to screen spotlight coordinates', () => {
    expect(
      getOnboardingDocumentTargetRect(
        { x: 512, y: 128, width: 300, height: 200 },
        { x: 72, y: 48, zoom: 0.5 },
        { left: 100, top: 80 },
      ),
    ).toEqual({
      left: 428,
      top: 192,
      right: 578,
      bottom: 292,
      width: 150,
      height: 100,
    })
  })

  it('targets the prompt input when the tour reaches prompt guidance', () => {
    const promptStep = ONBOARDING_STEPS.find((step) => step.id === ONBOARDING_STEP_PROMPT)

    expect(promptStep).toMatchObject({
      targetId: ONBOARDING_TARGET_PROMPT_INPUT,
      primaryAction: ONBOARDING_ACTION_FOCUS_PROMPT,
    })
  })

  it('opens with a diffusion-model welcome before setup steps', () => {
    const welcomeStep = ONBOARDING_STEPS.find((step) => step.id === ONBOARDING_STEP_WELCOME)

    expect(welcomeStep).toMatchObject({
      kicker: 'Welcome',
      title: 'Explore what diffusion models can imagine',
    })
    expect(welcomeStep?.body).toContain('image generation with diffusion models')
    expect(welcomeStep?.body).toContain('warm up a model')
    expect(welcomeStep?.body).not.toContain('Expandiffusion')
  })

  it('auto-advances the upload step once a real image is present', () => {
    expect(
      shouldAutoAdvanceOnboardingStep(ONBOARDING_STEP_UPLOAD_IMAGE, {
        modelLoaded: true,
        modelLoading: false,
        imageLoaded: true,
        outpaintReady: false,
        generationStarted: false,
      }),
    ).toBe(true)
    expect(
      shouldAutoAdvanceOnboardingStep(ONBOARDING_STEP_UPLOAD_IMAGE, {
        modelLoaded: true,
        modelLoading: false,
        imageLoaded: false,
        outpaintReady: false,
        generationStarted: false,
      }),
    ).toBe(false)
  })

  it('uses canvas and frame copy for the post-upload guidance', () => {
    const canvasStep = ONBOARDING_STEPS.find((step) => step.id === ONBOARDING_STEP_IMAGE_FOCUS)
    const frameStep = ONBOARDING_STEPS.find((step) => step.id === ONBOARDING_STEP_PREPARE_OUTPAINT)

    expect(canvasStep).toMatchObject({
      title: 'Now the image becomes editable space',
      body: 'The canvas is where generations are previewed and composed. Hold Shift and drag anytime to pan around without switching tools.',
    })
    expect(frameStep?.body).toContain('overlap part of the original')
  })
})
