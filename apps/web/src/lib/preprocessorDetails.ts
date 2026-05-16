import {
  FILL_EDGE_EXTEND,
  FILL_GAUSSIAN_NOISE,
  FILL_OPENCV_NS,
  FILL_OPENCV_TELEA,
  FILL_PATCHMATCH,
  FILL_PERLIN_NOISE,
  FILL_TRANSPARENT,
} from '../constants/domain'

export interface PreprocessorDetails {
  title: string
  badge: string
  description: string
  bestFor: string
  caution: string
}

export const PREPROCESSOR_DETAILS: Record<string, PreprocessorDetails> = {
  [FILL_PATCHMATCH]: {
    title: 'PatchMatch',
    badge: 'Best default',
    description: 'Copies similar image patches into the empty area.',
    bestFor: 'Textured backgrounds, clothes, walls, foliage, ground, and skies.',
    caution: 'Can repeat patterns or clone unwanted details.',
  },
  [FILL_EDGE_EXTEND]: {
    title: 'Edge extend',
    badge: 'Simple edges',
    description: 'Stretches border pixels outward from the existing image.',
    bestFor: 'Plain skies, walls, gradients, and borders without key objects.',
    caution: 'Can create bands or smears near high-contrast edges.',
  },
  [FILL_OPENCV_NS]: {
    title: 'Navier-Stokes',
    badge: 'Small repairs',
    description: 'OpenCV inpainting that continues colors and structures.',
    bestFor: 'Small gaps, seams, scratches, and thin reconstruction zones.',
    caution: 'Large outpaints can become too smooth or warped.',
  },
  [FILL_OPENCV_TELEA]: {
    title: 'Telea',
    badge: 'Smooth fixes',
    description: 'Fast-marching OpenCV inpainting for quick transitions.',
    bestFor: 'Small or medium masks, soft transitions, and cleanup passes.',
    caution: 'Large areas can look blurry or plastic.',
  },
  [FILL_PERLIN_NOISE]: {
    title: 'Perlin noise',
    badge: 'Organic fill',
    description: 'Soft low-frequency noise that lets the model invent more.',
    bestFor: 'Landscapes, clouds, rocks, foliage, and abstract backgrounds.',
    caution: 'Less continuity with the original lighting or palette.',
  },
  [FILL_GAUSSIAN_NOISE]: {
    title: 'Gaussian noise',
    badge: 'Most freedom',
    description: 'Random grain for high-denoise generative exploration.',
    bestFor: 'High denoising when the model should freely rebuild the area.',
    caution: 'More unstable; may leave noise, seams, or weaker composition.',
  },
  [FILL_TRANSPARENT]: {
    title: 'Transparent alpha',
    badge: 'PNG workflow',
    description: 'Keeps the empty area alpha-based when the pipeline supports it.',
    bestFor: 'PNG or alpha workflows designed around transparent masks.',
    caution: 'Some pipelines convert alpha to black, white, or gray before VAE.',
  },
}

/**
 * Resolve UI help for a fill preprocessor id.
 *
 * @param id - Technical fill mode id.
 * @returns Preprocessor details or null for adapter-provided custom modes.
 */
export function preprocessorDetailsFor(id: string): PreprocessorDetails | null {
  return PREPROCESSOR_DETAILS[id] ?? null
}
