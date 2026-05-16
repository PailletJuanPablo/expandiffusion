import type { TranslateFunction } from '../i18n/i18n'

export interface ControlOptionDetail {
  title: string
  badge?: string
  description: string
}

interface ControlHelp {
  description: string
  optionDetails?: Record<string, ControlOptionDetail>
}

const CONTROL_HELP: Record<string, ControlHelp> = {
  prompt: {
    description: 'Describe what should appear in the generated area.',
  },
  negative_prompt: {
    description: 'Optional things to avoid, such as artifacts, blur, or unwanted objects.',
  },
  steps: {
    description: 'More steps can improve quality but every sample takes longer.',
  },
  sample_count: {
    description: 'Creates multiple alternatives for the same settings.',
  },
  guidance_scale: {
    description: 'Controls how strongly the prompt steers the result.',
  },
  strength: {
    description: 'Higher values allow stronger changes to the selected or masked area.',
  },
  scheduler: {
    description: 'Sampling method used by the loaded pipeline.',
    optionDetails: {
      auto: {
        title: 'Auto',
        badge: 'Default',
        description: 'Use the scheduler configured by the adapter.',
      },
      dpmpp_2m: {
        title: 'DPM++ 2M',
        description: 'Balanced quality option for common SD checkpoints.',
      },
      euler: {
        title: 'Euler',
        description: 'Simple sampler that is useful for quick comparisons.',
      },
      ddim: {
        title: 'DDIM',
        description: 'Deterministic sampler for stable prompt experiments.',
      },
      lms: {
        title: 'LMS',
        description: 'Older sampler kept for model compatibility.',
      },
    },
  },
  fill_mode: {
    description: 'Prepares empty pixels before the model fills them.',
  },
  inpaint_area: {
    description: 'Chooses whether inpaint sees the full selected image or a crop around the mask.',
    optionDetails: {
      whole_selection: {
        title: 'Whole selection',
        description: 'Use the full selected image as context for the masked edit.',
      },
      only_masked: {
        title: 'Only masked crop',
        badge: 'Lower VRAM',
        description: 'Crop around the mask before generation to reduce memory use.',
      },
    },
  },
  mask_crop_padding: {
    description: 'Extra pixels kept around a masked crop when only the mask area is processed.',
  },
  mask_blur: {
    description: 'Softens mask edges so the edit blends into nearby pixels.',
  },
  outpaint_max_width: {
    description: 'Maximum working width before the outpaint input is resized.',
  },
  outpaint_max_height: {
    description: 'Maximum working height before the outpaint input is resized.',
  },
  result_mode: {
    description: 'Controls how generated pixels are composed back onto the canvas.',
    optionDetails: {
      generated_selection: {
        title: 'Generated selection',
        description: 'Return the raw generated selection.',
      },
      preserve_known: {
        title: 'Preserve known',
        badge: 'Safer',
        description: 'Keep original known pixels and replace only generated areas.',
      },
      feather_known: {
        title: 'Feather known',
        description: 'Blend original known pixels softly into the generated result.',
      },
      restore_original_soft: {
        title: 'Restore original soft',
        description: 'Restore source pixels with a softer mask edge.',
      },
    },
  },
  random_seed: {
    description: 'When enabled, each run uses a new seed.',
  },
  seed: {
    description: 'Fixed seed used when random seed is off.',
  },
  img2img: {
    description: 'Uses the current image as stronger visual guidance when the adapter supports it.',
  },
  controlnet_model_id: {
    description: 'ControlNet model used for sketch or tile guidance.',
  },
  controlnet_conditioning_scale: {
    description: 'How much the sketch or guide image influences generation.',
  },
  control_guidance_start: {
    description: 'Generation progress where ControlNet guidance starts.',
  },
  control_guidance_end: {
    description: 'Generation progress where ControlNet guidance stops.',
  },
  outpaint_strategy: {
    description: 'Chooses how the frame is prepared before generation.',
    optionDetails: {
      hf_space_fill: {
        title: 'HF Space fill',
        badge: 'Recommended',
        description: 'SDXL fill workflow for expanding an image with overlap controls.',
      },
      directional: {
        title: 'Directional SDXL',
        description: 'One-side expansion with explicit direction, generated size, and context.',
      },
      local_context: {
        title: 'Local context',
        description: 'Use nearby visible pixels around the frame as context.',
      },
      full_context_crop: {
        title: 'Full context crop',
        description: 'Crop a larger context region around the selected frame.',
      },
      whole_resized: {
        title: 'Whole resized',
        description: 'Resize the whole visible canvas into the model input.',
      },
      selected_frame: {
        title: 'Selected frame',
        description: 'Generate only the selected frame without adding extra context.',
      },
    },
  },
  outpaint_direction: {
    description: 'Side of the image that should be extended.',
    optionDetails: {
      right: {
        title: 'Right',
        description: 'Add new content to the right side.',
      },
      left: {
        title: 'Left',
        description: 'Add new content to the left side.',
      },
      down: {
        title: 'Down',
        description: 'Add new content below the image.',
      },
      up: {
        title: 'Up',
        description: 'Add new content above the image.',
      },
      around: {
        title: 'Around',
        description: 'Expand all sides in one pass.',
      },
    },
  },
  hf_space_resize_option: {
    description: 'Scale used by the HF Space fill preprocessing step.',
    optionDetails: {
      Full: {
        title: 'Full',
        description: 'Use the current size without downscaling first.',
      },
      '50%': {
        title: '50%',
        description: 'Generate at half size to reduce memory and time.',
      },
      '33%': {
        title: '33%',
        description: 'Generate at one third size for very large inputs.',
      },
      '25%': {
        title: '25%',
        description: 'Generate at quarter size for the lowest memory use.',
      },
      Custom: {
        title: 'Custom',
        description: 'Use the custom resize percentage field below.',
      },
    },
  },
  hf_space_custom_resize_percentage: {
    description: 'Custom HF Space resize percentage used only when resize is set to Custom.',
  },
  hf_space_overlap_percentage: {
    description: 'Amount of existing image overlapped into the generated region.',
  },
  hf_space_overlap_left: {
    description: 'Allow overlap on the left edge of the generated area.',
  },
  hf_space_overlap_right: {
    description: 'Allow overlap on the right edge of the generated area.',
  },
  hf_space_overlap_top: {
    description: 'Allow overlap on the top edge of the generated area.',
  },
  hf_space_overlap_bottom: {
    description: 'Allow overlap on the bottom edge of the generated area.',
  },
  visual_refine_enabled: {
    description: 'Optional second pass with IP-Adapter. Leave off unless you need extra harmonization.',
  },
  visual_refine_strength: {
    description: 'How strongly the second pass can alter the first outpaint result.',
  },
  ip_adapter_scale: {
    description: 'How strongly the reference crop influences the visual refine pass.',
  },
  visual_refine_steps: {
    description: 'Number of steps used only by the visual refine pass.',
  },
  visual_refine_reference: {
    description: 'Reference crop used by IP-Adapter during visual refine.',
    optionDetails: {
      near_edge: {
        title: 'Near edge',
        description: 'Use known pixels close to the generated border.',
      },
      visible_source: {
        title: 'Visible source',
        description: 'Use the visible source image as the style reference.',
      },
    },
  },
}

export function controlHelpFor(controlId: string, t?: TranslateFunction): string | null {
  const help = CONTROL_HELP[controlId]
  if (!help) {
    return null
  }
  return t ? t(`controlHelp.${controlId}.description`, {}, help.description) : help.description
}

export function controlOptionDetailsFor(
  controlId: string,
  t?: TranslateFunction,
): Record<string, ControlOptionDetail> | undefined {
  const details = CONTROL_HELP[controlId]?.optionDetails
  if (!details || !t) {
    return details
  }
  return Object.fromEntries(
    Object.entries(details).map(([id, detail]) => [
      id,
      {
        ...detail,
        title: t(`controlHelp.${controlId}.options.${id}.title`, {}, detail.title),
        badge: detail.badge
          ? t(`controlHelp.${controlId}.options.${id}.badge`, {}, detail.badge)
          : detail.badge,
        description: t(
          `controlHelp.${controlId}.options.${id}.description`,
          {},
          detail.description,
        ),
      },
    ]),
  )
}
