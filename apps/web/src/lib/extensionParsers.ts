import type { LoraConfig, TextualInversionConfig } from '../domain/types'

const EXTENSION_PART_SEPARATOR = '|'
const DEFAULT_LORA_SCALE = 1

/**
 * Parse newline-delimited LoRA references.
 *
 * @param value - Textarea value using `path | scale` lines.
 * @returns Normalized LoRA configs.
 */
export function parseLoras(value: string): LoraConfig[] {
  return value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseLoraLine)
    .filter((item) => Boolean(item.path))
}

/**
 * Parse newline-delimited textual inversion references.
 *
 * @param value - Textarea value using `path | token` lines.
 * @returns Normalized textual inversion configs.
 */
export function parseTextualInversions(value: string): TextualInversionConfig[] {
  return value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseTextualInversionLine)
    .filter((item) => Boolean(item.path))
}

function parseLoraLine(line: string): LoraConfig {
  const parts = splitExtensionLine(line)
  const rawScale = parts[1] ?? ''
  const scale = rawScale ? Number(rawScale) : DEFAULT_LORA_SCALE
  return {
    path: parts[0] ?? '',
    scale: Number.isNaN(scale) ? DEFAULT_LORA_SCALE : scale,
  }
}

function parseTextualInversionLine(line: string): TextualInversionConfig {
  const parts = splitExtensionLine(line)
  const token = parts[1] ?? ''
  return {
    path: parts[0] ?? '',
    token: token || null,
  }
}

function splitExtensionLine(line: string): string[] {
  return line.split(EXTENSION_PART_SEPARATOR).map((part) => part.trim())
}
