import { describe, expect, it } from 'vitest'
import { parseLoras, parseTextualInversions } from './extensionParsers'

describe('extensionParsers', () => {
  it('parses LoRA paths and optional scales', () => {
    expect(parseLoras('a.safetensors | 0.75\n\nb.safetensors')).toEqual([
      { path: 'a.safetensors', scale: 0.75 },
      { path: 'b.safetensors', scale: 1 },
    ])
  })

  it('falls back to scale 1 for invalid LoRA scales', () => {
    expect(parseLoras('a.safetensors | nope')).toEqual([
      { path: 'a.safetensors', scale: 1 },
    ])
  })

  it('parses textual inversions with optional token', () => {
    expect(parseTextualInversions('style.bin | <style>\nplain.bin')).toEqual([
      { path: 'style.bin', token: '<style>' },
      { path: 'plain.bin', token: null },
    ])
  })
})
