import { describe, expect, it } from 'vitest'
import { isEditableShortcutTarget } from './keyboardTargets'

describe('keyboardTargets', () => {
  it('treats text entry controls as editable shortcut targets', () => {
    expect(isEditableShortcutTarget({ tagName: 'INPUT' })).toBe(true)
    expect(isEditableShortcutTarget({ tagName: 'textarea' })).toBe(true)
    expect(isEditableShortcutTarget({ tagName: 'select' })).toBe(true)
  })

  it('treats contenteditable targets as editable shortcut targets', () => {
    expect(isEditableShortcutTarget({ isContentEditable: true })).toBe(true)
  })

  it('ignores non-editable shortcut targets', () => {
    expect(isEditableShortcutTarget({ tagName: 'button' })).toBe(false)
    expect(isEditableShortcutTarget(null)).toBe(false)
  })

  it('uses editable ancestors for nested shortcut targets', () => {
    expect(
      isEditableShortcutTarget({
        tagName: 'span',
        closest: () => ({ tagName: 'textarea' }),
      }),
    ).toBe(true)
  })
})
