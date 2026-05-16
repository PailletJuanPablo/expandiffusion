type ShortcutTarget = {
  tagName?: string
  isContentEditable?: boolean
  closest?: (selector: string) => unknown
}

export function isEditableShortcutTarget(target: unknown): boolean {
  if (!isShortcutTarget(target)) {
    return false
  }

  if (target.isContentEditable) {
    return true
  }

  const tagName = target.tagName?.toLowerCase()
  if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
    return true
  }

  return Boolean(target.closest?.('input, textarea, select'))
}

function isShortcutTarget(target: unknown): target is ShortcutTarget {
  return typeof target === 'object' && target !== null
}
