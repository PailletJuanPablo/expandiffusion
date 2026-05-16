import { forwardRef, type ComponentPropsWithoutRef } from 'react'
import { cn } from '../../lib/utils'

export const Textarea = forwardRef<HTMLTextAreaElement, ComponentPropsWithoutRef<'textarea'>>(
  ({ className, ...props }, ref) => (
    <textarea ref={ref} className={cn('ui-textarea', className)} {...props} />
  ),
)

Textarea.displayName = 'Textarea'
