import { forwardRef, type ComponentPropsWithoutRef } from 'react'
import { cn } from '../../lib/utils'

export const Input = forwardRef<HTMLInputElement, ComponentPropsWithoutRef<'input'>>(
  ({ className, ...props }, ref) => (
    <input ref={ref} className={cn('ui-input', className)} {...props} />
  ),
)

Input.displayName = 'Input'
