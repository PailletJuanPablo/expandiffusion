import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { forwardRef, type ComponentPropsWithoutRef, type ElementRef } from 'react'
import { cn } from '../../lib/utils'

const buttonVariants = cva('ui-button', {
  variants: {
    variant: {
      primary: 'ui-button-primary',
      secondary: 'ui-button-secondary',
      ghost: 'ui-button-ghost',
      tool: 'ui-button-tool',
      toolActive: 'ui-button-tool ui-button-tool-active',
      accept: 'ui-button-accept',
      danger: 'ui-button-danger',
    },
    size: {
      default: 'ui-button-md',
      compact: 'ui-button-compact',
      large: 'ui-button-lg',
      icon: 'ui-button-icon',
      smallIcon: 'ui-button-small-icon',
    },
  },
  defaultVariants: {
    variant: 'secondary',
    size: 'default',
  },
})

export interface ButtonProps
  extends ComponentPropsWithoutRef<'button'>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export const Button = forwardRef<ElementRef<'button'>, ButtonProps>(
  ({ asChild = false, className, variant, size, ...props }, ref) => {
    const Component = asChild ? Slot : 'button'
    return (
      <Component
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    )
  },
)

Button.displayName = 'Button'
