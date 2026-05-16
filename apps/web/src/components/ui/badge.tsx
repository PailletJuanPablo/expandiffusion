import { cva, type VariantProps } from 'class-variance-authority'
import type { ComponentPropsWithoutRef } from 'react'
import { cn } from '../../lib/utils'

const badgeVariants = cva('ui-badge', {
  variants: {
    variant: {
      default: 'ui-badge-default',
      ready: 'ui-badge-ready',
      muted: 'ui-badge-muted',
      warning: 'ui-badge-warning',
      error: 'ui-badge-error',
      accent: 'ui-badge-accent',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

export type BadgeProps = ComponentPropsWithoutRef<'span'> & VariantProps<typeof badgeVariants>

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
