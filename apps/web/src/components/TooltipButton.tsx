import type { CSSProperties, ReactNode } from 'react'
import { Button } from './ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'

interface TooltipButtonProps {
  label: string
  active?: boolean
  disabled?: boolean
  className?: string
  style?: CSSProperties
  children: ReactNode
  onClick: () => void
}

export function TooltipButton({
  label,
  active = false,
  disabled = false,
  className,
  style,
  children,
  onClick,
}: TooltipButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          aria-label={label}
          variant={active ? 'toolActive' : 'tool'}
          size="icon"
          className={className}
          style={style}
          disabled={disabled}
          onClick={onClick}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent side="right" sideOffset={10}>
          {label}
      </TooltipContent>
    </Tooltip>
  )
}
