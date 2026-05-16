import { Minus, Plus } from 'lucide-react'
import { useI18n } from '../i18n/useI18n'

interface NumberStepperProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  disabled?: boolean
  onChange: (value: number) => void
}

export function NumberStepper({
  label,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  onChange,
}: NumberStepperProps) {
  const { t } = useI18n()
  const canDecrease = !disabled && value > min
  const canIncrease = !disabled && value < max

  return (
    <div className="field-label number-field">
      <span>{label}</span>
      <div className="number-stepper">
        <button
          type="button"
          className="stepper-button"
          disabled={!canDecrease}
          aria-label={t('common.decreaseValue', { label })}
          onClick={() => onChange(clampNumber(roundForStep(value - step, step), min, max))}
        >
          <Minus size={13} />
        </button>
        <input
          className="number-input"
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          aria-label={label}
          onChange={(event) => onChange(clampNumber(Number(event.target.value), min, max))}
        />
        <button
          type="button"
          className="stepper-button"
          disabled={!canIncrease}
          aria-label={t('common.increaseValue', { label })}
          onClick={() => onChange(clampNumber(roundForStep(value + step, step), min, max))}
        >
          <Plus size={13} />
        </button>
      </div>
    </div>
  )
}

function roundForStep(value: number, step: number): number {
  const decimals = Math.max(0, String(step).split('.')[1]?.length ?? 0)
  return decimals > 0 ? Number(value.toFixed(decimals)) : value
}

function clampNumber(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) {
    return min
  }
  return Math.max(min, Math.min(max, value))
}
