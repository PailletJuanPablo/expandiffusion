import {
  CONTROL_NUMBER,
  CONTROL_SELECT,
  CONTROL_SLIDER,
  CONTROL_SWITCH,
  CONTROL_TEXTAREA,
} from '../constants/domain'
import type { ControlSchema } from '../domain/types'
import { NumberStepper } from './NumberStepper'
import { SelectControl } from './SelectControl'
import {
  PREPROCESSOR_DETAILS,
  type PreprocessorDetails,
  preprocessorDetailsFor,
} from '../lib/preprocessorDetails'
import { Input } from './ui/input'
import { Textarea } from './ui/textarea'

interface SchemaControlProps {
  control: ControlSchema
  value: unknown
  disabled?: boolean
  onChange: (id: string, value: unknown) => void
}

/**
 * Render a generic control described by adapter metadata.
 *
 * @param props - Schema, current value and change handler.
 * @returns Form control element.
 */
export function SchemaControl({
  control,
  value,
  disabled = false,
  onChange,
}: SchemaControlProps) {
  if (control.kind === CONTROL_SELECT) {
    const selectedValue = stringValue(value, control.default_value)
    if (control.id === 'fill_mode') {
      return (
        <div className="preprocessor-control">
          <SelectControl
            label={control.label}
            value={selectedValue}
            options={control.options}
            disabled={disabled}
            optionDetails={PREPROCESSOR_DETAILS}
            onChange={(nextValue) => onChange(control.id, nextValue)}
          />
          <PreprocessorSummary details={preprocessorDetailsFor(selectedValue)} />
        </div>
      )
    }

    return (
      <SelectControl
        label={control.label}
        value={selectedValue}
        options={control.options}
        disabled={disabled}
        onChange={(nextValue) => onChange(control.id, nextValue)}
      />
    )
  }

  if (control.kind === CONTROL_SLIDER) {
    const step = numericDefault(control.step, 1)
    const numberValue = numericValue(value, control.default_value)
    return (
      <label className="field-label">
        <span className="label-row">
          {control.label}
          <strong>{formatSliderValue(numberValue, step)}</strong>
        </span>
        <input
          className="range-input"
          type="range"
          min={numericDefault(control.min, 0)}
          max={numericDefault(control.max, 100)}
          step={step}
          value={numberValue}
          disabled={disabled}
          onChange={(event) => onChange(control.id, Number(event.target.value))}
        />
      </label>
    )
  }

  if (control.kind === CONTROL_NUMBER) {
    const min = numericDefault(control.min, 0)
    const max = numericDefault(control.max, 100)
    return (
      <NumberStepper
        label={control.label}
        value={numericValue(value, control.default_value)}
        min={min}
        max={max}
        step={numericDefault(control.step, 1)}
        disabled={disabled}
        onChange={(nextValue) => onChange(control.id, nextValue)}
      />
    )
  }

  if (control.kind === CONTROL_SWITCH) {
    return (
      <div className="switch-row">
        <span>{control.label}</span>
        <button
          type="button"
          className={value ? 'switch-root switch-root-checked' : 'switch-root'}
          disabled={disabled}
          onClick={() => onChange(control.id, !value)}
        >
          <span className={value ? 'switch-thumb switch-thumb-checked' : 'switch-thumb'} />
        </button>
      </div>
    )
  }

  if (control.kind === CONTROL_TEXTAREA) {
    return (
      <label className="field-label">
        {control.label}
        <Textarea
          value={stringValue(value, control.default_value)}
          disabled={disabled}
          rows={control.rows ?? 3}
          placeholder={control.placeholder ?? undefined}
          onChange={(event) => onChange(control.id, event.target.value)}
        />
      </label>
    )
  }

  return (
    <label className="field-label">
      {control.label}
      <Input
        value={stringValue(value, control.default_value)}
        disabled={disabled}
        placeholder={control.placeholder ?? undefined}
        onChange={(event) => onChange(control.id, event.target.value)}
      />
    </label>
  )
}

function PreprocessorSummary({
  details,
}: {
  details: PreprocessorDetails | null
}) {
  if (!details) {
    return null
  }

  return (
    <div className="preprocessor-summary">
      <div className="preprocessor-summary-heading">
        <strong>{details.title}</strong>
        <span>{details.badge}</span>
      </div>
      <p>{details.description}</p>
      <div className="preprocessor-summary-grid">
        <div>
          <span>Best for</span>
          <strong>{details.bestFor}</strong>
        </div>
        <div>
          <span>Watch</span>
          <strong>{details.caution}</strong>
        </div>
      </div>
    </div>
  )
}

function stringValue(value: unknown, fallback: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (typeof fallback === 'string') {
    return fallback
  }
  return ''
}

function numericValue(value: unknown, fallback: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof fallback === 'number' && Number.isFinite(fallback)) {
    return fallback
  }
  return 0
}

function numericDefault(value: number | null, fallback: number): number {
  return value ?? fallback
}

function formatSliderValue(value: number, step: number): string {
  if (Number.isInteger(step)) {
    return String(Math.round(value))
  }
  return value.toFixed(step < 0.1 ? 2 : 1)
}
