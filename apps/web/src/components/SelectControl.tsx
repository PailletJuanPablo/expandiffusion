import * as Select from '@radix-ui/react-select'
import { Check, ChevronDown } from 'lucide-react'
import type { ControlOption } from '../domain/types'

export interface SelectOptionDetail {
  title: string
  badge?: string
  description?: string
}

interface SelectControlProps {
  label: string
  value: string
  options: ControlOption[]
  disabled?: boolean
  optionDetails?: Record<string, SelectOptionDetail>
  showSelectedDescription?: boolean
  onChange: (value: string) => void
}

export function SelectControl({
  label,
  value,
  options,
  disabled = false,
  optionDetails,
  showSelectedDescription = true,
  onChange,
}: SelectControlProps) {
  const selectedDetail = optionDetails?.[value]
  return (
    <div className="field-label select-field">
      <span>{label}</span>
      <Select.Root value={value} disabled={disabled} onValueChange={onChange}>
        <Select.Trigger className="select-trigger" aria-label={label}>
          <Select.Value />
          <Select.Icon asChild>
            <ChevronDown size={14} />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            className="select-content"
            position="popper"
            sideOffset={5}
            collisionPadding={10}
          >
            <Select.Viewport className="select-viewport">
              {options.map((option) => {
                const detail = optionDetails?.[option.id]
                return (
                  <Select.Item
                    key={option.id}
                    value={option.id}
                    className={detail ? 'select-item select-item-rich' : 'select-item'}
                  >
                    <Select.ItemText>
                      <span className="select-item-title">{detail?.title ?? option.label}</span>
                    </Select.ItemText>
                    {detail?.badge ? (
                      <span className="select-item-badge">{detail.badge}</span>
                    ) : null}
                    {detail?.description ? (
                      <span className="select-item-description">{detail.description}</span>
                    ) : null}
                    <Select.ItemIndicator className="select-item-indicator">
                      <Check size={13} />
                    </Select.ItemIndicator>
                  </Select.Item>
                )
              })}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
      {showSelectedDescription && selectedDetail?.description ? (
        <span className="select-selected-description">
          {selectedDetail.description}
        </span>
      ) : null}
    </div>
  )
}
