import { CLASSES } from '../config'
import { inspect } from '../lib/decide'
import type { Sample } from '../types'

interface Props {
  samples: Sample[]
  selectedId: string
  onSelect: (id: string) => void
}

const STATUS_COLOR = { normal: CLASSES.normal.color, defect: '#f04452', hold: '#ff9f1c' } as const

export default function SamplePicker({ samples, selectedId, onSelect }: Props) {
  return (
    <div className="picker" role="tablist" aria-label="검사 샘플 선택">
      {samples.map((s) => {
        const r = inspect(s)
        const active = s.id === selectedId
        return (
          <button
            key={s.id}
            role="tab"
            aria-selected={active}
            className={'picker-item' + (active ? ' active' : '')}
            onClick={() => onSelect(s.id)}
          >
            <span className="picker-dot" style={{ background: STATUS_COLOR[r.decision] }} />
            <span className="picker-id">{s.id}</span>
            <span className="picker-note">{s.note}</span>
          </button>
        )
      })}
    </div>
  )
}
