import { CLASSES, SCORE_AXIS } from '../config'
import { inspect } from '../lib/decide'
import type { Sample } from '../types'

interface Props {
  samples: Sample[]
  selectedId: string
  onSelect: (id: string) => void
}

const LABEL = { normal: '정상', defect: '불량', hold: '확인 필요' } as const

export default function HistoryTable({ samples, selectedId, onSelect }: Props) {
  return (
    <div className="card table-card">
      <table className="table">
        <thead>
          <tr>
            <th>블레이드</th>
            <th className="hide-sm">촬영 시각</th>
            <th>MMR 점수</th>
            <th className="hide-sm">검사 단계</th>
            <th>판정</th>
          </tr>
        </thead>
        <tbody>
          {samples.map((s) => {
            const r = inspect(s)
            const t = r.defectType ? CLASSES[r.defectType] : null
            return (
              <tr
                key={s.id}
                className={s.id === selectedId ? 'selected' : ''}
                onClick={() => {
                  onSelect(s.id)
                  document.getElementById('inspect')?.scrollIntoView({ behavior: 'smooth' })
                }}
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && onSelect(s.id)}
              >
                <td>
                  <b>{s.id}</b>
                </td>
                <td className="hide-sm muted">{s.capturedAt}</td>
                <td>
                  <div className="mini">
                    <div className="mini-track">
                      <div
                        className={'mini-fill ' + r.decision}
                        style={{ width: `${((s.mmrScore - SCORE_AXIS.min) / (SCORE_AXIS.max - SCORE_AXIS.min)) * 100}%` }}
                      />
                    </div>
                    <span>{s.mmrScore.toFixed(2)}</span>
                  </div>
                </td>
                <td className="hide-sm muted">{r.usedVlm ? 'MMR + Qwen' : 'MMR만'}</td>
                <td>
                  <span className={'pill ' + r.decision}>
                    {LABEL[r.decision]}
                    {t && ` · ${t.label}`}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
