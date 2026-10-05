import { useState } from 'react'
import { inspectionImageUrl } from '../api'
import { CLASSES, DEFECT_ORDER, TOOLS, ZONES, typeLabel } from '../config'
import type { Inspection, Verdict } from '../types'
import HeatmapCanvas from './HeatmapCanvas'

interface Props {
  item: Inspection
  fresh?: boolean
}

export default function ResultCard({ item, fresh }: Props) {
  const [open, setOpen] = useState(false)
  const defect = item.decision === 'defect'
  const color = defect && item.defect_type && item.defect_type in CLASSES ? CLASSES[item.defect_type as Verdict].color : undefined
  return (
    <article className={'rc card' + (defect ? ' defect' : ' normal') + (fresh ? ' fresh' : '')}>
      <HeatmapCanvas
        mapB64={item.map_b64}
        shape={item.map_shape}
        imageSrc={item.has_image ? inspectionImageUrl(item.id) : undefined}
        imageSize={item.image_size}
      />
      <div className="rc-body">
        <div className="rc-top">
          <span className={'pill ' + (defect ? 'defect' : 'normal')}>
            {defect ? '불량' : '정상'}
            {defect && item.defect_type && <> · {typeLabel(item.defect_type)}</>}
          </span>
          <span className="rc-id">#{item.id}</span>
        </div>
        <div className="rc-score" style={color ? { color } : undefined}>
          {item.mmr_score.toFixed(2)}
          <small>MMR 점수</small>
        </div>
        <div className="chip-row">
          <span className={'chip zone-' + item.zone}>{ZONES[item.zone].label}</span>
          <span className="chip">{item.timings.total_s.toFixed(1)}초</span>
          <span className="chip">{item.source === 'gallery' ? '샘플' : '업로드'}</span>
        </div>
        {item.sep_flag && <p className="notice warn small">참고·보정 이미지와 같은 파일이에요</p>}
        <p className="rc-tools">{item.tools.map((t) => TOOLS[t] ?? t).join(' → ')}</p>
        <p className="rc-time">{item.created_at}</p>
        <button className="link-btn" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          {open ? '에이전트 기록 접기' : '에이전트 기록 보기'}
        </button>
      </div>
      {open && (
        <div className="rc-detail fade-up">
          <p className="muted">{item.why}</p>
          {item.trace.map((s, i) => (
            <div className="tr" key={i}>
              <p className="eyebrow">
                {i + 1}단계 · {TOOLS[s.action] ?? s.action}
              </p>
              {s.thought && <div className="vlm-quote">“{s.thought}”</div>}
              {s.action !== 'decide' && <div className="obs">{s.observation}</div>}
            </div>
          ))}
          {defect && item.type_probs && (
            <div className="prob-list">
              {DEFECT_ORDER.map((k) => (
                <div className="prob-row" key={k}>
                  <span className="prob-name">{CLASSES[k].label}</span>
                  <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${(item.type_probs![k] ?? 0) * 100}%`, background: CLASSES[k].color }} />
                  </div>
                  <span className="prob-val">{Math.round((item.type_probs![k] ?? 0) * 100)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </article>
  )
}
