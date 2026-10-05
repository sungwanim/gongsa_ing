import { useEffect, useState } from 'react'
import { CLASSES, DEFECT_ORDER, TOOLS, ZONES, typeLabel } from '../config'
import type { Verdict } from '../types'
import type { RunState } from '../runState'
import HeatmapCanvas from './HeatmapCanvas'
import { IconAlert, IconCheck } from './Icons'

export default function RunView({ run }: { run: RunState }) {
  const [sec, setSec] = useState(0)
  useEffect(() => {
    if (run.final || run.error) return
    const t0 = run.startedAt
    const id = setInterval(() => setSec((Date.now() - t0) / 1000), 200)
    return () => clearInterval(id)
  }, [run.startedAt, run.final, run.error])

  const total = run.final?.timings.total_s ?? sec
  return (
    <div className="run card fade-up" aria-live="polite">
      <div className="run-head">
        <div>
          <p className="eyebrow">검사 진행 중</p>
          <h3 className="card-title">{run.final ? '검사를 마쳤어요' : run.error ? '검사에 실패했어요' : '에이전트가 이미지를 살펴보고 있어요'}</h3>
        </div>
        <span className="timer">{total.toFixed(1)}초</span>
      </div>

      {run.warning && <div className="notice warn">{run.warning}</div>}
      {run.error && <div className="notice err">{run.error}</div>}

      <ol className="timeline">
        <li className={'tl ' + (run.mmr ? 'done' : 'active')}>
          <div className="tl-mark">{run.mmr ? <IconCheck width={16} height={16} /> : <span className="spin" />}</div>
          <div className="tl-body">
            <p className="eyebrow">1차 · MMR</p>
            {run.mmr ? (
              <div className="mmr-row">
                <HeatmapCanvas mapB64={run.mmr.map_b64} shape={run.mmr.map_shape} large imageSrc={run.preview} imageSize={run.size} />
                <div className="mmr-facts">
                  <span className="big-number">{run.mmr.score.toFixed(2)}</span>
                  <span className="gauge-unit">MMR 이상 점수</span>
                  <div className="chip-row">
                    <span className={'chip zone-' + run.mmr.zone}>{ZONES[run.mmr.zone].label}</span>
                    <span className="chip">{run.mmr.sec.toFixed(1)}초</span>
                  </div>
                  <p className="muted">{ZONES[run.mmr.zone].hint}</p>
                </div>
              </div>
            ) : (
              <p className="muted">MMR이 이상 맵을 계산하고 있어요…</p>
            )}
          </div>
        </li>

        {run.steps.map((s, i) => (
          <li key={i} className={'tl ' + (s.running ? 'active' : 'done')}>
            <div className="tl-mark">{s.running ? <span className="spin" /> : <IconCheck width={16} height={16} />}</div>
            <div className="tl-body">
              <p className="eyebrow">에이전트 · {i + 1}단계</p>
              {s.thought && <div className="vlm-quote">“{s.thought}”</div>}
              {s.tool && (
                <div className="chip-row">
                  <span className="chip primary">선택 · {TOOLS[s.tool] ?? s.tool}</span>
                  {s.choiceProbs &&
                    Object.entries(s.choiceProbs)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([k, v]) => (
                        <span key={k} className="chip">
                          {TOOLS[k] ?? k} {Math.round(v * 100)}%
                        </span>
                      ))}
                </div>
              )}
              {s.running && <p className="muted">{TOOLS[s.tool ?? ''] ?? '도구'}를 실행하는 중이에요…</p>}
              {s.observation && (
                <div className="obs">
                  {s.observation}
                  {s.sec !== undefined && <em> · {s.sec.toFixed(1)}초</em>}
                </div>
              )}
            </div>
          </li>
        ))}

        {run.final && (
          <li className={'tl final ' + run.final.decision}>
            <div className="tl-mark">{run.final.decision === 'defect' ? <IconAlert width={16} height={16} /> : <IconCheck width={16} height={16} />}</div>
            <div className="tl-body">
              <p className="eyebrow">최종 판정</p>
              <h3 className="verdict-title">
                {run.final.decision === 'defect' ? '불량' : '정상'}
                {run.final.decision === 'defect' && run.final.defect_type && (
                  <span style={{ color: CLASSES[(run.final.defect_type as Verdict) in CLASSES ? (run.final.defect_type as Verdict) : 'breakdown'].color }}>
                    {' · '}
                    {typeLabel(run.final.defect_type)}
                  </span>
                )}
              </h3>
              <p className="muted">{run.final.why}</p>
              {run.final.type_probs && run.final.decision === 'defect' && (
                <div className="prob-list">
                  {DEFECT_ORDER.map((k) => (
                    <div className="prob-row" key={k}>
                      <span className="prob-name">{CLASSES[k].label}</span>
                      <div className="prob-track">
                        <div className="prob-fill" style={{ width: `${(run.final!.type_probs![k] ?? 0) * 100}%`, background: CLASSES[k].color }} />
                      </div>
                      <span className="prob-val">{Math.round((run.final!.type_probs![k] ?? 0) * 100)}%</span>
                    </div>
                  ))}
                </div>
              )}
              <p className="hint">대시보드에 추가하고 있어요…</p>
            </div>
          </li>
        )}
      </ol>
    </div>
  )
}
