import { BAND, INNER, MMR, SCORE_AXIS } from '../config'
import { useCountUp } from '../lib/useCountUp'

const pct = (v: number) => ((v - SCORE_AXIS.min) / (SCORE_AXIS.max - SCORE_AXIS.min)) * 100

export default function ScoreGauge({ score }: { score: number }) {
  const shown = useCountUp(score)
  const zones = [
    { from: SCORE_AXIS.min, to: BAND.lo, cls: 'z-normal', label: '정상 확정' },
    { from: BAND.lo, to: INNER.lo, cls: 'z-outer', label: '' },
    { from: INNER.lo, to: INNER.hi, cls: 'z-inner', label: 'Qwen 재확인' },
    { from: INNER.hi, to: BAND.hi, cls: 'z-outer', label: '' },
    { from: BAND.hi, to: SCORE_AXIS.max, cls: 'z-defect', label: '불량 확정' },
  ]
  return (
    <div className="gauge">
      <div className="gauge-top">
        <span className="big-number">{shown.toFixed(2)}</span>
        <span className="gauge-unit">MMR 이상 점수</span>
      </div>
      <div className="gauge-track" role="img" aria-label={`MMR 점수 ${score.toFixed(2)}`}>
        {zones.map((z, i) => (
          <div key={i} className={'gauge-zone ' + z.cls} style={{ left: `${pct(z.from)}%`, width: `${pct(z.to) - pct(z.from)}%` }} />
        ))}
        <div className="gauge-tick" style={{ left: `${pct(MMR.threshold)}%` }}>
          <span>기준 {MMR.threshold.toFixed(2)}</span>
        </div>
        <div className="gauge-marker" style={{ left: `${pct(score)}%` }} />
      </div>
      <div className="gauge-labels">
        {zones
          .filter((z) => z.label)
          .map((z) => (
            <span key={z.label} className={z.cls} style={{ left: `${(pct(z.from) + pct(z.to)) / 2}%` }}>
              {z.label}
            </span>
          ))}
      </div>
    </div>
  )
}
