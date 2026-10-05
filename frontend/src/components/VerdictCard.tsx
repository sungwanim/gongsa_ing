import { CLASSES, DEFECT_ORDER } from '../config'
import { useCountUp } from '../lib/useCountUp'
import type { InspectionResult, Sample } from '../types'
import { IconAlert, IconCheck } from './Icons'

interface Props {
  sample: Sample
  result: InspectionResult
}

const LABEL = { normal: '정상', defect: '불량', hold: '확인 필요' } as const

export default function VerdictCard({ sample, result }: Props) {
  const showType = result.decision !== 'normal' && sample.typeProbs
  const defectP = useCountUp((sample.vlm?.defectProb ?? (sample.mmrScore >= 0.43 ? 1 : 0)) * 100)
  const typeInfo = result.defectType ? CLASSES[result.defectType] : null

  return (
    <section className={'card verdict ' + result.decision} aria-live="polite">
      <div className="verdict-main">
        <div className="verdict-icon">{result.decision === 'normal' ? <IconCheck width={26} height={26} /> : <IconAlert width={26} height={26} />}</div>
        <div>
          <p className="eyebrow">최종 판정 · {sample.id}</p>
          <h2 className="verdict-title">
            {LABEL[result.decision]}
            {result.decision !== 'normal' && typeInfo && (
              <span className="verdict-type" style={{ color: typeInfo.color }}>
                {' · '}
                {typeInfo.label}
                {result.decision === 'hold' && ' 의심'}
              </span>
            )}
          </h2>
          <p className="verdict-sub">{result.reason}</p>
        </div>
      </div>

      <div className="verdict-stats">
        <div className="stat">
          <span className="stat-label">MMR 점수</span>
          <span className="stat-value">{sample.mmrScore.toFixed(2)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">{sample.vlm ? 'Qwen 불량 확률' : '재확인'}</span>
          <span className="stat-value">{sample.vlm && result.zone === 'recheck' ? `${defectP.toFixed(0)}%` : '건너뜀'}</span>
        </div>
        <div className="stat">
          <span className="stat-label">촬영 시각</span>
          <span className="stat-value small">{sample.capturedAt}</span>
        </div>
      </div>

      {showType && sample.typeProbs && (
        <div className="type-bars">
          <p className="eyebrow">불량 종류 확률</p>
          {DEFECT_ORDER.map((k) => (
            <div className="prob-row" key={k}>
              <span className="prob-name">{CLASSES[k].label}</span>
              <div className="prob-track">
                <div className="prob-fill" style={{ width: `${sample.typeProbs![k] * 100}%`, background: CLASSES[k].color }} />
              </div>
              <span className="prob-val">{Math.round(sample.typeProbs![k] * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {result.decision === 'hold' && (
        <button className="btn primary wide" type="button">
          검사원에게 확인 요청하기
        </button>
      )}
    </section>
  )
}
