import { CLASSES } from '../config'
import type { InspectionResult, Sample } from '../types'
import { IconAlert, IconCheck, IconSkip } from './Icons'
import ScoreGauge from './ScoreGauge'

interface Props {
  sample: Sample
  result: InspectionResult
}

export default function PipelineSteps({ sample, result }: Props) {
  const used = result.usedVlm
  const recheck = result.zone === 'recheck'
  return (
    <ol className="steps">
      <li className="step done">
        <div className="step-mark"><IconCheck /></div>
        <div className="step-body card">
          <p className="eyebrow">1차 · MMR</p>
          <h3 className="card-title">이미지 전체를 빠르게 살펴봤어요</h3>
          <ScoreGauge score={sample.mmrScore} />
        </div>
      </li>

      <li className={'step ' + (recheck ? 'active' : used ? 'done' : 'skip')}>
        <div className="step-mark">{recheck || used ? <IconCheck /> : <IconSkip />}</div>
        <div className="step-body card">
          <p className="eyebrow">2차 · Qwen VLM</p>
          {recheck && sample.vlm ? (
            <>
              <h3 className="card-title">점수가 애매해서 다시 확인했어요</h3>
              <div className="vlm-quote">“{sample.vlm.evidence}”</div>
              <div className="vlm-meta">
                <span className="chip">위치 · {sample.vlm.location}</span>
                <span className={'chip conf-' + sample.vlm.confidence}>
                  신뢰도 · {{ high: '높음', medium: '보통', low: '낮음' }[sample.vlm.confidence]}
                </span>
              </div>
              <div className="prob-list">
                {(Object.keys(sample.vlm.probs) as (keyof typeof sample.vlm.probs)[]).map((k) => (
                  <div className="prob-row" key={k}>
                    <span className="prob-name">{CLASSES[k].label}</span>
                    <div className="prob-track">
                      <div className="prob-fill" style={{ width: `${sample.vlm!.probs[k] * 100}%`, background: CLASSES[k].color }} />
                    </div>
                    <span className="prob-val">{Math.round(sample.vlm!.probs[k] * 100)}%</span>
                  </div>
                ))}
              </div>
            </>
          ) : used ? (
            <>
              <h3 className="card-title">불량 종류를 분류했어요</h3>
              <p className="muted">MMR이 불량으로 판단한 이미지는 Qwen이 종류만 확인해요.</p>
            </>
          ) : (
            <>
              <h3 className="card-title">이번에는 건너뛰었어요</h3>
              <p className="muted">MMR이 충분히 확신해서 Qwen 재확인이 필요하지 않았어요.</p>
            </>
          )}
        </div>
      </li>

      <li className={'step final ' + result.decision}>
        <div className="step-mark">{result.decision === 'normal' ? <IconCheck /> : <IconAlert />}</div>
        <div className="step-body card">
          <p className="eyebrow">최종 판정</p>
          <h3 className="card-title">{result.headline}</h3>
          <p className="muted">{result.reason}</p>
        </div>
      </li>
    </ol>
  )
}
