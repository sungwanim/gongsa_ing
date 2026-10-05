import { BAND, INNER, MMR, VLM_RULE } from '../config'
import type { DefectType, InspectionResult, Sample, Zone } from '../types'
import { DEFECT_ORDER } from '../config'

export function zoneOf(score: number): Zone {
  if (score < BAND.lo) return 'normal-sure'
  if (score >= BAND.hi) return 'defect-sure'
  if (score >= INNER.lo && score < INNER.hi) return 'recheck'
  return score < MMR.threshold ? 'outer-low' : 'outer-high'
}

function topType(s: Sample): DefectType | null {
  if (!s.typeProbs) return null
  return DEFECT_ORDER.reduce((a, b) => (s.typeProbs![b] > s.typeProbs![a] ? b : a))
}

/** 실험에서 고정한 규칙을 그대로 옮긴 데모용 판정 로직 */
export function inspect(s: Sample): InspectionResult {
  const zone = zoneOf(s.mmrScore)
  const mmrDefect = s.mmrScore >= MMR.threshold

  if (zone === 'recheck' && s.vlm) {
    const p = s.vlm.defectProb
    if (Math.abs(p - VLM_RULE.threshold) < VLM_RULE.holdDelta) {
      return {
        zone,
        usedVlm: true,
        decision: 'hold',
        defectType: topType(s),
        headline: '검사원 확인이 필요해요',
        reason: 'MMR과 Qwen 모두 확신하기 어려운 점수예요. 사람이 한 번 더 확인해 주세요.',
      }
    }
    if (p >= VLM_RULE.threshold) {
      return {
        zone,
        usedVlm: true,
        decision: 'defect',
        defectType: topType(s),
        headline: '불량으로 판단했어요',
        reason: 'MMR 점수가 애매해서 Qwen이 다시 확인했고, 불량 가능성이 높다고 봤어요.',
      }
    }
    return {
      zone,
      usedVlm: true,
      decision: 'normal',
      defectType: null,
      headline: '정상으로 판단했어요',
      reason: 'MMR 점수는 애매했지만 Qwen이 다시 확인해 보니 눈에 띄는 손상이 없었어요.',
    }
  }

  if (mmrDefect) {
    return {
      zone,
      usedVlm: s.typeProbs !== undefined,
      decision: 'defect',
      defectType: topType(s),
      headline: '불량으로 판단했어요',
      reason:
        zone === 'defect-sure'
          ? 'MMR 점수가 충분히 높아서 불량으로 확정했어요. Qwen은 불량 종류만 분류했어요.'
          : 'MMR 판정을 그대로 사용했어요. Qwen은 불량 종류만 분류했어요.',
    }
  }
  return {
    zone,
    usedVlm: false,
    decision: 'normal',
    defectType: null,
    headline: '정상으로 판단했어요',
    reason:
      zone === 'normal-sure'
        ? 'MMR 점수가 충분히 낮아서 정상으로 확정했어요. Qwen 재판정은 건너뛰었어요.'
        : 'MMR 판정을 그대로 사용했어요.',
  }
}
