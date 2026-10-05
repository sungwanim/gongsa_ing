import type { DefectType, Verdict } from './types'

/** 실험에서 고정한 값(frozen_config.json). 백엔드 연결 시 서버 응답으로 대체하세요. */
export const MMR = { threshold: 0.431763, sigma: 0.109, k: 1.0 }
export const BAND = { lo: MMR.threshold - MMR.k * MMR.sigma, hi: MMR.threshold + MMR.k * MMR.sigma }
export const INNER = { lo: MMR.threshold - 0.5 * MMR.sigma, hi: MMR.threshold + 0.5 * MMR.sigma }
export const VLM_RULE = { threshold: 0.3, holdDelta: 0.05 }
export const SCORE_AXIS = { min: 0.2, max: 0.8 }

export interface ClassInfo {
  label: string
  en: string
  desc: string
  color: string
}

export const CLASSES: Record<Verdict, ClassInfo> = {
  normal: { label: '정상', en: 'Normal', desc: '눈에 띄는 손상이 없는 상태예요', color: '#12b76a' },
  ablation: { label: '삭마', en: 'Ablation', desc: '표면이 그을리거나 변색되고 거칠어진 넓은 영역', color: '#ff7a45' },
  breakdown: { label: '파손', en: 'Breakdown', desc: '표면에 작은 구멍이나 움푹 파인 곳, 떨어져 나간 조각', color: '#f04452' },
  fracture: { label: '파단', en: 'Fracture', desc: '끝이나 모서리가 부러지거나 깨져서 외곽선 일부가 없어진 상태', color: '#7c5cfc' },
  groove: { label: '홈', en: 'Groove', desc: '표면이나 모서리에 가늘게 파인 선 모양의 홈·틈', color: '#0ea5e9' },
}

export const DEFECT_ORDER: DefectType[] = ['ablation', 'breakdown', 'fracture', 'groove']
export const VERDICT_ORDER: Verdict[] = ['normal', ...DEFECT_ORDER]
