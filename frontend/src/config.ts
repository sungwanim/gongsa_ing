import type { DefectType, Verdict, Zone } from './types'

/** 히트맵 색 눈금(MMR 이상 점수 범위). 서버 응답의 점수와 같은 단위 */
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

export const ZONES: Record<Zone, { label: string; hint: string }> = {
  clear_normal: { label: '정상 확정 구간', hint: 'MMR이 확실히 정상으로 본 구간이에요' },
  amb: { label: '애매 구간', hint: '점수가 애매해서 에이전트가 더 확인해요' },
  confident: { label: '확실한 불량 구간', hint: 'MMR이 확실히 불량으로 본 구간이에요' },
}

export const TOOLS: Record<string, string> = {
  read_map: '이상 맵 읽기',
  ask_whole: 'Qwen 전체 판정',
  second_prompt: '두 번째 프롬프트',
  zoom_check: '확대 확인',
  decide: '판정 확정',
}

export const typeLabel = (t: string | null): string =>
  t && t in CLASSES ? CLASSES[t as Verdict].label : t === 'unknown' ? '종류 미확인' : t ?? ''

/** 서버가 보낸 글에 남아 있는 영어 도구 이름·분류 이름을 한국어 화면 용어로 바꾼다 (표시 전용) */
export const koText = (text: string): string =>
  text
    .replace(/\b(read_map|ask_whole|second_prompt|zoom_check|decide)\b/g, (m) => `‘${TOOLS[m]}’`)
    .replace(/\b(normal|ablation|breakdown|fracture|groove)\b/gi, (m) => CLASSES[m.toLowerCase() as Verdict]?.label ?? m)
