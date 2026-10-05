export type DefectType = 'ablation' | 'breakdown' | 'fracture' | 'groove'
export type Verdict = 'normal' | DefectType
export type Decision = 'normal' | 'defect' | 'hold'
export type Confidence = 'high' | 'medium' | 'low'

/** 히트맵을 만드는 데모용 설명(실제 연동 시에는 서버가 주는 224x224 이상 맵으로 대체). 좌표는 0~1 */
export interface Blob {
  x: number
  y: number
  r: number
  a: number
}

export interface HeatSpec {
  seed: number
  base: number
  blobs: Blob[]
}

export interface VlmResult {
  /** 1 - P(normal). 서버에서 결합 분류기가 계산한 불량 확률 */
  defectProb: number
  /** Qwen 이 각 클래스에 준 확률(합 1) */
  probs: Record<Verdict, number>
  confidence: Confidence
  evidence: string
  location: string
}

export interface Sample {
  id: string
  title: string
  note: string
  capturedAt: string
  /** MMR 이상 점수 (이상 맵 최댓값) */
  mmrScore: number
  heat: HeatSpec
  /** VLM 에 넘긴 크롭 영역 [x0, y0, x1, y1], 0~1 */
  box: [number, number, number, number]
  vlm?: VlmResult
  /** 불량일 때 종류별 확률 (VLM 확률 + MMR 맵 특징 결합) */
  typeProbs?: Record<DefectType, number>
}

export type Zone = 'normal-sure' | 'outer-low' | 'recheck' | 'outer-high' | 'defect-sure'

export interface InspectionResult {
  zone: Zone
  usedVlm: boolean
  decision: Decision
  defectType: DefectType | null
  headline: string
  reason: string
}
