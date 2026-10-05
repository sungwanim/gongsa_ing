export type DefectType = 'ablation' | 'breakdown' | 'fracture' | 'groove'
export type Verdict = 'normal' | DefectType
export type Zone = 'clear_normal' | 'amb' | 'confident'
export type Decision = 'defect' | 'normal'

export interface TraceStep {
  thought: string
  action: string
  observation: string
  auto?: boolean
  choice_probs?: Record<string, number> | null
}

export interface Timings {
  mmr_s: number
  agent_s: number
  total_s: number
}

/** GET /api/dashboard 의 항목 (서버 SQLite 에 저장된 한 번의 검사) */
export interface Inspection {
  id: number
  created_at: string
  source: 'upload' | 'gallery'
  sep_flag: string | null
  mmr_score: number
  rscore: number
  zone: Zone
  decision: Decision
  defect_type: string | null
  why: string
  tools: string[]
  trace: TraceStep[]
  type_probs: Record<string, number> | null
  timings: Timings
  map_b64: string
  map_shape: [number, number]
  image_size: [number, number] | null
}

export interface GalleryItem {
  id: string
  thumb: string
}

/** POST /api/inspect* 가 스트림으로 보내는 이벤트 */
export type ServerEvent =
  | { event: 'start'; data: { source: string; size: [number, number]; warning: string | null } }
  | { event: 'mmr'; data: { score: number; rscore: number; zone: Zone; map_b64: string; map_shape: [number, number]; sec: number } }
  | { event: 'thought'; data: { text: string; options: string[] } }
  | { event: 'action'; data: { tool: string; choice_probs: Record<string, number> | null } }
  | { event: 'tool_start'; data: { tool: string } }
  | { event: 'observation'; data: { tool: string; text: string; sec: number } }
  | {
      event: 'final'
      data: { id: number; decision: Decision; defect_type: string | null; why: string; zone: Zone; tools: string[]; type_probs: Record<string, number> | null; timings: Timings }
    }
  | { event: 'done'; data: { id: number } }
  | { event: 'error'; data: { message: string } }

export type InspectInput = { kind: 'upload'; file: File } | { kind: 'gallery'; id: string }
