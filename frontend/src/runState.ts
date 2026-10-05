import type { Decision, ServerEvent, Timings, Zone } from './types'

export interface Step {
  thought?: string
  options?: string[]
  tool?: string
  choiceProbs?: Record<string, number> | null
  running?: boolean
  observation?: string
  sec?: number
}

export interface RunState {
  startedAt: number
  warning?: string | null
  mmr?: { score: number; rscore: number; zone: Zone; map_b64: string; map_shape: [number, number]; sec: number }
  steps: Step[]
  final?: { id: number; decision: Decision; defect_type: string | null; why: string; zone: Zone; tools: string[]; type_probs: Record<string, number> | null; timings: Timings }
  doneId?: number
  error?: string
}

export const newRun = (): RunState => ({ startedAt: Date.now(), steps: [] })

/** 서버 이벤트를 화면 상태로 접는다 (순수 함수) */
export function reduceRun(s: RunState, e: ServerEvent): RunState {
  const steps = s.steps.slice()
  const last = () => steps[steps.length - 1]
  switch (e.event) {
    case 'start':
      return { ...s, warning: e.data.warning }
    case 'mmr':
      return { ...s, mmr: e.data }
    case 'thought':
      steps.push({ thought: e.data.text, options: e.data.options })
      return { ...s, steps }
    case 'action':
      if (last()) steps[steps.length - 1] = { ...last(), tool: e.data.tool, choiceProbs: e.data.choice_probs }
      return { ...s, steps }
    case 'tool_start':
      if (last()) steps[steps.length - 1] = { ...last(), running: true }
      return { ...s, steps }
    case 'observation':
      if (last()) steps[steps.length - 1] = { ...last(), running: false, observation: e.data.text, sec: e.data.sec }
      return { ...s, steps }
    case 'final':
      return { ...s, final: e.data }
    case 'done':
      return { ...s, doneId: e.data.id }
    case 'error':
      return { ...s, error: e.data.message }
  }
}
