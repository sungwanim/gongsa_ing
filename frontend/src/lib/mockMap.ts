import type { HeatSpec } from '../types'
import { SCORE_AXIS } from '../config'

function mulberry32(a: number) {
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const MAP_SIZE = 112

/**
 * 데모용 이상 맵. 값의 최댓값이 mmrScore 가 되도록 만든다(실제 MMR 은 이상 맵의 max 가 점수).
 * 서버 연동 후에는 224x224 맵을 Float32Array 로 넘기면 된다.
 */
export function makeMap(spec: HeatSpec, mmrScore: number, size = MAP_SIZE): Float32Array {
  const rnd = mulberry32(spec.seed)
  const waves = Array.from({ length: 4 }, () => ({
    fx: 1 + rnd() * 3,
    fy: 1 + rnd() * 3,
    px: rnd() * Math.PI * 2,
    py: rnd() * Math.PI * 2,
    a: 0.4 + rnd() * 0.6,
  }))
  const field = new Float32Array(size * size)
  let max = 0
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const x = i / (size - 1)
      const y = j / (size - 1)
      let v = 0
      for (const w of waves) v += w.a * (0.5 + 0.5 * Math.sin(w.fx * x * 6.28 + w.px) * Math.cos(w.fy * y * 6.28 + w.py))
      v = (v / waves.length) * spec.base * 1.7
      for (const b of spec.blobs) {
        const d2 = (x - b.x) ** 2 + (y - b.y) ** 2
        v += b.a * Math.exp(-d2 / (2 * b.r * b.r))
      }
      field[j * size + i] = v
      if (v > max) max = v
    }
  }
  const lo = SCORE_AXIS.min
  for (let k = 0; k < field.length; k++) field[k] = lo + Math.pow(field[k] / (max || 1), 0.72) * (mmrScore - lo)
  return field
}
