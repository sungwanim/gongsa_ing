import { SCORE_AXIS } from '../config'

const STOPS: [number, string][] = [
  [0.0, '#0b1f4b'],
  [0.22, '#2d6cdf'],
  [0.42, '#2ec4b6'],
  [0.6, '#a3e635'],
  [0.76, '#ffd23f'],
  [0.9, '#ff8a3d'],
  [1.0, '#e5322d'],
]

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

const RGB = STOPS.map(([p, c]) => [p, hexToRgb(c)] as const)

/** 0~1 -> [r, g, b] */
export function colorAt(t: number): [number, number, number] {
  const v = Math.min(1, Math.max(0, t))
  for (let i = 1; i < RGB.length; i++) {
    const [p1, c1] = RGB[i]
    if (v <= p1) {
      const [p0, c0] = RGB[i - 1]
      const f = (v - p0) / (p1 - p0 || 1)
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f]
    }
  }
  return RGB[RGB.length - 1][1] as unknown as [number, number, number]
}

/** 점수(SCORE_AXIS 범위) -> 0~1 */
export function normalizeScore(s: number): number {
  return (s - SCORE_AXIS.min) / (SCORE_AXIS.max - SCORE_AXIS.min)
}

export const gradientCss = `linear-gradient(90deg, ${STOPS.map(([p, c]) => `${c} ${p * 100}%`).join(', ')})`
