import { useEffect, useMemo, useRef, useState } from 'react'
import { SCORE_AXIS } from '../config'
import { colorAt, gradientCss, normalizeScore } from '../lib/colormap'
import { decodeMap } from '../lib/f16'

interface Props {
  mapB64: string
  shape: [number, number]
  /** 큰 뷰(마우스로 점수 확인, 범례, 최고점 표시) 또는 카드용 작은 뷰 */
  large?: boolean
  caption?: string
}

export default function HeatmapCanvas({ mapB64, shape, large, caption }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [h, w] = shape
  const values = useMemo(() => decodeMap(mapB64), [mapB64])
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null)

  const peak = useMemo(() => {
    let best = 0
    for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k
    return { x: (best % w) / (w - 1), y: Math.floor(best / w) / (h - 1), v: values[best] }
  }, [values, w, h])

  useEffect(() => {
    const c = ref.current
    const ctx = c?.getContext('2d')
    if (!c || !ctx) return
    c.width = w
    c.height = h
    const img = ctx.createImageData(w, h)
    for (let k = 0; k < values.length; k++) {
      const [r, g, b] = colorAt(normalizeScore(values[k]))
      img.data[k * 4] = r
      img.data[k * 4 + 1] = g
      img.data[k * 4 + 2] = b
      img.data[k * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [values, w, h])

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
    setHover({ x, y, v: values[Math.round(y * (h - 1)) * w + Math.round(x * (w - 1))] })
  }

  return (
    <figure className={'hm' + (large ? ' large' : '')}>
      <div
        className="heat"
        onPointerMove={large ? onMove : undefined}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`이상 히트맵, 최고 점수 ${peak.v.toFixed(2)}`}
      >
        <canvas ref={ref} />
        <div className="heat-grid" />
        <div className="heat-peak" style={{ left: `${peak.x * 100}%`, top: `${peak.y * 100}%` }}>
          <i />
          {large && <b>{peak.v.toFixed(2)}</b>}
        </div>
        {hover && (
          <>
            <div className="heat-cross" style={{ left: `${hover.x * 100}%`, top: `${hover.y * 100}%` }} />
            <div className="heat-tip" style={{ left: `${Math.min(hover.x, 0.72) * 100}%`, top: `${Math.max(hover.y - 0.12, 0.02) * 100}%` }}>
              {hover.v.toFixed(2)}
            </div>
          </>
        )}
      </div>
      {large && (
        <div className="legend">
          <div className="legend-bar" style={{ background: gradientCss }} />
          <div className="legend-labels">
            <span>정상 · {SCORE_AXIS.min.toFixed(1)}</span>
            <span>이상 점수</span>
            <span>{SCORE_AXIS.max.toFixed(1)} · 불량</span>
          </div>
        </div>
      )}
      {caption && <figcaption className="hint">{caption}</figcaption>}
    </figure>
  )
}
