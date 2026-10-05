import { useEffect, useMemo, useRef, useState } from 'react'
import { SCORE_AXIS } from '../config'
import { colorAt, gradientCss, normalizeScore } from '../lib/colormap'
import { MAP_SIZE, makeMap } from '../lib/mockMap'
import type { Sample } from '../types'

interface Props {
  sample: Sample
}

export default function HeatmapViewer({ sample }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [showBox, setShowBox] = useState(true)
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null)
  const map = useMemo(() => makeMap(sample.heat, sample.mmrScore), [sample])

  const peak = useMemo(() => {
    let best = 0
    for (let k = 1; k < map.length; k++) if (map[k] > map[best]) best = k
    return { x: (best % MAP_SIZE) / (MAP_SIZE - 1), y: Math.floor(best / MAP_SIZE) / (MAP_SIZE - 1) }
  }, [map])

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    c.width = MAP_SIZE
    c.height = MAP_SIZE
    const ctx = c.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(MAP_SIZE, MAP_SIZE)
    for (let k = 0; k < map.length; k++) {
      const [r, g, b] = colorAt(normalizeScore(map[k]))
      img.data[k * 4] = r
      img.data[k * 4 + 1] = g
      img.data[k * 4 + 2] = b
      img.data[k * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [map])

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
    const i = Math.round(x * (MAP_SIZE - 1))
    const j = Math.round(y * (MAP_SIZE - 1))
    setHover({ x, y, v: map[j * MAP_SIZE + i] })
  }

  const [x0, y0, x1, y1] = sample.box

  return (
    <div className="card viewer">
      <div className="viewer-head">
        <div>
          <p className="eyebrow">이상 히트맵</p>
          <h2 className="card-title">{sample.title}</h2>
        </div>
        <button
          className={'chip toggle' + (showBox ? ' on' : '')}
          onClick={() => setShowBox((v) => !v)}
          aria-pressed={showBox}
        >
          Qwen 확인 영역
        </button>
      </div>

      <div
        className="heat"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`${sample.title} 이상 히트맵, 최고 점수 ${sample.mmrScore.toFixed(2)}`}
      >
        <canvas ref={canvas} />
        <div className="heat-grid" />
        {showBox && (
          <div
            className="heat-box"
            style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}
          >
            <span className={y0 < 0.1 ? 'inside' : ''}>Qwen이 확대해서 본 곳</span>
          </div>
        )}
        <div className="heat-peak" style={{ left: `${peak.x * 100}%`, top: `${peak.y * 100}%` }}>
          <i />
          <b>{sample.mmrScore.toFixed(2)}</b>
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

      <div className="legend">
        <div className="legend-bar" style={{ background: gradientCss }} />
        <div className="legend-labels">
          <span>정상 · {SCORE_AXIS.min.toFixed(1)}</span>
          <span>이상 점수</span>
          <span>{SCORE_AXIS.max.toFixed(1)} · 불량</span>
        </div>
      </div>
      <p className="hint">히트맵 위에 마우스를 올리면 그 위치의 점수를 볼 수 있어요.</p>
    </div>
  )
}
