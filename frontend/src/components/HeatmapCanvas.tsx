import { useEffect, useMemo, useRef, useState } from 'react'
import { SCORE_AXIS } from '../config'
import { colorAt, gradientCss, normalizeScore } from '../lib/colormap'
import { decodeMap } from '../lib/f16'
import { MAP_INSET, overlayAlpha } from '../lib/overlay'

type Mode = 'overlay' | 'image' | 'heat'

interface Props {
  mapB64: string
  shape: [number, number]
  /** 큰 뷰(마우스로 점수 확인, 범례, 보기 전환·투명도) 또는 카드용 작은 뷰 */
  large?: boolean
  caption?: string
  /** 검사한 이미지(미리보기). 있으면 이 위에 히트맵을 겹쳐 그린다. 없으면 히트맵만(이전 기록) */
  imageSrc?: string
  imageSize?: [number, number] | null
}

const MODES: { id: Mode; label: string }[] = [
  { id: 'overlay', label: '오버레이' },
  { id: 'image', label: '원본' },
  { id: 'heat', label: '히트맵만' },
]

export default function HeatmapCanvas({ mapB64, shape, large, caption, imageSrc, imageSize }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const layer = useRef<HTMLDivElement>(null)
  const [h, w] = shape
  const values = useMemo(() => decodeMap(mapB64), [mapB64])
  const [mode, setMode] = useState<Mode>('overlay')
  const [opacity, setOpacity] = useState(0.75)
  const [hover, setHover] = useState<{ x: number; y: number; v: number } | null>(null)
  const [imgFailed, setImgFailed] = useState(false)

  const hasImage = !!imageSrc && !imgFailed
  const eff: Mode = hasImage ? mode : 'heat'
  // 히트맵이 놓이는 범위: 이미지가 있으면 MMR 이 본 안쪽 영역, 없으면 전체
  const inset = hasImage ? `${MAP_INSET * 100}%` : '0'
  const aspect = hasImage && imageSize ? `${imageSize[0]} / ${imageSize[1]}` : '1 / 1'

  const peak = useMemo(() => {
    let best = 0
    for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k
    return { x: (best % w) / (w - 1), y: Math.floor(best / w) / (h - 1), v: values[best] }
  }, [values, w, h])

  useEffect(() => {
    const c = canvas.current
    const ctx = c?.getContext('2d')
    if (!c || !ctx) return
    c.width = w
    c.height = h
    const img = ctx.createImageData(w, h)
    const overlay = eff === 'overlay'
    for (let k = 0; k < values.length; k++) {
      const norm = normalizeScore(values[k])
      const [r, g, b] = colorAt(norm)
      img.data[k * 4] = r
      img.data[k * 4 + 1] = g
      img.data[k * 4 + 2] = b
      img.data[k * 4 + 3] = overlay ? Math.round(255 * overlayAlpha(norm, opacity)) : 255
    }
    ctx.putImageData(img, 0, 0)
  }, [values, w, h, eff, opacity])

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = layer.current
    if (!el || eff === 'image') return setHover(null)
    const r = el.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width
    const y = (e.clientY - r.top) / r.height
    if (x < 0 || x > 1 || y < 0 || y > 1) return setHover(null)       // MMR 이 보지 않은 가장자리
    setHover({ x, y, v: values[Math.round(y * (h - 1)) * w + Math.round(x * (w - 1))] })
  }

  return (
    <figure className={'hm' + (large ? ' large' : '')}>
      {large && hasImage && (
        <div className="hm-controls">
          <div className="seg" role="tablist" aria-label="보기 방식">
            {MODES.map((m) => (
              <button key={m.id} role="tab" aria-selected={mode === m.id} className={mode === m.id ? 'on' : ''} onClick={() => setMode(m.id)}>
                {m.label}
              </button>
            ))}
          </div>
          {eff === 'overlay' && (
            <label className="slider">
              <span>투명도</span>
              <input type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="히트맵 투명도" />
            </label>
          )}
        </div>
      )}
      <div
        className={'heat' + (hasImage ? ' with-img' : '')}
        style={{ aspectRatio: aspect }}
        onPointerMove={large ? onMove : undefined}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`이상 히트맵${hasImage ? '(이미지 위에 겹침)' : ''}, 최고 점수 ${peak.v.toFixed(2)}`}
      >
        {hasImage && <img className="hm-img" src={imageSrc} alt="" onError={() => setImgFailed(true)} draggable={false} />}
        <div ref={layer} className="hm-layer" style={{ inset, display: eff === 'image' ? 'none' : 'block' }}>
          <canvas ref={canvas} />
          {!hasImage && <div className="heat-grid" />}
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
        {hasImage && eff !== 'image' && <div className="hm-frame" style={{ inset }} aria-hidden="true" />}
      </div>
      {large && eff !== 'image' && (
        <div className="legend">
          <div className="legend-bar" style={{ background: gradientCss }} />
          <div className="legend-labels">
            <span>정상 · {SCORE_AXIS.min.toFixed(1)}</span>
            <span>이상 점수</span>
            <span>{SCORE_AXIS.max.toFixed(1)} · 불량</span>
          </div>
        </div>
      )}
      {large && hasImage && <p className="hint">점선 안쪽이 MMR이 실제로 본 영역이에요(가장자리 약 6%는 모델이 보지 않아요).</p>}
      {caption && <figcaption className="hint">{caption}</figcaption>}
    </figure>
  )
}
