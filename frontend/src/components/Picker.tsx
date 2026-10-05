import { useEffect, useRef, useState } from 'react'
import { ApiError, assetUrl, getGallery } from '../api'
import type { GalleryItem, InspectInput } from '../types'
import { IconUpload } from './Icons'

interface Props {
  disabled: boolean
  onPick: (input: InspectInput) => void
  onNeedToken: () => void
}

export default function Picker({ disabled, onPick, onNeedToken }: Props) {
  const [items, setItems] = useState<GalleryItem[] | null>(null)
  const [err, setErr] = useState('')
  const [drag, setDrag] = useState(false)
  const file = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getGallery()
      .then(setItems)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) onNeedToken()
        setErr(e instanceof Error ? e.message : '갤러리를 불러오지 못했어요')
      })
  }, [onNeedToken])

  const take = (f?: File | null) => {
    if (!f) return
    if (!f.type.startsWith('image/') && !/\.(png|jpe?g|bmp|webp)$/i.test(f.name)) {
      setErr('이미지 파일(PNG, JPG, BMP, WEBP)만 올릴 수 있어요')
      return
    }
    setErr('')
    onPick({ kind: 'upload', file: f })
  }

  return (
    <div className={'picker-wrap' + (disabled ? ' disabled' : '')}>
      <div
        className={'dropzone' + (drag ? ' drag' : '')}
        onDragOver={(e) => {
          e.preventDefault()
          setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          if (!disabled) take(e.dataTransfer.files?.[0])
        }}
        onClick={() => !disabled && file.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && file.current?.click()}
        aria-label="이미지 업로드"
      >
        <IconUpload width={28} height={28} />
        <b>검사할 이미지를 끌어다 놓거나 눌러서 선택하세요</b>
        <span>PNG · JPG · BMP · WEBP, 최대 30MB</span>
        <input ref={file} type="file" accept="image/*" hidden onChange={(e) => take(e.target.files?.[0])} />
      </div>

      <p className="sub-label">또는 샘플 이미지로 시험해 보기</p>
      <div className="gallery">
        {items === null && !err && Array.from({ length: 8 }, (_, i) => <div key={i} className="thumb skeleton" />)}
        {items?.map((g) => (
          <button key={g.id} className="thumb" disabled={disabled} onClick={() => onPick({ kind: 'gallery', id: g.id })} aria-label="샘플 이미지 선택">
            <img src={assetUrl(g.thumb)} alt="" loading="lazy" />
          </button>
        ))}
      </div>
      {err && <p className="form-error">{err}</p>}
    </div>
  )
}
