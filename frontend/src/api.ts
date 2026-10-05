import type { GalleryItem, InspectInput, Inspection, ServerEvent } from './types'

const BASE: string = import.meta.env.VITE_API_BASE ?? ''
const TOKEN_KEY = 'e2e_token'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export function getToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setToken(t: string) {
  try {
    sessionStorage.setItem(TOKEN_KEY, t)
  } catch {
    /* 저장소를 못 쓰는 환경 */
  }
}

function authHeaders(): Record<string, string> {
  const t = getToken()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

/** <img src> 는 헤더를 못 붙이므로 토큰을 쿼리로 붙인다 */
export function assetUrl(path: string): string {
  const t = getToken()
  return BASE + path + (t ? `${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(t)}` : '')
}

async function json<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path, { headers: authHeaders() })
  if (!r.ok) throw new ApiError(r.status, r.status === 401 ? '접근 토큰이 필요해요' : `요청에 실패했어요 (${r.status})`)
  return r.json()
}

/** 검사한 이미지의 미리보기(JPEG). <img src> 용이라 토큰을 쿼리로 붙인다 */
export const inspectionImageUrl = (id: number): string => assetUrl(`/api/dashboard/${id}/image`)

export const getGallery = async (): Promise<GalleryItem[]> => (await json<{ items: GalleryItem[] }>('/api/gallery')).items
export const getDashboard = async (): Promise<Inspection[]> => (await json<{ items: Inspection[] }>('/api/dashboard')).items
export const getHealth = () => json<{ status: string; ready: boolean; busy: boolean }>('/api/health')

/**
 * 검사를 시작하고 서버가 보내는 SSE 이벤트를 하나씩 onEvent 로 전달한다.
 * (EventSource 는 POST 를 못 쓰므로 fetch 스트림을 직접 파싱한다.)
 */
export async function streamInspect(input: InspectInput, onEvent: (e: ServerEvent) => void, signal?: AbortSignal): Promise<void> {
  const init: RequestInit =
    input.kind === 'upload'
      ? { method: 'POST', body: input.file, headers: { 'Content-Type': 'application/octet-stream', ...authHeaders() }, signal }
      : { method: 'POST', headers: authHeaders(), signal }
  const url = input.kind === 'upload' ? '/api/inspect' : `/api/inspect/gallery/${input.id}`
  const r = await fetch(BASE + url, init)
  if (!r.ok || !r.body) {
    let msg = `요청에 실패했어요 (${r.status})`
    try {
      msg = (await r.json()).error ?? msg
    } catch {
      /* 본문 없음 */
    }
    throw new ApiError(r.status, msg)
  }
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i: number
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i)
      buf = buf.slice(i + 2)
      const ev = /^event: (.+)$/m.exec(block)?.[1]
      const data = /^data: (.+)$/m.exec(block)?.[1]
      if (ev && data) onEvent({ event: ev, data: JSON.parse(data) } as ServerEvent)
    }
  }
}
