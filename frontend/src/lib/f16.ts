/** 서버가 보내는 히트맵(float16 리틀엔디안, base64)을 Float32Array 로 푼다. */
function half(h: number): number {
  const s = (h & 0x8000) >> 15
  const e = (h & 0x7c00) >> 10
  const f = h & 0x03ff
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}

export function decodeMap(b64: string): Float32Array {
  const bin = atob(b64)
  const out = new Float32Array(bin.length / 2)
  for (let i = 0; i < out.length; i++) out[i] = half(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8))
  return out
}
