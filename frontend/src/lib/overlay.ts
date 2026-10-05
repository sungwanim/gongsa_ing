/** MMR 은 이미지를 256x256 으로 늘린 뒤 가운데 224x224 만 본다 -> 이상 맵은 원본의 안쪽 224/256 영역에 해당한다. */
export const MAP_INSET = 16 / 256        // 가장자리 6.25%

/** 겹쳐 그릴 때의 알파: 낮은 점수(정상 수준)는 투명, 높은 점수만 진하게 -> 이상 부위가 이미지 위에 떠오른다 */
export function overlayAlpha(norm: number, opacity: number): number {
  const lo = 0.14
  const hi = 0.58
  const t = Math.min(1, Math.max(0, (norm - lo) / (hi - lo)))
  return opacity * t * t * (3 - 2 * t)      // smoothstep
}
