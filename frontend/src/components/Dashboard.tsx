import type { Inspection } from '../types'
import ResultCard from './ResultCard'

interface Props {
  items: Inspection[]
  freshId: number | null
  loading: boolean
}

export default function Dashboard({ items, freshId, loading }: Props) {
  const defects = items.filter((i) => i.decision === 'defect').length
  const avg = items.length ? items.reduce((a, i) => a + i.timings.total_s, 0) / items.length : 0
  return (
    <>
      <div className="summary">
        <div className="stat">
          <span className="stat-label">검사한 이미지</span>
          <span className="stat-value">{items.length}장</span>
        </div>
        <div className="stat">
          <span className="stat-label">불량 판정</span>
          <span className="stat-value">{defects}장</span>
        </div>
        <div className="stat">
          <span className="stat-label">평균 소요 시간</span>
          <span className="stat-value">{items.length ? `${avg.toFixed(1)}초` : '-'}</span>
        </div>
      </div>
      {loading && !items.length && <p className="muted">대시보드를 불러오는 중이에요…</p>}
      {!loading && !items.length && (
        <div className="empty card">
          <b>아직 검사한 이미지가 없어요</b>
          <span>위에서 이미지를 올리거나 샘플을 선택하면 결과가 여기에 쌓여요.</span>
        </div>
      )}
      <div className="rc-grid">
        {items.map((it) => (
          <ResultCard key={it.id} item={it} fresh={it.id === freshId} />
        ))}
      </div>
    </>
  )
}
