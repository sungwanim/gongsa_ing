import { IconSpark, IconUpload } from './Icons'

export default function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero-badge">
        <IconSpark width={16} height={16} />
        MMR + Qwen 2단계 검사
      </div>
      <h1>
        블레이드 이상,
        <br />
        <span className="grad">한눈에 확인해요</span>
      </h1>
      <p className="hero-sub">
        MMR이 먼저 살펴보고, 판단이 애매한 블레이드만 Qwen이 한 번 더 확인해요.
        <br className="hide-sm" />
        점수와 확률, 이상 위치를 히트맵으로 모두 보여드려요.
      </p>
      <div className="hero-actions">
        <a className="btn primary" href="#inspect">
          샘플 검사 보기
        </a>
        <button className="btn ghost" disabled title="데모에서는 비활성화되어 있어요">
          <IconUpload width={18} height={18} />
          이미지 올리기
          <span className="soon">준비 중</span>
        </button>
      </div>
    </section>
  )
}
