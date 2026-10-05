import { useEffect, useMemo, useState } from 'react'
import DefectGlossary from './components/DefectGlossary'
import HeatmapViewer from './components/HeatmapViewer'
import Hero from './components/Hero'
import HistoryTable from './components/HistoryTable'
import NavBar from './components/NavBar'
import PipelineSteps from './components/PipelineSteps'
import SamplePicker from './components/SamplePicker'
import VerdictCard from './components/VerdictCard'
import { inspect } from './lib/decide'
import { SAMPLES } from './mock/samples'

type Theme = 'light' | 'dark'

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* 저장소를 못 쓰는 환경 */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [selectedId, setSelectedId] = useState(SAMPLES[3].id)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#101114' : '#f2f4f6')
    try {
      localStorage.setItem('theme', theme)
    } catch {
      /* 무시 */
    }
  }, [theme])

  const sample = useMemo(() => SAMPLES.find((s) => s.id === selectedId) ?? SAMPLES[0], [selectedId])
  const result = useMemo(() => inspect(sample), [sample])

  return (
    <>
      <NavBar theme={theme} onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
      <main>
        <Hero />

        <section className="section" id="inspect">
          <h2 className="section-title">검사 결과</h2>
          <p className="section-sub">샘플을 골라서 MMR과 Qwen이 어떻게 판단했는지 살펴보세요.</p>
          <SamplePicker samples={SAMPLES} selectedId={selectedId} onSelect={setSelectedId} />

          <div key={sample.id} className="fade-up">
            <VerdictCard sample={sample} result={result} />
            <div className="inspect-grid">
              <HeatmapViewer sample={sample} />
              <PipelineSteps sample={sample} result={result} />
            </div>
          </div>
        </section>

        <section className="section" id="history">
          <h2 className="section-title">검사 기록</h2>
          <p className="section-sub">행을 누르면 위에서 자세한 결과를 볼 수 있어요.</p>
          <HistoryTable samples={SAMPLES} selectedId={selectedId} onSelect={setSelectedId} />
        </section>

        <section className="section" id="guide">
          <h2 className="section-title">불량 종류 가이드</h2>
          <p className="section-sub">Qwen이 구분하는 4가지 불량이에요.</p>
          <DefectGlossary />
        </section>
      </main>
      <footer className="footer">
        <p>Blade Inspector · 항공기 블레이드 이상 탐지 데모 화면이에요. 표시된 데이터는 예시이며 실제 검사 결과가 아니에요.</p>
      </footer>
    </>
  )
}
