import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { ApiError, getDashboard, streamInspect } from './api'
import DefectGlossary from './components/DefectGlossary'
import Dashboard from './components/Dashboard'
import Hero from './components/Hero'
import NavBar from './components/NavBar'
import Picker from './components/Picker'
import RunView from './components/RunView'
import TokenGate from './components/TokenGate'
import { newRun, reduceRun, type RunState } from './runState'
import type { InspectInput, Inspection, ServerEvent } from './types'

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
  const [run, dispatch] = useReducer(
    (s: RunState | null, a: { type: 'start' } | { type: 'event'; e: ServerEvent } | { type: 'fail'; message: string } | { type: 'clear' }) => {
      if (a.type === 'start') return newRun()
      if (a.type === 'clear') return null
      if (!s) return s
      return a.type === 'fail' ? { ...s, error: a.message } : reduceRun(s, a.e)
    },
    null,
  )
  const [items, setItems] = useState<Inspection[]>([])
  const [loading, setLoading] = useState(true)
  const [freshId, setFreshId] = useState<number | null>(null)
  const [toast, setToast] = useState('')
  const [needToken, setNeedToken] = useState(false)
  const busy = useRef(false)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#101114' : '#f2f4f6')
    try {
      localStorage.setItem('theme', theme)
    } catch {
      /* 무시 */
    }
  }, [theme])

  const refresh = useCallback(async () => {
    try {
      setItems(await getDashboard())
      setNeedToken(false)
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setNeedToken(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const showToast = (m: string) => {
    setToast(m)
    window.setTimeout(() => setToast(''), 3500)
  }

  const start = useCallback(
    async (input: InspectInput) => {
      if (busy.current) return
      busy.current = true
      dispatch({ type: 'start' })
      let doneId: number | null = null
      try {
        await streamInspect(input, (e) => {
          dispatch({ type: 'event', e })
          if (e.event === 'done') doneId = e.data.id
        })
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) setNeedToken(true)
        dispatch({ type: 'fail', message: e instanceof Error ? e.message : '검사에 실패했어요' })
        window.setTimeout(() => dispatch({ type: 'clear' }), 4500)
        busy.current = false
        return
      }
      if (doneId === null) {
        // 스트림이 done 없이 끝남(서버 오류 이벤트 등): 오류 표시 후 복귀
        window.setTimeout(() => dispatch({ type: 'clear' }), 4500)
        busy.current = false
        return
      }
      // 한 루프 종료: 대시보드에 추가 -> 이미지 선택 화면으로 복귀
      await refresh()
      setFreshId(doneId)
      showToast('결과를 대시보드에 추가했어요')
      window.setTimeout(() => {
        dispatch({ type: 'clear' })
        busy.current = false
      }, 1400)
    },
    [refresh],
  )

  const running = run !== null
  return (
    <>
      <NavBar theme={theme} onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
      <main>
        <Hero />

        <section className="section" id="inspect">
          <h2 className="section-title">검사하기</h2>
          <p className="section-sub">{running ? '검사가 끝나면 결과가 대시보드에 추가되고 다음 이미지를 고를 수 있어요.' : '이미지를 고르면 바로 검사를 시작해요.'}</p>
          {needToken ? (
            <TokenGate onSaved={() => { setNeedToken(false); void refresh() }} />
          ) : running && run ? (
            <RunView run={run} />
          ) : (
            <Picker disabled={running} onPick={(i) => void start(i)} onNeedToken={() => setNeedToken(true)} />
          )}
        </section>

        <section className="section" id="dashboard">
          <h2 className="section-title">대시보드</h2>
          <p className="section-sub">검사한 결과가 최신순으로 쌓여요. 새로고침해도 유지돼요.</p>
          <Dashboard items={items} freshId={freshId} loading={loading} />
        </section>

        <section className="section" id="guide">
          <h2 className="section-title">불량 종류 가이드</h2>
          <p className="section-sub">Qwen이 구분하는 4가지 불량이에요.</p>
          <DefectGlossary />
        </section>
      </main>
      {toast && <div className="toast" role="status">{toast}</div>}
      <footer className="footer">
        <p>Blade Inspector · MMR + Qwen 에이전트 2단계 검사 화면이에요.</p>
      </footer>
    </>
  )
}
