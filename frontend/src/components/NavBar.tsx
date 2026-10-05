import { IconMoon, IconSun, LogoMark } from './Icons'

interface Props {
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

export default function NavBar({ theme, onToggleTheme }: Props) {
  return (
    <header className="nav">
      <div className="nav-inner">
        <a className="brand" href="#top" aria-label="Blade Inspector 홈">
          <LogoMark />
          <span>Blade Inspector</span>
        </a>
        <nav className="nav-links" aria-label="주요 메뉴">
          <a href="#inspect">검사하기</a>
          <a href="#dashboard">대시보드</a>
          <a href="#guide">불량 가이드</a>
        </nav>
        <button
          className="icon-btn"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? '라이트 모드로 전환' : '다크 모드로 전환'}
        >
          {theme === 'dark' ? <IconSun /> : <IconMoon />}
        </button>
      </div>
    </header>
  )
}
