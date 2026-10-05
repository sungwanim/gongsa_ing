import { useState } from 'react'
import { setToken } from '../api'

export default function TokenGate({ onSaved }: { onSaved: () => void }) {
  const [v, setV] = useState('')
  return (
    <form
      className="card token"
      onSubmit={(e) => {
        e.preventDefault()
        setToken(v.trim())
        onSaved()
      }}
    >
      <p className="eyebrow">접근 토큰</p>
      <h3 className="card-title">서버가 접근 토큰을 요구해요</h3>
      <p className="muted">서버를 실행할 때 설정한 AGENT_TOKEN 값을 입력하세요. 이 브라우저 탭에만 저장돼요.</p>
      <input className="input" type="password" value={v} onChange={(e) => setV(e.target.value)} placeholder="토큰" autoFocus />
      <button className="btn primary" type="submit" disabled={!v.trim()}>
        확인
      </button>
    </form>
  )
}
