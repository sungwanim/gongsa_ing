import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// 개발 서버가 /api 를 에이전트 서비스로 넘긴다. 외부 접속은 포트 포워딩 대신 Tailscale 주소를 VITE_API_TARGET 에 넣는다.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8200'
  allowedHosts: [
      '*.loca.lt'
  ]
  return {
    plugins: [react()],
    server: { proxy: { '/api': { target, changeOrigin: true } } },
    preview: { proxy: { '/api': { target, changeOrigin: true } } },
  }
})
