# Blade Inspector (frontend)

항공기 블레이드 검사 웹 화면. 이미지 1장을 올리면 서버(`src/EndToEnd/agent_service`)가 MMR → 에이전트(Qwen) 루프로 판정하고,
진행 과정이 실시간으로 화면에 나타난 뒤 결과가 **대시보드에 추가되고 이미지 선택 화면으로 자동 복귀**한다.

- React 18 + Vite 5 + TypeScript (외부 UI 라이브러리 없음), Pretendard, 라이트/다크, 반응형

## 실행

```bash
cd frontend
cp .env.example .env      # VITE_API_TARGET 에 에이전트 서비스 주소
npm install
npm run dev               # http://localhost:5173  (/api 는 VITE_API_TARGET 으로 프록시)
npm run build             # 타입 검사 + 프로덕션 빌드
```

- **진짜 서버**: 서버에서 `scripts/run_mmr.sh`, `scripts/run_agent.sh` 를 실행하고, `VITE_API_TARGET` 에 그 주소를 넣는다.
  외부에서 접속할 때는 포트 포워딩 대신 **Tailscale 같은 VPN 주소**를 쓰고, 서버에 `AGENT_TOKEN` 을 설정한다.
  화면이 401 을 받으면 접근 토큰 입력 창을 띄우고, 토큰은 이 탭(sessionStorage)에만 저장한다.
- **GPU 없이 화면만 개발할 때**: `python src/EndToEnd/dev/run_mock_server.py` (가짜 서버, **판정은 무작위**라 성능·데모 자료로 쓰지 말 것)

## 화면 흐름
1. **검사하기**: 이미지 끌어다 놓기/선택, 또는 샘플 갤러리 썸네일 선택 (갤러리는 id 만 받고 파일 경로·정답은 받지 않음)
2. **진행 화면**(SSE): MMR 점수·히트맵·구간 → 에이전트의 생각 / 도구 선택(확률) / 관찰 → 최종 판정과 불량 종류 확률
3. 완료되면 토스트와 함께 **대시보드 맨 위에 카드 추가**, 선택 화면으로 자동 복귀 (처리 중에는 새 검사 불가)
4. **대시보드**: 서버 SQLite 에 저장된 카드(히트맵, 판정, 구간, 사용한 도구, 소요 시간, 에이전트 기록 펼치기). 새로고침해도 유지

## 구조
```
src/
  api.ts          fetch + SSE(POST 스트림 파싱) 클라이언트, 토큰
  types.ts        서버 응답/이벤트 타입 (서버와 같은 형태)
  runState.ts     서버 이벤트 -> 진행 화면 상태 (순수 reducer)
  config.ts       클래스/구간/도구 이름(한글), 히트맵 색 눈금
  lib/            colormap, f16(서버가 보내는 float16 히트맵 해석), useCountUp
  components/     Picker, RunView, HeatmapCanvas, Dashboard, ResultCard, TokenGate, NavBar, Hero, DefectGlossary
```

참고: 에이전트에는 "보류" 상태가 없고 최종 판정은 정상/불량(불량이면 종류)이다. 불량 종류 설명은 샘플 관찰 기반 초안이다.
