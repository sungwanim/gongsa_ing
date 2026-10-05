# Blade Inspector (frontend)

항공기 블레이드 이상 탐지 Agent의 웹 화면입니다. 1차 MMR, 2차 Qwen VLM 판정과 이상 히트맵을 보여 주는 **디자인 데모**이며, 지금은 서버 없이 mock 데이터로 동작합니다.

- React 18 + Vite 5 + TypeScript (외부 UI 라이브러리 없음)
- 폰트: Pretendard (CDN), 라이트/다크 모드, 모바일 반응형

## 실행

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # 타입 검사 + 프로덕션 빌드
```

## 화면 구성

| 구역 | 내용 |
|---|---|
| 히어로 | 서비스 소개, 샘플 보기 |
| 검사 결과 | 최종 판정 카드, **이상 히트맵 뷰어**(마우스를 올리면 위치별 점수, 최고점·Qwen 확인 영역 표시), 1차 MMR 점수 게이지, 2차 Qwen 확률, 최종 판정 |
| 검사 기록 | 샘플별 점수·단계·판정 표 |
| 불량 가이드 | 삭마·파손·파단·홈 설명 (샘플 관찰 기반 초안) |

## 판정 규칙 (`src/config.ts`, `src/lib/decide.ts`)

실험에서 고정한 값을 그대로 옮겼습니다. 백엔드 연결 시 서버 응답으로 대체하세요.

- 바깥 구간 [0.323, 0.541): 이 밖은 MMR 판정 확정
- 안쪽 [0.377, 0.486): Qwen이 재판정. 불량 확률 t=0.30, ±0.05 이면 "검사원 확인 필요"
- MMR이 불량으로 본 이미지는 Qwen이 종류(ablation / breakdown / fracture / groove)만 분류

## 백엔드 연결 방법

1. `src/types.ts`의 `Sample`이 서버 응답 형태입니다 (`mmrScore`, `vlm`, `typeProbs`, `box`).
2. `src/mock/samples.ts`를 API 호출로 바꾸고, `HeatmapViewer`의 `makeMap(...)`(mock 히트맵)을 서버가 주는 이상 맵(예: 224×224 float 배열)으로 바꾸면 됩니다. 좌표(`box`)는 0~1 비율입니다.
3. 화면의 수치는 모두 예시이며 실제 검사 결과나 성능을 뜻하지 않습니다.

## 폴더

```
src/
  components/   NavBar, Hero, SamplePicker, HeatmapViewer, ScoreGauge,
                PipelineSteps, VerdictCard, HistoryTable, DefectGlossary
  lib/          colormap, mockMap, decide, useCountUp
  mock/         samples (데모 데이터)
  config.ts     임계값·클래스 정보  types.ts  타입  styles.css  디자인 토큰
```
