# 팀명_공사중

항공엔진 블레이드 이상 탐지 엔드투엔드 데모: 이미지 선택/업로드 → MMR(1차) → Qwen2-VL 에이전트(2차) → 대시보드.

- 실행·점검: `src/EndToEnd/INTEGRATION_GUIDE.md`, `src/EndToEnd/scripts/e2e.sh`
- 프론트엔드: `frontend/` (React + Vite)
- 1차 모델: `src/ModelA/MMR_Test`, 에이전트: `src/ModelB/velm/agent.py`

## 데이터셋 (`AeBAD/AeBAD_S`)

AeBAD_S가 저장소에 포함되어 있다 (출처·라이선스: `AeBAD/README.md`, `AeBAD/LICENSE-DATASET`, 무결성: `AeBAD/AeBAD_S.sha256`).
데이터셋 폴더는 프로젝트 루트(`/AeBAD`)에 있어야 한다.
