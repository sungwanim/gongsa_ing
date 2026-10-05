# EndToEnd (MMR → 에이전트 → 대시보드)

업로드한 이미지 1장을 **실시간으로** MMR → 에이전트(Qwen) 루프 → 판정까지 추론하고, 결과를 대시보드에 추가한다.
기존 `ModelA/MMR_Test`, `ModelB/velm` 코드는 수정하지 않고 이 폴더에서 감싸서 쓴다.

## 결정 사항 (인터뷰 결과)
- 실행: GPU 서버(team14)에서 Docker compose 2개 컨테이너 — `mmr`(torch 2.1.2, 단일 이미지 추론 API), `agent`(torch 2.8, Qwen 에이전트·대시보드 API·웹 서빙). 맥 브라우저는 포트 포워딩으로 접속.
- 메모리(16GB MIG): Qwen 상주, MMR 은 요청마다 로드 후 해제(방식 B). 환경변수로 상주 방식 전환 가능하게.
- 에이전트: `ModelB/velm/agent.py`(4c50316), assist 끔, Qwen 두뇌, 라우터, zoom_check 제외. "에이전트는 판정을 불량 쪽으로만 바꾼다" 안전장치 유지.
- 도구 온라인화: `read_map`(MMR 즉석 추론 + 이상 맵 특징), `ask_whole`(참고 이미지 + `CachedRefClassifier`), `second_prompt`(v7 프롬프트 직접 호출), `decide`.
- 입력: 브라우저 업로드 + 샘플 갤러리. **저장된 CSV/npz/Qwen 결과는 앱이 조회하지 않는다.** 체크포인트와 작은 파라미터 JSON은 사용.
- 기준값(구간 tau_lo/tau_hi, 임계값 t, 라우터 계수)은 `offline/build_params.py`로 한 번 계산해 JSON 으로 고정.
- 엄격한 분리: 퓨샷 참고 이미지 / 파라미터 계산(보정 절반) / 데모 갤러리가 겹치지 않는다. 갤러리 = 보고용 절반 − 모든 제외·참고 목록, 해시(SHA-256)로 시작 시 검사. 업로드 이미지가 참고·보정 이미지와 같은 파일이면 경고.
- 루프: 업로드/선택 → 진행 표시(SSE: 생각·도구·관찰) → 판정 → 대시보드 상단에 카드 추가 → 선택 화면으로 자동 복귀.
- 저장: SQLite + Docker 볼륨. 정답(양품/불량 종류)은 화면에 표시하지 않고 에이전트에도 전달하지 않는다.
- 갤러리 API 는 파일 경로를 브라우저에 보내지 않는다 (경로에 폴더명=라벨이 들어 있음). 이미지는 id 로만 제공.

## 서버 통합 점검
서버 점검은 `bash scripts/e2e.sh all` 로 자동화되어 있다(환경 점검 → 기준값 생성 → MMR 검증 → 서비스 실행 → 통합·안전 테스트, 결과표 `~/end2end/logs/report.md`). 사용법과 실패 시 조치는 [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md).

## 단계
1. 서버 점검(Docker/GPU, 참고 이미지 위치) 2. `offline/` 기준값 생성·분리 검증 3. `mmr` 서비스 4. `agent` 서비스 5. 프론트 연동 6. compose 와 서버 점검.

## 오프라인 단계 실행 (서버, velm_qwen 환경)
```bash
python offline/build_params.py \
  --mmr-out  <MMR_Test/log_MMR_AeBAD_S_54> \
  --results-dir <velm/results> \
  --data-root <AeBAD 폴더> \
  --out <새 폴더>
python offline/separation.py --artifacts <새 폴더>
```

## 구현 현황
- [x] 2단계 `offline/` 기준값 생성·분리 검증 (서버 실행 필요, 미실행)
- [x] 3단계 `mmr_service/` MMR 단일 이미지 추론 서비스(표준 라이브러리 HTTP, 요청 때 로드·해제)
- [x] 4단계 `agent_service/` 에이전트 서비스 (표준 라이브러리 HTTP + SSE + SQLite). 기존 `agent.py` 무수정, 온라인 어댑터로 실행
- [x] 5단계 프론트 연동 (`frontend/`): 업로드/갤러리 → SSE 진행 화면 → 대시보드 추가 → 자동 복귀. 실제 브라우저(Chrome 자동 조작) 17개 시나리오 통과(가짜 서버 기준)
- [ ] 6단계 서버 통합 점검

모의 모드 테스트(GPU 불필요): `python dev/test_local.py`

## 실행 (서버, conda 환경 2개, 설치 없음)
```bash
# 1) MMR 서비스 (환경 mmr)  - 로컬(127.0.0.1)에만 열림
MMR_CKPT=<.../checkpoints/MMR_aebad_S_AeBAD_S.pth> bash scripts/run_mmr.sh

# 2) 에이전트 서비스 (환경 velm_qwen)
AGENT_ARTIFACTS=<build_params 출력 폴더> AGENT_DATA_ROOT=<AeBAD 폴더> bash scripts/run_agent.sh
```

## 외부 접속 (포트 포워딩 금지, Tailscale 등 VPN 전제)
- 기본은 `127.0.0.1` 에만 연다. 외부(Tailscale 주소 등)에 열려면 `AGENT_HOST=<주소>` 와 `AGENT_TOKEN=<긴 무작위 문자열>` 이 **둘 다** 필요하다(토큰 없이 로컬이 아닌 주소로는 시작하지 않음).
- MMR 서비스는 내부용이라 항상 로컬에만 둔다. 인터넷에 직접 노출하지 않는다.
- 보안 장치: 접근 토큰(Bearer 또는 `?token=`), 업로드 크기·형식 검사, 동시 처리 1건, 갤러리는 id 로만 접근(경로 노출 없음), 업로드 파일명 미저장.
- Tailscale 설치·연결 상태는 확인하지 못했다(서버 접속 불가). 서버에서 Tailscale 을 쓸 수 있는지는 관리자 확인이 필요하다.

## 검증하지 못한 것 (서버에서 확인)
- 실제 MMR 추론이 기존 점수와 같은지 (`mmr_service/verify_mmr.py`), 교사 네트워크 가중치 캐시 유무
- `qwen_backend.py` 의 실제 Qwen 경로(참고 이미지 캐시, 두뇌, v7)와 GPU 메모리(MMR 요청 때 로드 + Qwen 상주 합계, 18GB 조각)
- 이미지 한 장당 소요 시간 (가정: 30초 안팎)
