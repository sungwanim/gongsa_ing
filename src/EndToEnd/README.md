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
