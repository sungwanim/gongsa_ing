# 서버 통합 점검 가이드 (end2end, 자동화판)

GPU 서버(team14)에서 **MMR → 에이전트(Qwen) → 대시보드** 전체 흐름을 처음 돌려 보며 확인한다.
점검은 `scripts/e2e.sh` 하나로 자동화되어 있고, 각 단계가 **PASS / WARN / FAIL을 스스로 판정**해서 결과표(`report.md`)를 만든다.
사람이 하는 일은 명령어를 실행하고 **결과표 하나를 전달하는 것**이다.

- 서버 프로젝트 루트: `~/end2end/gongsa_ing` (브랜치 `end2end`). 스크립트가 자기 위치에서 루트를 알아내므로 별도 설정이 필요 없다.
- 이 문서의 명령어는 **서버 터미널**(`team14@aisa`)에서 실행한다. 맥에서 하는 일은 "9. 맥에서 화면 확인" 하나뿐이다.

## 빠른 시작

```bash
cd ~/end2end/gongsa_ing && git pull                 # 최신 코드 (서버에서 허용되는 명령)
bash src/EndToEnd/scripts/e2e.sh all                # 전체 점검 (실패하면 그 단계에서 멈추고 결과표를 보여 줌)
cat ~/end2end/logs/report.md                        # 결과표 (이 내용을 그대로 전달)
```

`all`은 `init → check → params → verify-mmr → up → test`를 순서대로 실행한다. **중간에 FAIL이 나면 멈춘다.**
원인을 고친 뒤 같은 명령을 다시 실행하면 된다(`params`는 산출물이 있으면 건너뛴다). 끝나면 서비스는 켜진 채로 남으므로 쓰지 않을 때는 `down`으로 끈다.

시간 어림값(미측정): 점검과 기준값 생성 수 분, MMR 검증 수 분, 에이전트 시작(Qwen 로딩) 1~3분, 통합 테스트는 이미지 3장 처리 시간에 따라 달라진다.

## 지켜야 할 규칙 (서버)

| 규칙 | 이유 |
|---|---|
| 서버에서 `git commit` / `git push` 금지 (`git pull`은 허용) | 팀 규칙 |
| `pip install` / `conda install` / `conda update` 금지 (공유 환경 `mmr`, `velm_qwen`) | 앞서 환경이 깨진 적이 있음. **이 점검은 설치가 필요 없다** |
| `sudo` 금지 | 권한 문제는 관리자에게 문의 |
| `/home/team14/gongsa_ing`, `/home/team14/gongsa_velm`, MMR 결과 폴더는 **읽기 전용** | 팀원 작업 폴더 |
| 새로 만드는 파일은 `~/end2end/` 아래에만 (스크립트가 그렇게 동작함) | 기존 결과와 섞이지 않게 |
| GPU는 다른 사람과 함께 쓴다. 쓰지 않을 때는 `down` | 16~18GB MIG 조각 공유 |

---

## 명령어

| 명령 | 하는 일 | 자동 판정 (PASS 기준) |
|---|---|---|
| `e2e.sh init` | `~/end2end/{artifacts,data,logs}`와 설정 파일 `e2e.conf` 생성 | - |
| `e2e.sh check` | 코드·conda 환경·GPU·입력 파일 점검 (읽기 전용) | 브랜치 `end2end`, `mmr`은 torch 2.1.x, `velm_qwen`은 2.8.x이고 둘 다 GPU 사용 가능, GPU 조각이 비어 있음, 체크포인트·이미지 1639장·csv/npz 각 4개·Qwen 저장 결과·목록 파일 존재 |
| `e2e.sh params` | 기준값·분리 목록 생성(`build_params.py`) + 겹침 검사. `--force`로 재생성 | 라우터 재현 일치, Qwen 저장 결과가 보정용 이미지 **전체**를 덮음, 참고/보정/갤러리 **겹침 0** |
| `e2e.sh verify-mmr` | 온라인 MMR 추론을 기존 CSV 점수와 비교 (15장), GPU 메모리·시간 측정 | 최대 차이 `1e-3` 미만 |
| `e2e.sh up` | MMR 서비스 → 에이전트 서비스 순서로 시작, 준비될 때까지 대기 | 두 서비스 health 정상 |
| `e2e.sh test` | 통합·안전 테스트 (갤러리 N장 검사, 이벤트 순서, 소요 시간, 참고 이미지 경고, 이미지가 아닌 파일 거절, 경로·해시 비노출, 대시보드 저장) | 모든 항목 통과 (소요 시간 초과는 WARN) |
| `e2e.sh status` / `down` | 서비스·GPU 상태 / 서비스 종료 | - |
| `e2e.sh expose` / `unexpose` | Tailscale 주소 + 접근 토큰으로 외부 접속 설정 / 해제 | 토큰 없이 접근하면 401 |
| `e2e.sh report` | 결과표 출력(`~/end2end/logs/report.md`) | - |

- 단계별로 따로 실행해도 되고(`bash src/EndToEnd/scripts/e2e.sh params` 등), 결과표는 실행할 때마다 갱신된다.
- 원본 로그는 `~/end2end/logs/`에 남는다: `params.log`, `separation.log`, `verify_mmr.log`, `mmr.log`, `agent.log`, `test.log`, `test.json`.
- 맥에서 `check`를 돌리면 서버 환경이 없어서 FAIL이 나오는 것이 정상이다(점검이 제대로 실패를 보고하는지 확인용).

## 결과표 읽는 법

`report.md`에는 아래 항목이 들어 있고, 마지막 줄에 `FAIL N건, WARN N건`이 나온다.

| 항목 | 의미 |
|---|---|
| 코드(브랜치@커밋) | 서버 코드 버전 |
| 환경 `mmr` / `velm_qwen` | torch 버전과 GPU 사용 가능 여부 |
| GPU 사용량(시작 전) | 우리 MIG 조각의 사용량. 높으면 다른 작업이 쓰는 중 |
| 입력 파일 점검 | 이후 단계가 읽는 파일 존재 여부 |
| 구간 기준 / 임계값 | `tau_lo`, `tau_hi`, `t` (기록용) |
| 보정 / 보고 / 참고 / 갤러리 장수 | 이미지 분리 규모 |
| Qwen 저장 결과가 보정용 이미지를 덮는 비율 | `N / N`이어야 PASS. 다르면 `t`가 기존 최종값과 다를 수 있음 |
| 참고/보정/갤러리 겹침 검사 | 겹침 0이어야 PASS |
| MMR 온라인 vs 기존 CSV 점수 최대 차이 | `1e-3` 미만이어야 PASS |
| MMR 모델 로드 시간 / 추론 시간 / GPU 피크 | 요청마다 로드하는 방식(B)의 비용 |
| MMR 서비스 / 에이전트 서비스 | 시작 성공 여부와 주소 |
| Qwen 상주 후 GPU 사용량 | Qwen + 참고 이미지 캐시가 차지하는 메모리 |
| 통합·안전 테스트 | PASS/WARN/FAIL 개수 |
| 이미지당 소요 시간 | 첫 이미지(콜드) / 이후 중앙값 / 최대. 기준은 30초(가정, 미측정) |
| 검사 후 GPU 사용량 | 처리 후 메모리 |

---

## FAIL이 나왔을 때

| 결과표의 메시지 | 의미 / 조치 |
|---|---|
| `conda 환경이 없음` | 환경 이름이 다름. `conda env list`로 확인해서 `~/end2end/e2e.conf`의 `MMR_ENV`/`AGENT_ENV` 수정 |
| `공유 환경이 훼손된 것으로 보임` (`iJIT_NotifyEvent`, `torchaudio` 등) | **설치하지 말고** 결과표와 함께 알려 준다 (앞의 02:37 사고와 같은 증상) |
| `MMR 체크포인트 없음`, `데이터셋 없음` | `e2e.conf`에서 `MMR_CKPT`/`DATA_ROOT` 수정 |
| `MMR 점수 csv/npz 0개` | `MMR_OUT` 경로 확인 (`build_params`가 읽는 팀원의 읽기 전용 결과) |
| `Qwen 저장 결과 없음` | `VELM_RESULTS` 경로 확인 (`qwen_results_v8_n12.jsonl`, `qwen_results_v7.jsonl`) |
| `기준값 생성 실패` + 로그에 `KeyError`/`AssertionError` | 결과표의 "로그 끝부분"이 원인이다. 그대로 전달 |
| `기준값 생성 실패` + `분리 위반` | 참고/보정/갤러리 이미지가 겹침. **조용히 넘어가지 않는 것이 정상 동작.** `params.log`를 알려 준다 |
| `라우터 재현이 기존 계산과 다름` | 기존 `attach_router_scores`와 결과가 다름. `params.log`를 알려 준다 |
| `MMR 점수 … 불일치` | 온라인 전처리·계산이 기존과 다름. **다음 단계로 가지 말고** `verify_mmr.log`를 알려 준다 |
| `MMR/에이전트 서비스 … 시작 직후 종료됨` / `시간 초과` | 화면에 출력된 로그 마지막 줄을 확인. 아래 표 참고 |
| `포트 … 이미 사용 중` | 다른 사용자의 서비스일 수 있다. 임의로 끄지 말고 `e2e.conf`에서 `MMR_PORT`/`AGENT_PORT` 변경 |
| `통합 테스트 … 준비되어 있지 않음` | `e2e.sh up`을 먼저 실행 |

서비스 시작 실패 시 로그의 흔한 원인:

| 로그 | 조치 |
|---|---|
| `MMR 서비스에 연결할 수 없습니다` | MMR 서비스가 안 떠 있음. `mmr.log` 확인 |
| `참고 이미지를 찾을 수 없습니다` | `DATA_ROOT` 또는 목록(`holdout_manifest_12.csv`)의 경로 문제 |
| `갤러리 이미지가 변경되었습니다` | 갤러리 파일의 해시가 `params` 때와 다름(데이터 변경). 알려 준다 |
| `CUDA out of memory` | 다른 작업의 GPU 사용 여부 확인(`e2e.sh status`). `e2e.conf`에 `AGENT_REF_PX=96`(참고 이미지 토큰 수를 줄임)을 켜고 `down` 후 `up` |
| `ImportError … iJIT` / `torchaudio` | 환경 훼손. 설치하지 말고 알려 준다 |

## 설정 파일 `~/end2end/e2e.conf`

`init`이 주석 처리된 기본값과 함께 만든다. 기본값이 맞지 않을 때만 주석을 풀어 고친다.

| 변수 | 기본값 / 설명 |
|---|---|
| `DATA_ROOT` | `<루트>/AeBAD`, 없으면 `/home/team14/gongsa_ing/AeBAD`(읽기 전용) |
| `MMR_OUT`, `MMR_CKPT`, `VELM_RESULTS` | 팀원의 MMR 결과 폴더·체크포인트·Qwen 저장 결과(읽기 전용) |
| `MMR_PORT`, `AGENT_PORT` | `8101`, `8200` |
| `MMR_ENV`, `AGENT_ENV` | `mmr`, `velm_qwen` |
| `N_TEST` / `LATENCY_WARN` | 테스트할 갤러리 이미지 수(3) / 이미지당 허용 시간(30초, 넘으면 WARN) |
| `AGENT_REFS_MANIFEST` | 참고 이미지 목록. 기본 `holdout_manifest_12.csv`(12장). 15장 목록(`holdout_manifest.csv`)으로 바꾸려면 **팀원 확인 후** 지정하고 `e2e.sh params --force`로 다시 만든다 |
| `AGENT_REF_PX` | GPU 메모리가 부족하면 `96` 등으로 낮춤 |

---

## 9. 외부(맥) 접속 확인 (Tailscale)

포트 포워딩은 쓰지 않는다. **MMR 서비스는 계속 `127.0.0.1`에만 열려 있다.**

```bash
bash src/EndToEnd/scripts/e2e.sh expose      # Tailscale 주소 + 무작위 접근 토큰으로 에이전트를 다시 시작하고 테스트
```

- `expose`는 `tailscale`이 없거나 주소를 못 얻으면 FAIL로 알려 준다. 서버에 Tailscale이 있는지는 제가 확인하지 못했고, 없으면 **관리자에게 설치/권한을 문의**해야 한다(이 경우 서버 안에서의 `test` 통과까지가 점검 기준이다).
- 토큰은 화면에 한 번 출력되고 `~/end2end/expose.env`(권한 600)에 저장된다. **채팅에 붙이지 말고** 맥 화면에만 입력한다.
- 로컬 주소가 아닌 곳에 열 때 `AGENT_TOKEN`이 없으면 에이전트는 시작을 거부한다(그래서 `expose`가 토큰을 함께 만든다).
- 해제: `bash src/EndToEnd/scripts/e2e.sh unexpose`

### 맥에서 (맥 터미널) — 자동화되지 않는 부분
`expose`가 마지막에 안내하는 대로 진행한다.

```bash
cd /Users/kimchaeeun/Documents/gongsa_ing_end2end/frontend      # 맥의 end2end 작업 폴더
cp .env.example .env        # VITE_API_TARGET=http://<expose 가 알려 준 주소>:8200 로 수정
npm install && npm run dev  # http://localhost:5173
```

브라우저에서 `http://localhost:5173`을 열고 토큰을 입력한 뒤 확인한다.
- [ ] 샘플 갤러리 썸네일이 보인다
- [ ] 샘플을 누르면 진행 화면에 MMR 히트맵 → 에이전트의 생각·도구 선택·관찰이 하나씩 나타난다
- [ ] 완료되면 대시보드 맨 위에 카드가 추가되고 선택 화면으로 자동 복귀한다
- [ ] 새로고침해도 대시보드가 유지된다
- [ ] 맥에 있는 이미지 1장을 업로드해도 같은 흐름이 동작한다

막히면 화면 설명이나 브라우저 개발자도구 콘솔의 빨간 오류 줄을 알려 준다.

---

## 종료

```bash
bash src/EndToEnd/scripts/e2e.sh down      # 서비스 종료 (결과 DB·기준값은 ~/end2end/ 아래에 그대로 남음)
```
종료 신호만 보내고 최대 45초 기다린다. 아직 살아 있으면 경고만 하고 **강제 종료는 하지 않는다**(`status`로 확인). 다시 시작은 `e2e.sh up`.

## 파일 위치

| 무엇 | 어디 |
|---|---|
| 코드 / 스크립트 | `~/end2end/gongsa_ing/src/EndToEnd` (`scripts/e2e.sh`) |
| 설정 / 외부 접속 설정 | `~/end2end/e2e.conf`, `~/end2end/expose.env` |
| 기준값·분리·갤러리 목록 | `~/end2end/artifacts/{params,separation,gallery,summary}.json` |
| 대시보드 DB | `~/end2end/data/dashboard.sqlite3` (초기화하려면 서비스를 끄고 파일 이름만 바꿔 둔다. 삭제하지 말 것) |
| 로그 / 결과표 / PID | `~/end2end/logs/` (`report.md`, `results.tsv`, `*.log`, `*.pid`) |
| 읽기 전용 입력 | MMR 체크포인트·점수(`MMR_OUT`), Qwen 저장 결과(`VELM_RESULTS`), AeBAD(`DATA_ROOT`) |

## 보고 방법

1. `cat ~/end2end/logs/report.md`의 내용 전체를 전달한다(FAIL/WARN이 있어도 그대로).
2. FAIL이 있으면 위 표에서 안내한 로그 파일의 **마지막 40줄**을 추가로 전달한다: `tail -40 ~/end2end/logs/<로그 이름>.log`
3. 토큰(`expose.env`)은 전달하지 않는다.

## 부록: 스크립트 자체 검증 (맥, 서버 불필요)

스크립트를 고쳤을 때 서버에 올리기 전에 맥에서 확인한다.

```bash
bash src/EndToEnd/dev/test_e2e_script.sh         # e2e.sh 의 up / status / test / down 흐름을 가짜 서비스로 검증 (12개)
python3 src/EndToEnd/dev/test_local.py           # 에이전트 서비스 통합 테스트 (23개)
python3 src/EndToEnd/dev/test_build_params.py    # 기준값 생성(build_params)을 합성 데이터로 끝까지 실행 (16개, 분리 위반 중단 포함)
```

## 부록: 수동으로 하나씩 실행할 때 (디버깅용)

스크립트가 막힐 때 한 단계만 따로 보고 싶다면 아래를 쓴다. **서비스를 직접 백그라운드로 띄울 때는 `cd`를 먼저 하고 서비스 명령만 `&`로 보낸다.** `cd x && cmd &`는 묶음 전체가 백그라운드가 되어 `$!`가 서비스의 PID가 아니게 되고, 나중에 `kill`이 서비스를 못 끈다.

```bash
cd ~/end2end/gongsa_ing/src/EndToEnd
source "$(conda info --base)/etc/profile.d/conda.sh"

# 기준값 생성 (환경 velm_qwen)
conda activate velm_qwen && (cd offline && python build_params.py --mmr-out "$MMR_OUT" --results-dir "$VELM_RESULTS" --data-root "$DATA_ROOT" --out ~/end2end/artifacts)

# MMR 검증 (환경 mmr)
conda activate mmr && (cd mmr_service && python verify_mmr.py --ckpt "$MMR_CKPT" --csv "$MMR_OUT/image_scores_aebad_S_AeBAD_S_same.csv" --data-root "$DATA_ROOT" -n 10)

# 서비스 상태를 사람이 읽기 쉽게 한 장 검사 (표준 라이브러리만 사용)
python3 dev/sse_probe.py inspect --gallery 0
python3 dev/sse_probe.py dashboard
```
위 `$MMR_OUT`, `$VELM_RESULTS`, `$DATA_ROOT`, `$MMR_CKPT`는 `~/end2end/e2e.conf`에 적힌 값(기본값은 `e2e.sh init`이 출력)을 셸에 넣어야 한다.
