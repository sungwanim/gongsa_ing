# 서버 통합 점검 가이드 (end2end)

GPU 서버(team14)에서 **MMR → 에이전트(Qwen) → 대시보드** 전체 흐름을 처음 돌려 보며 확인하는 순서서다.
각 단계는 "목적 → 명령어 → 정상 출력 → 실패하면 → 확인할 값" 순서이고, **앞 단계가 통과해야 다음 단계로 간다.**

- 서버 프로젝트 루트: `~/end2end/gongsa_ing` (브랜치 `end2end`)
- 이 문서의 명령어는 **서버 터미널**(`team14@aisa`)에서 실행한다. 6, 9번 일부(맥 화면)만 맥에서 한다.
- 단계마다 "붙여 주세요"라고 표시한 출력을 복사해서 전달하면 된다. 화면이 길면 `| tail -N` 이 이미 붙어 있다.

## 지켜야 할 규칙 (서버)

| 규칙 | 이유 |
|---|---|
| 서버에서 `git commit` / `git push` 금지 (`git pull`, `git log`, `git status`는 허용) | 팀 규칙 |
| `pip install` / `conda install` / `conda update` 금지 (공유 환경 `mmr`, `velm_qwen`) | 앞서 환경이 깨진 적이 있음. **이 가이드는 설치가 필요 없다** |
| `sudo` 금지 | 권한 문제는 관리자에게 문의 |
| `/home/team14/gongsa_ing`, `/home/team14/gongsa_velm`, MMR 결과 폴더는 **읽기 전용** | 팀원 작업 폴더 |
| 새로 만드는 파일은 `~/end2end/` 아래에만 | 기존 결과와 섞이지 않게 |
| GPU는 다른 사람과 함께 쓴다. 쓰지 않을 때는 서비스를 끈다(10번) | 16~18GB MIG 조각 공유 |

---

## 0. 환경 변수 파일 만들기 (한 번만)

**목적**: 경로를 한곳에 모아 두고 이후 명령어에서 `$변수`로 쓴다. 새 터미널을 열 때마다 `source ~/end2end/e2e_env.sh`를 먼저 실행한다.

```bash
mkdir -p ~/end2end/artifacts ~/end2end/data ~/end2end/logs
cat > ~/end2end/e2e_env.sh <<'EOF'
export E2E_ROOT="$HOME/end2end/gongsa_ing"
export E2E_SRC="$E2E_ROOT/src/EndToEnd"
# AeBAD 데이터셋: 이 루트에 없으면 팀원의 읽기 전용 위치를 쓴다
export DATA_ROOT="$E2E_ROOT/AeBAD"
[ -d "$DATA_ROOT/AeBAD_S" ] || export DATA_ROOT="/home/team14/gongsa_ing/AeBAD"
# 팀원이 만든 읽기 전용 입력 (MMR 체크포인트·점수, Qwen 저장 결과)
export MMR_OUT="/home/team14/gongsa_ing/src/ModelA/MMR_Test/log_MMR_AeBAD_S_54"
export MMR_CKPT="$MMR_OUT/checkpoints/MMR_aebad_S_AeBAD_S.pth"
export VELM_RESULTS="/home/team14/gongsa_velm/src/ModelB/velm/results"
# 우리가 새로 만드는 폴더
export ART="$HOME/end2end/artifacts"      # 기준값·분리 목록(build_params 출력)
export E2E_DATA="$HOME/end2end/data"      # 대시보드 SQLite
export MMR_PORT=8101
export AGENT_PORT=8200
source "$(conda info --base)/etc/profile.d/conda.sh"
EOF
source ~/end2end/e2e_env.sh && echo "DATA_ROOT=$DATA_ROOT" && ls -d "$DATA_ROOT/AeBAD_S" "$MMR_OUT" "$VELM_RESULTS"
```

**정상**: `DATA_ROOT=...` 한 줄과 경로 3개가 에러 없이 나온다.
**실패하면**: `No such file or directory`가 나오는 경로를 알려 준다(경로가 다르면 `e2e_env.sh`의 해당 줄만 고친다).
**붙여 주세요**: 출력 전체(4줄 안팎).

---

## 1. 코드 확인

**목적**: 서버 루트가 `end2end` 최신인지 확인한다.

```bash
source ~/end2end/e2e_env.sh
cd "$E2E_ROOT" && git branch --show-current && git log --oneline -1 && git status --short | head -5 && ls "$E2E_SRC"
```

**정상**: 브랜치 `end2end`, 최신 커밋(이 문서가 들어 있는 커밋 이후), `git status`는 비어 있고, 목록에 `INTEGRATION_GUIDE.md agent_service common dev mmr_service offline scripts README.md`가 보인다.

**브랜치/버전이 다르면** (서버에서 허용되는 `pull`만 사용):
```bash
cd "$E2E_ROOT" && git fetch origin end2end && git checkout end2end && git pull
```
**루트 폴더가 아직 없으면** (새로 받기. 이력까지 받으면 오래 걸리므로 `--depth 1` 권장):
```bash
mkdir -p ~/end2end && cd ~/end2end && git clone --depth 1 -b end2end https://github.com/sungwanim/gongsa_ing.git gongsa_ing
```
**붙여 주세요**: 위 출력.

---

## 2. conda 환경과 GPU 점검 (읽기 전용)

**목적**: 두 환경(`mmr`, `velm_qwen`)이 정상이고 GPU가 보이는지 확인한다.

```bash
source ~/end2end/e2e_env.sh
conda env list | grep -E "^(mmr|velm_qwen) "
for e in mmr velm_qwen; do echo "-- $e: $(conda run -n $e python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"; done
echo "-- mmr 패키지: $(conda run -n mmr python -c 'import timm,scipy,sklearn,torchvision;print(timm.__version__, scipy.__version__)' 2>&1 | tail -1)"
echo "-- velm_qwen 패키지: $(conda run -n velm_qwen python -c 'import transformers,sklearn,scipy,PIL;from qwen_vl_utils import process_vision_info;print(transformers.__version__)' 2>&1 | tail -1)"
nvidia-smi | sed -n '/MIG devices/,/Processes/p' | head -10
df -h "$HOME" | tail -1
```

**정상**: `mmr`에서 `2.1.2+cu121 True`, `velm_qwen`에서 `2.8.0+cu129 True`, 패키지 줄에 버전 번호(오류 없음), MIG 표에 우리 조각의 `Memory-Usage`가 거의 비어 있음(`15MiB / ...`).
**실패하면**:
- `ImportError ... iJIT_NotifyEvent` → `velm_qwen`이 다시 깨진 것. **설치하지 말고** 출력과 함께 알려 준다(02:37 사고와 같은 증상).
- `False`(GPU 못 봄) → 다른 사람이 GPU를 점유 중인지 `nvidia-smi` 전체 출력으로 확인.
- MIG `Memory-Usage`가 이미 높으면 다른 작업이 우리 조각을 쓰고 있다는 뜻 → 끝날 때까지 대기.
**붙여 주세요**: 출력 전체.

---

## 3. 입력 파일 확인 (읽기 전용)

**목적**: 이후 단계가 읽는 파일이 모두 있는지 본다.

```bash
source ~/end2end/e2e_env.sh
ls -la "$MMR_CKPT" | cut -c1-110
echo "테스트 이미지: $(find "$DATA_ROOT/AeBAD_S/test" -type f -name '*.png' ! -name '._*' | wc -l)장"
echo "MMR 점수 csv: $(ls "$MMR_OUT"/image_scores_*.csv | wc -l)개, 이상 맵 npz: $(ls "$MMR_OUT"/anomaly_maps_*.npz | wc -l)개"
ls -la "$VELM_RESULTS"/qwen_results_v8_n12.jsonl "$VELM_RESULTS"/qwen_results_v7.jsonl | cut -c1-110
wc -l "$E2E_ROOT"/src/ModelB/velm/holdout_manifest*.csv
ls ~/.cache/torch/hub/checkpoints 2>/dev/null | head -5
```

**정상**: 체크포인트 약 408MB, 테스트 이미지 `1639장`, csv 4개 / npz 4개, `qwen_results_v8_n12.jsonl`·`v7.jsonl` 존재, 목록 파일 행 수(헤더 포함: `holdout_manifest_12.csv` 13줄, `holdout_manifest.csv` 16줄, `_60.csv` 61줄), 마지막에 `wide_resnet50_2-*.pth`가 보이면 MMR 교사 가중치 캐시가 있다는 뜻.
**주의**: 마지막 줄(교사 가중치)이 비어 있어도 인터넷이 되면 처음 실행 때 내려받는다. 그 경우 5번이 느릴 수 있다.
**붙여 주세요**: 출력 전체.

---

## 4. 기준값·분리 목록 생성 (`build_params.py`)

**목적**: 구간 기준(`tau_lo`/`tau_hi`), 임계값 `t`, 라우터 계수를 오프라인으로 한 번 계산해 파일로 고정하고, 참고/보정/갤러리 이미지의 **겹침이 없음**을 해시로 검사한다. 이 단계만 데이터셋 CSV/npz/저장된 Qwen 결과를 읽는다(앱 실행 중에는 읽지 않음). CPU 작업이고 입력은 읽기만 한다. 쓰는 곳은 `$ART`뿐이다.

```bash
source ~/end2end/e2e_env.sh && conda activate velm_qwen
cd "$E2E_SRC/offline" && python build_params.py --mmr-out "$MMR_OUT" --results-dir "$VELM_RESULTS" --data-root "$DATA_ROOT" --out "$ART" 2>&1 | tail -20
python separation.py --artifacts "$ART"
ls -la "$ART"
```

**정상(예)**:
```
행 N장 (보정용 ... / 보고용 ...)
저장된 Qwen v8_n12 결과가 보정용 X장 중 X장에 있음          <- 두 숫자가 같아야 한다
구간 기준 tau_lo=... tau_hi=... (rscore) | 임계값 t=...
제외 목록 3개 파일, 합계 ...장: ['holdout_manifest.csv', 'holdout_manifest_12.csv', 'holdout_manifest_60.csv']
갤러리 후보 ...장 (...), 해시 계산 ...초
분리 검사 통과: 참고 12 / 보정 ... / 갤러리 24 (서로 겹침 0)
저장: ... ['gallery.json', 'params.json', 'separation.json']
분리 검사 통과: {'refs': 12, 'calib': ..., 'gallery': 24}
```
**확인할 값 (기록)**: `tau_lo`, `tau_hi`, `t`, 보정/보고 장수, 참고 이미지 수.
**실패하면**:
| 증상 | 의미 / 조치 |
|---|---|
| `[경고] 보정용 이미지 일부에 Qwen 결과가 없습니다` | 팀원 저장 결과가 보정용 이미지 전체를 덮지 못함 → `t`가 기존 최종값과 다를 수 있음. 숫자와 함께 알려 준다 |
| `AssertionError: 라우터 재현이 ... 다릅니다` | 기존 `attach_router_scores`와 계산이 다름(중단이 맞음). 출력 전체를 알려 준다 |
| `AssertionError: 이미지 파일 N장을 찾을 수 없음` | `DATA_ROOT`가 틀림 → 0번 확인 |
| `AssertionError: 이미지 분리 위반` | 참고/보정/갤러리 중 겹침 발견. 출력 전체를 알려 준다(조용히 넘어가지 않는 것이 정상 동작) |
| `ModuleNotFoundError` | 환경 문제. 설치하지 말고 알려 준다 |

**참고 이미지를 15장(`holdout_manifest.csv`)으로 바꾸고 싶다면** (팀원 확인 후): 위 명령의 `build_params.py`에 `--refs-manifest "$E2E_ROOT/src/ModelB/velm/holdout_manifest.csv"`를 추가하고, 6번에서 에이전트 서비스를 시작할 때 `AGENT_REFS_MANIFEST`에 **같은 파일**을 지정한다. (보정에 쓰인 Qwen 저장 결과는 12장 기준이므로 바꾸면 `t`가 달라질 수 있다.)
**붙여 주세요**: 위 3개 명령의 출력 전체.

---

## 5. MMR 온라인 추론 검증 (`verify_mmr.py`)

**목적**: 이미지 1장을 즉석에서 추론한 점수가 기존 배치 평가의 점수(CSV)와 같은지 확인하고, **GPU 메모리와 시간**을 측정한다.

```bash
source ~/end2end/e2e_env.sh && conda activate mmr
cd "$E2E_SRC/mmr_service" && python verify_mmr.py --ckpt "$MMR_CKPT" --csv "$MMR_OUT/image_scores_aebad_S_AeBAD_S_same.csv" --data-root "$DATA_ROOT" -n 10 2>&1 | tail -14
python verify_mmr.py --ckpt "$MMR_CKPT" --csv "$MMR_OUT/image_scores_aebad_S_AeBAD_S_view.csv" --data-root "$DATA_ROOT" -n 5 2>&1 | tail -3
```

**정상**: 각 줄에 `online ... csv ... diff ...e-0x`와 `{'wait_s':..., 'load_s':..., 'infer_s':..., 'gpu_peak_mb':...}`가 나오고, 마지막에 `최대 차이 ... -> 통과`(기준 `1e-3` 미만).
**확인할 값 (기록)**: `load_s`(요청마다 로드하는 데 걸리는 시간), `infer_s`, `gpu_peak_mb`.
**실패하면**:
| 증상 | 의미 / 조치 |
|---|---|
| `최대 차이 ... -> 불일치(확인 필요)` | 온라인 전처리·계산이 기존과 다르다. **다음 단계로 가지 말고** 출력 전체를 알려 준다 |
| `FileNotFoundError` (체크포인트) | `MMR_CKPT` 경로 확인(3번) |
| `ModuleNotFoundError: config / models` | 실행 위치 확인(`cd "$E2E_SRC/mmr_service"`) |
| 교사 가중치 다운로드 오류 | 인터넷/캐시 문제. 오류 줄을 알려 준다 |
| `CUDA out of memory` | 다른 작업이 GPU 사용 중. 2번의 MIG 표 확인 |
**붙여 주세요**: 위 출력 전체.

---

## 6. 서비스 실행 (MMR → 에이전트 순서)

**목적**: 두 서비스를 백그라운드로 띄운다. 에이전트가 시작할 때 MMR 서비스에 연결하므로 **MMR을 먼저** 띄운다. 로그는 `~/end2end/logs/`에 남고, 터미널을 닫아도 계속 돈다.

### 6-1. 포트와 이전 프로세스 확인
```bash
source ~/end2end/e2e_env.sh
ss -ltn | grep -E ":($MMR_PORT|$AGENT_PORT) " || echo "포트 비어 있음"
```
**정상**: `포트 비어 있음`. 이미 점유 중이면 다른 사용자의 서비스일 수 있으니 임의로 끄지 말고 알려 준다(포트는 `0번 e2e_env.sh`의 `MMR_PORT`, `AGENT_PORT`로 바꿀 수 있다).

### 6-2. MMR 서비스 (환경 `mmr`, 로컬 `127.0.0.1`에만 열림, 기본은 요청마다 모델 로드·해제)
```bash
source ~/end2end/e2e_env.sh
cd "$E2E_SRC" && MMR_CKPT="$MMR_CKPT" nohup bash scripts/run_mmr.sh > ~/end2end/logs/mmr.log 2>&1 &
echo $! > ~/end2end/logs/mmr.pid
sleep 6; curl -s "http://127.0.0.1:$MMR_PORT/health"; echo; tail -3 ~/end2end/logs/mmr.log | cut -c1-160
```
**정상**: `{"status": "ok", "resident": false, "loaded": false}`.

### 6-3. 에이전트 서비스 (환경 `velm_qwen`, Qwen 로딩 1~3분)
```bash
source ~/end2end/e2e_env.sh
cd "$E2E_SRC" && AGENT_ARTIFACTS="$ART" AGENT_DATA_ROOT="$DATA_ROOT" AGENT_DATA_DIR="$E2E_DATA" AGENT_MMR_URL="http://127.0.0.1:$MMR_PORT" nohup bash scripts/run_agent.sh > ~/end2end/logs/agent.log 2>&1 &
echo $! > ~/end2end/logs/agent.pid
```
시작될 때까지 기다리며 확인한다(2~3분 후):
```bash
grep -E "MMR 서비스|참고 이미지|에이전트 서비스 시작|Error|SystemExit|OutOfMemory|분리 위반|변경되었습니다" ~/end2end/logs/agent.log | cut -c1-200
curl -s "http://127.0.0.1:$AGENT_PORT/api/health"; echo
```
**정상**: `MMR 서비스: {'status': 'ok', ...}`, `참고 이미지 12장 (ask_whole)`, `에이전트 서비스 시작 http://127.0.0.1:8200 (인증=없음(로컬 전용))`, health가 `{"status": "ok", "ready": true, "busy": false}`.
**실패하면**:
| 로그 | 의미 / 조치 |
|---|---|
| `MMR 서비스에 연결할 수 없습니다` | 6-2가 안 떠 있음. `~/end2end/logs/mmr.log` 확인 |
| `참고 이미지를 찾을 수 없습니다` | `DATA_ROOT` 또는 `holdout_manifest_12.csv`의 경로 문제 |
| `갤러리 이미지가 변경되었습니다` | 갤러리 파일의 해시가 4번 때와 다름(데이터 변경). 알려 준다 |
| `이미지 분리 위반` | 4번 산출물에 겹침. 알려 준다 |
| `CUDA out of memory` | 아래 "메모리가 부족하면" 참고 |
| `ImportError ... iJIT` / `torchaudio` | 환경이 다시 깨짐. 설치하지 말고 알려 준다 |

### 6-4. Qwen 상주 후 GPU 사용량 (기록)
```bash
nvidia-smi | sed -n '/MIG devices/,/Processes/p' | head -8
```
**확인할 값 (기록)**: 우리 조각의 `Memory-Usage` (Qwen + 참고 이미지 상주분).
**메모리가 부족하면**: 에이전트를 끄고(`kill $(cat ~/end2end/logs/agent.pid)`) `AGENT_REF_PX=96`(참고 이미지 토큰 수를 줄임)을 6-3 명령에 추가해 다시 시작한다. MMR은 이미 요청 때 로드·해제 방식이다.
**붙여 주세요**: 6-1 ~ 6-4 출력.

---

## 7. 통합 테스트 (서버에서, 실제 이미지 1~3장)

**목적**: 업로드 한 장이 **MMR → 에이전트 → 저장**까지 끝나는지, 진행 이벤트와 시간을 본다. (`dev/sse_probe.py`는 기본 주소가 `http://127.0.0.1:8200`이다. `AGENT_PORT`를 바꿨다면 명령마다 `--url http://127.0.0.1:<포트>`를 붙인다.) 첫 이미지는 참고 이미지 캐시를 처음 계산해서 오래 걸릴 수 있다(콜드 스타트).

```bash
source ~/end2end/e2e_env.sh && cd "$E2E_SRC"
python3 dev/sse_probe.py inspect --gallery 0 | cut -c1-175
python3 dev/sse_probe.py inspect --gallery 1 | tail -1
python3 dev/sse_probe.py inspect --gallery 2 | tail -1
python3 dev/sse_probe.py dashboard
grep -E "캐시 확인|메모리" ~/end2end/logs/agent.log | cut -c1-200
nvidia-smi | sed -n '/MIG devices/,/Processes/p' | head -8
```

**정상**: 첫 명령에서 `start → mmr → (thought → action → tool_start → observation) 반복 → final → done` 순서의 줄이 시간과 함께 나오고, 마지막에 `요약: 판정=... 종류=... 구간=... 도구=... 총 N초 (MMR ...s / 에이전트 ...s)`가 나온다. `dashboard`에 3건이 저장되고, 에이전트 로그에 `[캐시 확인 통과]`가 보인다.
**확인할 값 (기록)**:
- 이미지당 총 시간 (첫 이미지 / 이후): 가정은 **30초 안팎**이다
- MMR 시간(요청마다 로드 포함), 에이전트 시간, 에이전트가 고른 도구 순서
- 처리 후 GPU 사용량(6-4와 비교)
**실패하면**:
| 증상 | 의미 / 조치 |
|---|---|
| `error` 이벤트 `OutOfMemoryError` | GPU 부족. 6-4의 "메모리가 부족하면" |
| `HTTP 409` | 다른 검사가 처리 중. 잠시 후 다시 |
| `[캐시 사용 불가]` 로그 | 참고 이미지 캐시가 꺼져 전체 재계산 중 → 시간이 크게 늘 수 있음. 로그 줄을 알려 준다 |
| 도구를 하나도 안 쓰고 `decide`만 반복 | Qwen 두뇌의 선택 문제. `choice_probs`가 보이는 줄을 알려 준다 |
**붙여 주세요**: 첫 명령의 출력 전체, 나머지 명령의 출력.

---

## 8. 분리·안전 점검

**목적**: 참고 이미지를 올리면 경고가 뜨는지, 잘못된 파일은 거절되는지, API가 파일 경로를 노출하지 않는지 확인한다. (참고 이미지 업로드도 하나의 검사로 대시보드에 기록된다.)

```bash
source ~/end2end/e2e_env.sh && cd "$E2E_SRC"
REF=$(python3 -c "import json,os;print(json.load(open(os.environ['ART']+'/gallery.json'))['refs'][0]['file'])")
python3 dev/sse_probe.py inspect --file "$DATA_ROOT/$REF" | head -2 | cut -c1-260
echo "이미지가 아님" > /tmp/e2e_not_image.txt; python3 dev/sse_probe.py inspect --file /tmp/e2e_not_image.txt
curl -s "http://127.0.0.1:$AGENT_PORT/api/gallery" | head -c 260; echo
echo "대시보드 응답의 경로 노출 수: $(curl -s "http://127.0.0.1:$AGENT_PORT/api/dashboard" | grep -c 'AeBAD')"
python3 dev/sse_probe.py dashboard | tail -3
```

**정상**:
1. 첫 명령: `start` 줄의 `warning`에 `이 이미지는 퓨샷 참고 이미지와 같은 파일입니다...`
2. 두 번째: `HTTP 400: {"error": "이미지 파일이 아닙니다"}`
3. 갤러리 응답은 `{"items": [{"id": "...", "thumb": "/api/gallery/.../thumb"}, ...` 형태만(경로·정답 없음)
4. 경로 노출 수 `0`
5. 대시보드 마지막 줄에 `[경고: refs]` 표시
**붙여 주세요**: 출력 전체.

---

## 9. 맥에서 화면 확인 (Tailscale)

**목적**: 외부에서 Tailscale 주소로 접속해 실제 화면으로 한 루프(선택 → 진행 → 대시보드 추가 → 복귀)를 확인한다. 포트 포워딩은 쓰지 않는다. **MMR 서비스는 계속 `127.0.0.1`에만 둔다.**

### 9-1. 서버에서 Tailscale 확인
```bash
which tailscale && tailscale ip -4 | head -1 || echo "tailscale 없음"
```
- 주소(`100.x.x.x`)가 나오면 9-2로 간다.
- `tailscale 없음`이면 **관리자에게 설치/권한을 문의**해야 한다(제가 서버를 직접 확인할 수 없어 이 가이드는 설치 방법을 정하지 않았다). 그동안은 서버 안에서의 7~8번까지가 통과 기준이다.
- 임시 점검용 대안으로 맥에서 `ssh -N -L 8200:127.0.0.1:8200 team14@<서버>`(맥 `localhost`에만 열리는 SSH 터널)가 있지만, 사용 여부는 팀 판단에 맡긴다.

### 9-2. 에이전트를 Tailscale 주소 + 토큰으로 다시 시작
로컬이 아닌 주소에 열 때는 **`AGENT_TOKEN`이 없으면 서비스가 시작을 거부**한다.
```bash
source ~/end2end/e2e_env.sh
kill $(cat ~/end2end/logs/agent.pid); sleep 3
export AGENT_HOST="$(tailscale ip -4 | head -1)"
export AGENT_TOKEN="$(openssl rand -hex 24)"
echo "토큰(맥 화면에 입력): $AGENT_TOKEN"; echo "주소: http://$AGENT_HOST:$AGENT_PORT"
cd "$E2E_SRC" && AGENT_ARTIFACTS="$ART" AGENT_DATA_ROOT="$DATA_ROOT" AGENT_DATA_DIR="$E2E_DATA" AGENT_MMR_URL="http://127.0.0.1:$MMR_PORT" nohup bash scripts/run_agent.sh > ~/end2end/logs/agent.log 2>&1 &
echo $! > ~/end2end/logs/agent.pid
```
2~3분 뒤 `grep "에이전트 서비스 시작" ~/end2end/logs/agent.log`에 `(인증=토큰)`이 보이면 된다. 토큰은 채팅에 붙이지 말고 맥 화면에만 입력한다.

### 9-3. 맥에서 프론트엔드 실행 (맥 터미널)
```bash
cd /Users/kimchaeeun/Documents/gongsa_ing_end2end/frontend      # 맥의 end2end 작업 폴더
cp .env.example .env        # VITE_API_TARGET=http://<위 주소>:8200 으로 수정
npm install && npm run dev  # http://localhost:5173
```
브라우저에서 `http://localhost:5173`을 열고 토큰을 입력한 뒤 다음을 확인한다.
- [ ] 샘플 갤러리 썸네일이 보인다
- [ ] 샘플을 누르면 진행 화면에 MMR 히트맵 → 에이전트의 생각·도구 선택·관찰이 하나씩 나타난다
- [ ] 완료되면 대시보드 맨 위에 카드가 추가되고, 선택 화면으로 자동 복귀한다
- [ ] 새로고침해도 대시보드가 유지된다
- [ ] 맥에 있는 이미지 1장을 업로드해도 같은 흐름이 동작한다
**붙여 주세요**: 막히는 단계의 화면 설명(또는 브라우저 개발자도구 콘솔의 빨간 오류 줄).

---

## 10. 종료와 정리

GPU를 쓰지 않을 때는 서비스를 끈다. 결과(대시보드 DB, 기준값)는 `~/end2end/` 아래에 그대로 남는다.

```bash
kill $(cat ~/end2end/logs/agent.pid) $(cat ~/end2end/logs/mmr.pid) 2>/dev/null; sleep 3
ps aux | grep -E "agent_service|mmr_service|server.py" | grep -v grep | wc -l   # 0 이면 종료
```
다시 시작할 때는 `source ~/end2end/e2e_env.sh` 후 6번부터 실행하면 된다(4번 산출물이 있으면 4번은 생략).

---

## 결과 기록표 (단계마다 채워서 전달)

| 항목 | 값 | 단계 |
|---|---|---|
| 서버 커밋 |  | 1 |
| `tau_lo` / `tau_hi` / `t` |  | 4 |
| 보정 N / 보고 N / 갤러리 N / 참고 N |  | 4 |
| Qwen 저장 결과가 보정용을 덮는 비율 |  | 4 |
| MMR 온라인 vs CSV 최대 차이 |  | 5 |
| MMR `load_s` / `infer_s` / `gpu_peak_mb` |  | 5 |
| Qwen 상주 후 GPU 사용량 |  | 6-4 |
| 이미지당 총 시간 (첫 이미지 / 이후) |  | 7 |
| 처리 후 GPU 사용량 |  | 7 |
| 참고 이미지 업로드 시 경고 표시 |  | 8 |
| 맥 화면 한 루프 통과 |  | 9 |

## 문제 해결 요약

| 증상 | 먼저 볼 것 |
|---|---|
| `ImportError ... iJIT_NotifyEvent` / `torchaudio` | 공유 환경 훼손. **설치하지 말고** 알려 준다 |
| `CUDA out of memory` | 다른 작업의 GPU 사용(2번 MIG 표), `AGENT_REF_PX=96`, MMR은 이미 요청 때 로드·해제 |
| `HTTP 401` | `AGENT_TOKEN` 불일치 → 화면의 토큰 입력 창에 다시 입력 |
| `HTTP 409` | 다른 검사가 처리 중 (한 번에 한 장) |
| 에이전트가 안 뜸 | `~/end2end/logs/agent.log` 마지막 30줄 |
| MMR 점수 불일치 | 5번에서 중단하고 출력 전체를 알려 준다 |
| 결과 DB를 초기화하고 싶음 | 서비스를 끄고 `$E2E_DATA/dashboard.sqlite3`만 이름을 바꿔 둔다(삭제하지 말 것) |

## 파일 위치

| 무엇 | 어디 |
|---|---|
| 코드 | `~/end2end/gongsa_ing` (브랜치 `end2end`) |
| 기준값·분리·갤러리 목록 | `~/end2end/artifacts/{params,separation,gallery}.json` |
| 대시보드 DB | `~/end2end/data/dashboard.sqlite3` |
| 로그 / PID | `~/end2end/logs/{mmr,agent}.{log,pid}` |
| 환경 변수 | `~/end2end/e2e_env.sh` |
| 읽기 전용 입력 | MMR 체크포인트·점수(`MMR_OUT`), Qwen 저장 결과(`VELM_RESULTS`), AeBAD(`DATA_ROOT`) |
