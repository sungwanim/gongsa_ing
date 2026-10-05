#!/usr/bin/env bash
# 서버 통합 점검 자동화: MMR -> 에이전트(Qwen) -> 대시보드 흐름을 순서대로 점검하고 결과표(report.md)를 만든다.
#
#   bash scripts/e2e.sh <명령>
#     init         폴더·설정 파일 만들기 (설정: ~/end2end/e2e.conf)
#     check        코드·conda 환경·GPU·입력 파일 점검 (읽기 전용)
#     params       기준값·분리 목록 생성 + 겹침 검사 (offline/build_params.py)   [--force: 있어도 다시]
#     verify-mmr   MMR 온라인 추론이 기존 점수와 같은지 검증, GPU 메모리·시간 측정
#     up           서비스 시작 (MMR -> 에이전트), 준비될 때까지 대기
#     test         통합·안전 테스트 (검사 N장, 이벤트·시간, 경로 비노출, 참고 이미지 경고 등)
#     dashboard    대시보드 DB 관리: dashboard list | dashboard clear --test-only|--all [--yes]  (삭제 전 자동 백업)
#     status       서비스·GPU 상태
#     down         서비스 종료
#     ts-setup     (sudo 없이) Tailscale 을 ~/end2end/tailscale 에 받아 사용자 영역 모드로 실행하고 로그인 주소를 안내
#     ts-down      ts-setup 으로 띄운 tailscaled 종료
#     expose       Tailscale 주소 + 접근 토큰으로 에이전트를 다시 시작 (외부 접속용)
#     unexpose     expose 설정 해제(로컬 전용으로 복귀)
#     tunnel       Tailscale 이 없을 때 쓰는 SSH 로컬 터널 안내 (서버에는 아무것도 새로 열지 않음)
#     report       결과표 출력 (~/end2end/logs/report.md)
#     all          init -> check -> params -> verify-mmr -> up -> test  (실패하면 중단, KEEP_GOING=1 이면 계속)
#
# 규칙: 설치(pip/conda install)·sudo·git push 를 하지 않는다. 팀원 폴더와 MMR 결과 폴더는 읽기만 한다.
# 새 파일은 $E2E_HOME(기본 ~/end2end) 아래에만 만든다. 원본 로그는 $E2E_HOME/logs/ 에 남는다.
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
E2E_HOME="${E2E_HOME:-$HOME/end2end}"
LOGS="$E2E_HOME/logs"
RESULTS="$LOGS/results.tsv"
mkdir -p "$LOGS"
[ -f "$E2E_HOME/e2e.conf" ] && . "$E2E_HOME/e2e.conf"          # 사용자 설정(선택)
[ -f "$E2E_HOME/expose.env" ] && . "$E2E_HOME/expose.env"      # expose 가 만든 외부 접속 설정(AGENT_HOST, AGENT_TOKEN)
[ -f "$E2E_HOME/tailscale.env" ] && . "$E2E_HOME/tailscale.env"  # ts-setup 이 만든 사용자 영역 Tailscale 소켓

E2E_ROOT="${E2E_ROOT:-$(cd "$E2E_SRC/../.." && pwd)}"
DATA_ROOT_USER="${DATA_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-$E2E_ROOT/AeBAD}"
if [ -z "$DATA_ROOT_USER" ] && [ ! -d "$DATA_ROOT/AeBAD_S" ]; then DATA_ROOT="/home/team14/gongsa_ing/AeBAD"; fi
MMR_OUT="${MMR_OUT:-/home/team14/gongsa_ing/src/ModelA/MMR_Test/log_MMR_AeBAD_S_54}"
MMR_CKPT="${MMR_CKPT:-$MMR_OUT/checkpoints/MMR_aebad_S_AeBAD_S.pth}"
VELM_RESULTS="${VELM_RESULTS:-/home/team14/gongsa_velm/src/ModelB/velm/results}"
ART="${ART:-$E2E_HOME/artifacts}"
E2E_DATA="${E2E_DATA:-$E2E_HOME/data}"
MMR_PORT="${MMR_PORT:-8101}"
AGENT_PORT="${AGENT_PORT:-8200}"
MMR_ENV="${MMR_ENV:-mmr}"
AGENT_ENV="${AGENT_ENV:-velm_qwen}"
N_TEST="${N_TEST:-3}"
LATENCY_WARN="${LATENCY_WARN:-30}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-2000}"          # 시작 전에 우리 GPU 조각이 이 값(MiB) 이상 쓰이고 있으면 경고
FAILS=0
WARNS=0

# ------------------------------------------------------------------ 출력/기록 도구
ok()   { echo "[PASS] $*"; }
bad()  { echo "[FAIL] $*"; FAILS=$((FAILS + 1)); }
wr()   { echo "[WARN] $*"; WARNS=$((WARNS + 1)); }
info() { echo "[INFO] $*"; }
head_() { echo; echo "== $* =="; }

# res <키> <라벨> <값> <PASS|WARN|FAIL|INFO>   결과표에 기록하고 한 줄 출력한다
res() {
  local key="$1" label="$2" value="$3" status="$4"
  touch "$RESULTS"
  awk -F'\t' -v k="$key" '$1 != k' "$RESULTS" > "$RESULTS.tmp" && mv "$RESULTS.tmp" "$RESULTS"
  printf '%s\t%s\t%s\t%s\n' "$key" "$label" "$value" "$status" >> "$RESULTS"
  case "$status" in
    PASS) ok "$label: $value" ;;
    WARN) wr "$label: $value" ;;
    FAIL) bad "$label: $value" ;;
    *) info "$label: $value" ;;
  esac
}

run_in_env() {   # run_in_env <conda 환경> <작업 폴더> <명령...>   (서브셸이라 현재 셸을 바꾸지 않는다)
  local env="$1" dir="$2"; shift 2
  ( source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$env" && cd "$dir" && "$@" )
}

need_conda() { command -v conda >/dev/null 2>&1 || { bad "conda 를 찾을 수 없습니다 (이 셸에서 conda 가 보여야 합니다)"; return 1; }; }

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
port_open() { python3 - "$1" "$2" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
}

gpu_mib() {   # "사용 전체" (MiB). MIG 조각이 있으면 그 표에서 읽는다
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi 2>/dev/null | python3 -c '
import re, sys
t = sys.stdin.read()
i = t.find("MIG devices")
m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", t[i:] if i >= 0 else t)
print("{} {}".format(m.group(1), m.group(2)) if m else "")'
}

agent_bind() { echo "${AGENT_HOST:-127.0.0.1}"; }
agent_url() { echo "http://$(agent_bind):$AGENT_PORT"; }

json_get() {   # json_get <파일> <키...>  ->  값 (없으면 빈 문자열)
  python3 - "$@" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    d = d.get(k) if isinstance(d, dict) else None
print("" if d is None else d)
PY
}

show_tail_on_fail() {   # <로그 파일> [줄 수]
  echo "---- 로그 마지막 ${2:-25}줄 ($1) ----"; tail -n "${2:-25}" "$1" | cut -c1-220; echo "----"
}

# ------------------------------------------------------------------ init
cmd_init() {
  head_ "init"
  mkdir -p "$ART" "$E2E_DATA" "$LOGS"
  if [ ! -f "$E2E_HOME/e2e.conf" ]; then
    cat > "$E2E_HOME/e2e.conf" <<EOF
# e2e.sh 설정(선택). 기본값이 맞지 않을 때만 주석을 풀어 고친다.
#E2E_ROOT="$E2E_ROOT"
#DATA_ROOT="$DATA_ROOT"          # AeBAD 폴더 (그 아래 AeBAD_S/)
#MMR_OUT="$MMR_OUT"
#MMR_CKPT="$MMR_CKPT"
#VELM_RESULTS="$VELM_RESULTS"
#MMR_PORT=$MMR_PORT
#AGENT_PORT=$AGENT_PORT
#MMR_ENV=$MMR_ENV
#AGENT_ENV=$AGENT_ENV
#N_TEST=$N_TEST                  # test 에서 검사할 갤러리 이미지 수
#LATENCY_WARN=$LATENCY_WARN      # 이미지당 허용 시간(초), 넘으면 경고
#AGENT_REFS_MANIFEST=            # 참고 이미지 목록(기본: holdout_manifest_12.csv). 바꾸면 params 도 같은 값으로 다시 실행
#AGENT_REF_PX=128                # GPU 메모리가 부족하면 96 등으로 낮춤
EOF
    info "설정 파일 생성: $E2E_HOME/e2e.conf"
  fi
  info "E2E_ROOT=$E2E_ROOT"
  info "DATA_ROOT=$DATA_ROOT"
  info "산출물 폴더: $ART (기준값), $E2E_DATA (대시보드 DB), $LOGS (로그·결과표)"
}

# ------------------------------------------------------------------ check
cmd_check() {
  head_ "1. 코드"
  if git -C "$E2E_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    local br cm dirty
    br="$(git -C "$E2E_ROOT" branch --show-current)"; cm="$(git -C "$E2E_ROOT" log --oneline -1 | cut -c1-80)"
    dirty="$(git -C "$E2E_ROOT" status --short | wc -l | tr -d ' ')"
    if [ "$br" = "end2end" ]; then res code "코드 (브랜치@커밋)" "$br @ $cm" PASS; else res code "코드 (브랜치@커밋)" "$br @ $cm (end2end 가 아님: git fetch origin end2end && git checkout end2end && git pull)" WARN; fi
    [ "$dirty" = "0" ] || wr "서버 폴더에 커밋되지 않은 변경이 ${dirty}개 있습니다 (서버에서는 파일을 고치지 않는 것이 원칙)"
  else
    res code "코드 (브랜치@커밋)" "$E2E_ROOT 가 git 저장소가 아님" FAIL
  fi

  head_ "2. conda 환경"
  if need_conda; then
    # conda run 은 출력 형식이 환경마다 달라(빈 줄 등) 쓰지 않는다. 활성화 후 실행하고 마지막 비어 있지 않은 줄을 읽는다.
    local env want out raw key pair rest
    for pair in "$MMR_ENV:2.1:mmr" "$AGENT_ENV:2.8:agent"; do
      env="${pair%%:*}"; rest="${pair#*:}"; want="${rest%%:*}"; key="${rest##*:}"
      if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
        res "env_$key" "환경 $env" "conda 환경이 없음 (conda env list 로 이름 확인, 다르면 e2e.conf 의 MMR_ENV/AGENT_ENV)" FAIL
        continue
      fi
      raw="$(run_in_env "$env" "$E2E_SRC" python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
      out="$(printf '%s\n' "$raw" | grep -v '^[[:space:]]*$' | tail -1)"
      oneline="$(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-220)"
      if echo "$raw" | grep -qE "iJIT_NotifyEvent|undefined symbol|libtorch|libtorchaudio|OSError"; then
        res "env_$key" "환경 $env (torch)" "라이브러리 충돌 증상 — 환경이 훼손된 것으로 보임. 설치하지 말고 알려 주세요. 원문: $oneline" FAIL
      elif echo "$raw" | grep -qE "ModuleNotFoundError|ImportError|Traceback"; then
        res "env_$key" "환경 $env (torch)" "torch 를 불러오지 못함 (설치하지 말고 알려 주세요). 원문: $oneline" FAIL
      elif echo "$out" | grep -q "^$want" && echo "$out" | grep -q "True"; then
        res "env_$key" "환경 $env (torch, GPU)" "$out" PASS
      else
        res "env_$key" "환경 $env (torch, GPU)" "기대 $want.x + GPU True, 실제 출력: ${oneline:-(출력 없음)}" FAIL
      fi
    done
    if conda env list | awk '{print $1}' | grep -qx "$MMR_ENV"; then
      raw="$(run_in_env "$MMR_ENV" "$E2E_SRC" python -c 'import timm,scipy,sklearn,torchvision;print("timm", timm.__version__)' 2>&1)"
      out="$(printf '%s\n' "$raw" | grep -v '^[[:space:]]*$' | tail -1)"
      if echo "$out" | grep -q "^timm"; then res pkg_mmr "mmr 패키지(timm, scipy, sklearn, torchvision)" "$out" PASS
      else res pkg_mmr "mmr 패키지(timm, scipy, sklearn, torchvision)" "import 실패: $(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-220)" FAIL; fi
    fi
    if conda env list | awk '{print $1}' | grep -qx "$AGENT_ENV"; then
      raw="$(run_in_env "$AGENT_ENV" "$E2E_SRC" python -c 'import transformers,sklearn,scipy,PIL;from qwen_vl_utils import process_vision_info;print("transformers", transformers.__version__)' 2>&1)"
      out="$(printf '%s\n' "$raw" | grep -v '^[[:space:]]*$' | tail -1)"
      if echo "$out" | grep -q "^transformers"; then res pkg_agent "velm_qwen 패키지(transformers, sklearn, scipy, PIL, qwen_vl_utils)" "$out" PASS
      else res pkg_agent "velm_qwen 패키지(transformers, sklearn, scipy, PIL, qwen_vl_utils)" "import 실패: $(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-220)" FAIL; fi
    fi
  fi

  head_ "3. GPU / 디스크"
  local g used total
  g="$(gpu_mib)"
  if [ -z "$g" ]; then
    res gpu_before "GPU 사용량(시작 전)" "nvidia-smi 를 읽지 못함" WARN
  else
    used="${g% *}"; total="${g#* }"
    if [ "$used" -ge "$GPU_BUSY_MIB" ]; then res gpu_before "GPU 사용량(시작 전)" "${used}MiB / ${total}MiB (다른 작업이 우리 조각을 쓰는 중일 수 있음)" WARN
    else res gpu_before "GPU 사용량(시작 전)" "${used}MiB / ${total}MiB" PASS; fi
  fi
  info "디스크: $(df -h "$HOME" | tail -1 | awk '{print "남은 " $4 " (사용 " $5 ")"}')"

  head_ "4. 입력 파일 (읽기 전용)"
  if [ -f "$MMR_CKPT" ]; then res in_ckpt "MMR 체크포인트" "$(du -h "$MMR_CKPT" | cut -f1)  ($MMR_CKPT)" PASS; else res in_ckpt "MMR 체크포인트" "없음: $MMR_CKPT" FAIL; fi
  local n
  if [ -d "$DATA_ROOT/AeBAD_S/test" ]; then
    n="$(find "$DATA_ROOT/AeBAD_S/test" -type f -name '*.png' ! -name '._*' | wc -l | tr -d ' ')"
    if [ "$n" = "1639" ]; then res in_images "테스트 이미지 수 (DATA_ROOT=$DATA_ROOT)" "$n장" PASS; else res in_images "테스트 이미지 수 (DATA_ROOT=$DATA_ROOT)" "${n}장 (기대 1639)" WARN; fi
  else
    res in_images "테스트 이미지 (DATA_ROOT=$DATA_ROOT)" "폴더 없음: $DATA_ROOT/AeBAD_S/test" FAIL
  fi
  local nc nn
  nc="$(ls "$MMR_OUT"/image_scores_*.csv 2>/dev/null | wc -l | tr -d ' ')"; nn="$(ls "$MMR_OUT"/anomaly_maps_*.npz 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$nc" = "4" ] && [ "$nn" = "4" ]; then res in_mmr_files "MMR 점수 csv / 이상 맵 npz (build_params 가 읽음)" "$nc개 / $nn개" PASS; else res in_mmr_files "MMR 점수 csv / 이상 맵 npz (build_params 가 읽음)" "$nc개 / $nn개 (기대 4 / 4), MMR_OUT=$MMR_OUT" FAIL; fi
  local miss="" lines=""
  for f in qwen_results_v8_n12.jsonl qwen_results_v7.jsonl; do
    if [ -f "$VELM_RESULTS/$f" ]; then lines="$lines $f=$(wc -l < "$VELM_RESULTS/$f" | tr -d ' ')줄"; else miss="$miss $f"; fi
  done
  if [ -z "$miss" ]; then res in_qwen "Qwen 저장 결과(build_params 가 읽음)" "$lines" PASS; else res in_qwen "Qwen 저장 결과(build_params 가 읽음)" "없음:$miss (VELM_RESULTS=$VELM_RESULTS)" FAIL; fi
  miss=""; lines=""
  for f in holdout_manifest.csv holdout_manifest_12.csv holdout_manifest_60.csv; do
    if [ -f "$E2E_ROOT/src/ModelB/velm/$f" ]; then lines="$lines $f=$(wc -l < "$E2E_ROOT/src/ModelB/velm/$f" | tr -d ' ')줄"; else miss="$miss $f"; fi
  done
  if [ -z "$miss" ]; then res in_manifests "참고/제외 목록(헤더 포함 줄 수)" "$lines" PASS; else res in_manifests "참고/제외 목록" "없음:$miss" FAIL; fi
  if ls ~/.cache/torch/hub/checkpoints/wide_resnet50_2-* >/dev/null 2>&1; then res in_teacher "MMR 교사 가중치 캐시" "있음" PASS
  else res in_teacher "MMR 교사 가중치 캐시" "없음 (처음 실행 때 내려받음, 인터넷 필요)" WARN; fi
}

# ------------------------------------------------------------------ params
cmd_params() {
  head_ "기준값·분리 목록 생성 (build_params.py)"
  need_conda || return 1
  if [ -f "$ART/params.json" ] && [ "${1:-}" != "--force" ]; then
    info "이미 있음: $ART/params.json (다시 만들려면: e2e.sh params --force)"
    res params "기준값 생성" "기존 산출물 사용 ($ART)" PASS
  else
    mkdir -p "$ART"
    local extra=()
    [ -n "${AGENT_REFS_MANIFEST:-}" ] && extra=(--refs-manifest "$AGENT_REFS_MANIFEST")
    run_in_env "$AGENT_ENV" "$E2E_SRC/offline" python build_params.py --mmr-out "$MMR_OUT" --results-dir "$VELM_RESULTS" \
      --data-root "$DATA_ROOT" --out "$ART" "${extra[@]+"${extra[@]}"}" > "$LOGS/params.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      local errline; errline="$(grep -v '^[[:space:]]*$' "$LOGS/params.log" | tail -3 | tr '\n' ' ' | cut -c1-300)"
      res params "기준값 생성" "실패 (종료 코드 $rc). 로그 끝부분: $errline" FAIL
      if grep -q "분리 위반" "$LOGS/params.log"; then bad "참고/보정/갤러리 이미지가 겹침 — 조용히 넘어가지 않는 것이 정상 동작입니다. 로그를 알려 주세요"; fi
      if grep -q "라우터 재현" "$LOGS/params.log"; then bad "라우터 재현이 기존 계산과 다름 — 로그를 알려 주세요"; fi
      show_tail_on_fail "$LOGS/params.log" 20
      return 1
    fi
  fi
  run_in_env "$AGENT_ENV" "$E2E_SRC/offline" python separation.py --artifacts "$ART" > "$LOGS/separation.log" 2>&1
  if [ $? -ne 0 ]; then res params_sep "분리 검사" "$(tail -1 "$LOGS/separation.log" | cut -c1-160)" FAIL; return 1; fi
  local sj="$ART/summary.json"
  if [ ! -f "$sj" ]; then wr "summary.json 이 없습니다 (이전 버전으로 만든 산출물). e2e.sh params --force 로 다시 만드세요"; return 0; fi
  res params_sep "참고/보정/갤러리 겹침 검사" "$(tail -1 "$LOGS/separation.log")" PASS
  res params_values "구간 기준 / 임계값" "tau_lo=$(json_get "$sj" tau_lo) tau_hi=$(json_get "$sj" tau_hi) t=$(json_get "$sj" t)" INFO
  res params_counts "보정 / 보고 / 참고 / 갤러리 장수" "$(json_get "$sj" n_calib) / $(json_get "$sj" n_report) / $(json_get "$sj" n_refs) / $(json_get "$sj" n_gallery)  (참고 목록: $(json_get "$sj" refs_manifest))" INFO
  local have total
  have="$(json_get "$sj" qwen_calib_have)"; total="$(json_get "$sj" qwen_calib_total)"
  local eff mz; eff="$(json_get "$sj" qwen_missing_effective)"; mz="$(json_get "$sj" qwen_missing_by_zone)"
  if [ "$have" = "$total" ]; then res params_qwen "Qwen 저장 결과가 보정용 이미지를 덮는 비율" "$have / $total" PASS
  elif [ "$eff" = "0" ]; then res params_qwen "Qwen 저장 결과가 보정용 이미지를 덮는 비율" "$have / $total (누락 $((total - have))장은 모두 정상 확정 구간이라 임계값 t 계산에 영향 없음: $mz)" PASS
  elif [ -z "$eff" ]; then res params_qwen "Qwen 저장 결과가 보정용 이미지를 덮는 비율" "$have / $total (누락의 구간 분포를 모름: e2e.sh params --force 로 다시 만들면 판정됨)" WARN
  else res params_qwen "Qwen 저장 결과가 보정용 이미지를 덮는 비율" "$have / $total (누락 중 ${eff}장이 애매·불량 구간이라 t 가 기존 최종값과 다를 수 있음: $mz)" WARN; fi
  res params "기준값 생성" "완료 ($ART)" PASS      # 이전 실행의 실패 기록을 덮어쓴다
}

# ------------------------------------------------------------------ verify-mmr
cmd_verify_mmr() {
  head_ "MMR 온라인 추론 검증 (verify_mmr.py)"
  need_conda || return 1
  [ -f "$MMR_CKPT" ] || { res mmr_diff "MMR 점수 일치" "체크포인트 없음" FAIL; return 1; }
  local worst="0" rc=0 j
  : > "$LOGS/verify_mmr.log"
  for dom in same view; do
    local n=10; [ "$dom" = "view" ] && n=5
    j="$LOGS/verify_mmr_$dom.json"; rm -f "$j"
    run_in_env "$MMR_ENV" "$E2E_SRC/mmr_service" python verify_mmr.py --ckpt "$MMR_CKPT" \
      --csv "$MMR_OUT/image_scores_aebad_S_AeBAD_S_$dom.csv" --data-root "$DATA_ROOT" -n "$n" --json-out "$j" >> "$LOGS/verify_mmr.log" 2>&1 || rc=1
  done
  if [ $rc -ne 0 ] || [ ! -f "$LOGS/verify_mmr_same.json" ]; then
    res mmr_diff "MMR 점수 일치" "검증 실행 실패. 로그 끝부분: $(grep -v '^[[:space:]]*$' "$LOGS/verify_mmr.log" | tail -3 | tr '\n' ' ' | cut -c1-300)" FAIL; show_tail_on_fail "$LOGS/verify_mmr.log" 20; return 1
  fi
  local d1 d2 passed
  d1="$(json_get "$LOGS/verify_mmr_same.json" max_diff)"; d2="$(json_get "$LOGS/verify_mmr_view.json" max_diff 2>/dev/null)"; d2="${d2:-0}"
  worst="$(python3 -c "print(max($d1, $d2))")"
  passed="$(python3 -c "print('1' if $worst < 1e-3 else '0')")"
  if [ "$passed" = "1" ]; then res mmr_diff "MMR 온라인 vs 기존 CSV 점수 최대 차이" "$worst (기준 1e-3 미만, 이미지 $(( 10 + 5 ))장)" PASS
  else res mmr_diff "MMR 온라인 vs 기존 CSV 점수 최대 차이" "$worst  -> 불일치. 다음 단계로 가지 말고 로그를 알려 주세요" FAIL; show_tail_on_fail "$LOGS/verify_mmr.log" 25; return 1; fi
  res mmr_load "MMR 모델 로드 시간(요청마다 로드하는 방식, 중앙값)" "$(json_get "$LOGS/verify_mmr_same.json" load_s_median)초" INFO
  res mmr_infer "MMR 추론 시간(중앙값)" "$(json_get "$LOGS/verify_mmr_same.json" infer_s_median)초" INFO
  res mmr_gpu "MMR GPU 피크 메모리" "$(json_get "$LOGS/verify_mmr_same.json" gpu_peak_mb_max) MiB" INFO
}

# ------------------------------------------------------------------ 서비스
wait_ready() {   # <url> <대기 초> <pid 파일> <기대 문자열>   0=준비됨, 1=시간 초과, 2=프로세스 종료
  local url="$1" timeout="$2" pidf="$3" pat="$4" t=0
  while [ "$t" -lt "$timeout" ]; do
    alive "$pidf" || return 2
    curl -s -m 3 "$url" 2>/dev/null | grep -q "$pat" && return 0
    sleep 3; t=$((t + 3)); printf '.'
  done
  return 1
}

start_mmr() {
  if alive "$LOGS/mmr.pid"; then info "MMR 서비스는 이미 실행 중 (pid $(cat "$LOGS/mmr.pid"))"; return 0; fi
  if port_open 127.0.0.1 "$MMR_PORT"; then res svc_mmr "MMR 서비스" "포트 $MMR_PORT 가 이미 사용 중 (다른 프로세스). e2e.conf 의 MMR_PORT 를 바꾸세요" FAIL; return 1; fi
  # 주의: `cd x && cmd &` 는 묶음 전체가 백그라운드가 되어 $! 가 서비스 PID 가 아니게 된다 -> cd 를 먼저 하고 서비스만 백그라운드로 보낸다
  ( cd "$E2E_SRC" || exit 1
    MMR_CKPT="$MMR_CKPT" MMR_PORT="$MMR_PORT" MMR_ENV="$MMR_ENV" nohup bash scripts/run_mmr.sh > "$LOGS/mmr.log" 2>&1 &
    echo $! > "$LOGS/mmr.pid" )
  printf '[INFO] MMR 서비스 시작 대기'
  wait_ready "http://127.0.0.1:$MMR_PORT/health" 120 "$LOGS/mmr.pid" '"status": "ok"'; local rc=$?; echo
  if [ $rc -eq 0 ]; then res svc_mmr "MMR 서비스" "http://127.0.0.1:$MMR_PORT (요청마다 모델 로드·해제)" PASS; return 0; fi
  res svc_mmr "MMR 서비스" "$([ $rc -eq 2 ] && echo '시작 직후 종료됨' || echo '시간 초과'). 로그 끝부분: $(grep -v '^[[:space:]]*$' "$LOGS/mmr.log" | tail -3 | tr '\n' ' ' | cut -c1-300)" FAIL; show_tail_on_fail "$LOGS/mmr.log" 25; return 1
}

start_agent() {
  if alive "$LOGS/agent.pid"; then info "에이전트 서비스는 이미 실행 중 (pid $(cat "$LOGS/agent.pid"))"; return 0; fi
  [ -f "$ART/params.json" ] || { res svc_agent "에이전트 서비스" "기준값이 없습니다. 먼저 e2e.sh params" FAIL; return 1; }
  if port_open "$(agent_bind)" "$AGENT_PORT"; then res svc_agent "에이전트 서비스" "포트 $AGENT_PORT 가 이미 사용 중 (다른 프로세스). e2e.conf 의 AGENT_PORT 를 바꾸세요" FAIL; return 1; fi
  ( export AGENT_ARTIFACTS="$ART" AGENT_DATA_ROOT="$DATA_ROOT" AGENT_DATA_DIR="$E2E_DATA" AGENT_MMR_URL="http://127.0.0.1:$MMR_PORT" \
           AGENT_PORT="$AGENT_PORT" AGENT_ENV="$AGENT_ENV"
    [ -n "${AGENT_HOST:-}" ] && export AGENT_HOST
    [ -n "${AGENT_TOKEN:-}" ] && export AGENT_TOKEN
    [ -n "${AGENT_REQUIRE_TOKEN:-}" ] && export AGENT_REQUIRE_TOKEN
    [ -n "${AGENT_REFS_MANIFEST:-}" ] && export AGENT_REFS_MANIFEST
    [ -n "${AGENT_REF_PX:-}" ] && export AGENT_REF_PX
    cd "$E2E_SRC" || exit 1
    nohup bash scripts/run_agent.sh > "$LOGS/agent.log" 2>&1 &
    echo $! > "$LOGS/agent.pid" )
  printf '[INFO] 에이전트 서비스 시작 대기 (Qwen 로딩 1~3분)'
  wait_ready "$(agent_url)/api/health" 480 "$LOGS/agent.pid" '"ready": true'; local rc=$?; echo
  if [ $rc -eq 0 ]; then res svc_agent "에이전트 서비스" "$(agent_url) (인증: $([ -n "${AGENT_TOKEN:-}" ] && echo 토큰 || echo '없음, 로컬 전용'))" PASS; return 0; fi
  res svc_agent "에이전트 서비스" "$([ $rc -eq 2 ] && echo '시작 직후 종료됨' || echo '시간 초과'). 로그 끝부분: $(grep -v '^[[:space:]]*$' "$LOGS/agent.log" | tail -3 | tr '\n' ' ' | cut -c1-300)" FAIL
  grep -E "MMR 서비스에 연결|참고 이미지를 찾을|변경되었습니다|분리 위반|OutOfMemory|out of memory|ImportError|iJIT|AGENT_TOKEN" "$LOGS/agent.log" | head -5 | cut -c1-200
  show_tail_on_fail "$LOGS/agent.log" 25; return 1
}

cmd_up() {
  head_ "서비스 시작 (MMR -> 에이전트)"
  start_mmr || return 1
  start_agent || return 1
  local g; g="$(gpu_mib)"
  if [ -n "$g" ]; then res gpu_after_agent "Qwen 상주 후 GPU 사용량" "${g% *}MiB / ${g#* }MiB" INFO; fi
  return 0
}

cmd_dashboard() {   # dashboard list | clear [--test-only|--all] [--yes]
  head_ "대시보드 DB ($E2E_DATA)"
  python3 "$E2E_SRC/agent_service/dashboard_admin.py" --data-dir "$E2E_DATA" "$@"
}

cmd_status() {
  head_ "상태"
  for s in mmr agent; do
    if alive "$LOGS/$s.pid"; then info "$s: 실행 중 (pid $(cat "$LOGS/$s.pid"))"; else info "$s: 꺼져 있음"; fi
  done
  curl -s -m 3 "http://127.0.0.1:$MMR_PORT/health" 2>/dev/null | sed 's/^/[INFO] MMR health: /'; echo
  curl -s -m 3 "$(agent_url)/api/health" 2>/dev/null | sed 's/^/[INFO] 에이전트 health: /'; echo
  local g; g="$(gpu_mib)"
  if [ -n "$g" ]; then info "GPU 조각 사용량: ${g% *}MiB / ${g#* }MiB"; fi
  return 0
}

stop_service() {   # <mmr|agent>  : 종료 신호 후 최대 45초 기다린다. 아직 살아 있으면 1 (강제 종료는 하지 않는다)
  local s="$1" t=0
  alive "$LOGS/$s.pid" || return 0
  kill "$(cat "$LOGS/$s.pid")" 2>/dev/null
  while alive "$LOGS/$s.pid" && [ "$t" -lt 45 ]; do sleep 1.5; t=$((t + 2)); done
  alive "$LOGS/$s.pid" && return 1
  rm -f "$LOGS/$s.pid"; return 0
}

cmd_down() {
  head_ "서비스 종료"
  for s in agent mmr; do
    if ! alive "$LOGS/$s.pid"; then info "$s: 이미 꺼져 있음"; continue; fi
    if stop_service "$s"; then ok "$s 종료"; else wr "$s 가 아직 종료되지 않았습니다 (pid $(cat "$LOGS/$s.pid")). 잠시 후 e2e.sh status 로 확인하세요 (강제 종료는 하지 않습니다)"; fi
  done
}

# ------------------------------------------------------------------ test
cmd_test() {
  head_ "통합·안전 테스트"
  curl -s -m 5 "$(agent_url)/api/health" | grep -q '"ready": true' || { res test_summary "통합 테스트" "에이전트 서비스가 준비되어 있지 않음 (e2e.sh up 먼저)" FAIL; return 1; }
  local tok="${AGENT_TOKEN:-}"
  python3 "$E2E_SRC/dev/integration_test.py" --url "$(agent_url)" ${tok:+--token "$tok" --expect-auth} \
    --artifacts "$ART" --data-root "$DATA_ROOT" -n "$N_TEST" --latency-warn "$LATENCY_WARN" --json-out "$LOGS/test.json" 2>&1 | tee "$LOGS/test.log"
  local rc=${PIPESTATUS[0]} j="$LOGS/test.json"
  # 이 점검이 만든 테스트 검사 기록은 대시보드에 남기지 않는다 (실제 검사 기록은 건드리지 않음)
  python3 "$E2E_SRC/agent_service/dashboard_admin.py" --data-dir "$E2E_DATA" clear --test-only --yes --quiet >/dev/null 2>&1 || true
  [ -f "$j" ] || { res test_summary "통합 테스트" "결과 파일 없음" FAIL; return 1; }
  local p f w; p="$(json_get "$j" pass)"; f="$(json_get "$j" fail)"; w="$(json_get "$j" warn)"
  if [ "$f" = "0" ]; then res test_summary "통합·안전 테스트" "PASS $p / WARN $w / FAIL $f" "$([ "$w" = "0" ] && echo PASS || echo WARN)"
  else res test_summary "통합·안전 테스트" "PASS $p / WARN $w / FAIL $f" FAIL; show_tail_on_fail "$LOGS/agent.log" 15; fi
  local first rest mx
  python3 - "$j" "$RESULTS" <<'PY'
import json, sys
runs = json.load(open(sys.argv[1])).get("runs", [])
txt = "; ".join("#{} {} {} -> {}{} ({:.1f}초)".format(i, r.get("zone"), ">".join(r.get("tools") or []), r.get("decision"),
                "/" + r["defect_type"] if r.get("defect_type") else "", r.get("total_s") or 0) for i, r in enumerate(runs))
if txt:
    lines = [l for l in open(sys.argv[2]) if not l.startswith("test_runs\t")]
    lines.append("test_runs\t검사별 결과 (구간, 도구 순서 -> 판정/종류, 시간)\t{}\tINFO\n".format(txt))
    open(sys.argv[2], "w").writelines(lines)
PY
  first="$(json_get "$j" first_total_s)"; rest="$(json_get "$j" rest_median_s)"; mx="$(json_get "$j" max_total_s)"
  if [ -n "$first" ]; then res test_time "이미지당 소요 시간 (첫 이미지 / 이후 중앙값 / 최대)" "${first}초 / ${rest:--}초 / ${mx}초  (기준 ${LATENCY_WARN}초)" INFO; fi
  local g; g="$(gpu_mib)"
  if [ -n "$g" ]; then res test_gpu "검사 후 GPU 사용량" "${g% *}MiB / ${g#* }MiB" INFO; fi
  grep -E "캐시 확인|캐시 사용 불가" "$LOGS/agent.log" 2>/dev/null | tail -2 | cut -c1-200 | sed 's/^/[INFO] 에이전트 로그: /'
  [ "$rc" -eq 0 ]
}

# ------------------------------------------------------------------ expose / unexpose
TS_HOME="${TS_HOME:-$E2E_HOME/tailscale}"

ts_bin() {   # tailscale CLI 경로: TAILSCALE_BIN(지정 시 그것만, 테스트용) > PATH > ts-setup 이 받은 것
  if [ -n "${TAILSCALE_BIN:-}" ]; then [ -x "$TAILSCALE_BIN" ] && echo "$TAILSCALE_BIN"
  elif command -v tailscale >/dev/null 2>&1; then command -v tailscale
  elif [ -x "$TS_HOME/tailscale" ]; then echo "$TS_HOME/tailscale"
  fi
}

ts_cli() {   # TAILSCALE_SOCKET 이 있으면(사용자 영역 모드) 그 소켓으로 tailscale 을 실행
  local bin; bin="$(ts_bin)" || return 1
  [ -n "$bin" ] || return 1
  if [ -n "${TAILSCALE_SOCKET:-}" ]; then "$bin" --socket="$TAILSCALE_SOCKET" "$@"
  elif [ -S "$TS_HOME/tailscaled.sock" ] && [ "$bin" = "$TS_HOME/tailscale" ]; then "$bin" --socket="$TS_HOME/tailscaled.sock" "$@"
  else "$bin" "$@"; fi
}

ip_is_local() {   # 이 서버의 네트워크 인터페이스에 붙어 있는 주소인가 (사용자 영역 모드에서는 아님)
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind((sys.argv[1], 0))
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}

cmd_ts_setup() {
  head_ "Tailscale 사용자 영역 설치 (sudo 없음, $TS_HOME 아래에만 설치)"
  if [ -n "$(ts_bin)" ] && ts_cli ip -4 >/dev/null 2>&1; then info "이미 연결되어 있습니다: $(ts_cli ip -4 | head -1)"; return 0; fi
  if [ -z "$(ts_bin)" ] || [ ! -x "$TS_HOME/tailscaled" ] && ! command -v tailscaled >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || { res ts_setup "Tailscale 설치" "curl 이 없습니다" FAIL; return 1; }
    local arch; case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64|arm64) arch=arm64 ;; *) res ts_setup "Tailscale 설치" "지원하지 않는 CPU: $(uname -m)" FAIL; return 1 ;; esac
    local tgz; tgz="$(curl -fsSL https://pkgs.tailscale.com/stable/ 2>/dev/null | grep -oE "tailscale_[0-9]+\.[0-9]+\.[0-9]+_${arch}\.tgz" | sort -V | tail -1)"
    if [ -z "$tgz" ]; then res ts_setup "Tailscale 설치" "내려받을 파일을 찾지 못함 (서버에서 pkgs.tailscale.com 에 접속이 안 될 수 있음. 관리자에게 설치를 요청하세요)" FAIL; return 1; fi
    info "내려받는 파일: https://pkgs.tailscale.com/stable/$tgz"
    mkdir -p "$TS_HOME"
    curl -fsSL "https://pkgs.tailscale.com/stable/$tgz" -o "$TS_HOME/$tgz" && tar -xzf "$TS_HOME/$tgz" -C "$TS_HOME" --strip-components=1 \
      || { res ts_setup "Tailscale 설치" "내려받기/풀기 실패" FAIL; return 1; }
    rm -f "$TS_HOME/$tgz"
  fi
  [ -x "$TS_HOME/tailscaled" ] || { res ts_setup "Tailscale 설치" "$TS_HOME/tailscaled 가 없습니다" FAIL; return 1; }
  if ! alive "$LOGS/tailscaled.pid"; then
    ( cd "$TS_HOME" || exit 1
      nohup "$TS_HOME/tailscaled" --tun=userspace-networking --socket="$TS_HOME/tailscaled.sock" --state="$TS_HOME/tailscaled.state" > "$LOGS/tailscaled.log" 2>&1 &
      echo $! > "$LOGS/tailscaled.pid" )
    sleep 3
    alive "$LOGS/tailscaled.pid" || { res ts_setup "tailscaled 실행" "시작 직후 종료됨" FAIL; show_tail_on_fail "$LOGS/tailscaled.log" 15; return 1; }
  fi
  local authkey=(); [ -n "${TS_AUTHKEY:-}" ] && authkey=(--authkey="$TS_AUTHKEY")
  ( "$TS_HOME/tailscale" --socket="$TS_HOME/tailscaled.sock" up --hostname="${TS_HOSTNAME:-$(hostname)-e2e}" "${authkey[@]+"${authkey[@]}"}" > "$LOGS/tailscale_up.log" 2>&1 & echo $! > "$LOGS/tailscale_up.pid" )
  local t=0 ip=""
  while [ "$t" -lt 90 ]; do
    ip="$("$TS_HOME/tailscale" --socket="$TS_HOME/tailscaled.sock" ip -4 2>/dev/null | head -1)"
    [ -n "$ip" ] && break
    if grep -qE "https://login\.tailscale\.com/[A-Za-z0-9/_-]+" "$LOGS/tailscale_up.log" 2>/dev/null && [ "$t" -eq 0 -o "$t" -eq 15 -o "$t" -eq 45 ]; then
      info "아래 주소를 브라우저(아무 기기)에서 열어 Tailscale 에 로그인하세요: $(grep -oE 'https://login\.tailscale\.com/[A-Za-z0-9/_-]+' "$LOGS/tailscale_up.log" | head -1)"
    fi
    sleep 3; t=$((t + 3))
  done
  if [ -n "$ip" ]; then
    export TAILSCALE_SOCKET="$TS_HOME/tailscaled.sock"
    printf 'export TAILSCALE_SOCKET=%s\n' "$TS_HOME/tailscaled.sock" > "$E2E_HOME/tailscale.env"
    res ts_setup "Tailscale (사용자 영역 모드)" "연결됨: $ip" PASS
    info "다음: bash scripts/e2e.sh expose"
  else
    res ts_setup "Tailscale (사용자 영역 모드)" "로그인이 완료되지 않음. 로그: $LOGS/tailscale_up.log" FAIL; show_tail_on_fail "$LOGS/tailscale_up.log" 10; return 1
  fi
}

cmd_ts_down() {
  head_ "Tailscale(사용자 영역) 종료"
  for p in tailscale_up tailscaled; do
    if alive "$LOGS/$p.pid"; then kill "$(cat "$LOGS/$p.pid")" 2>/dev/null; rm -f "$LOGS/$p.pid"; ok "$p 종료"; else info "$p: 이미 꺼져 있음"; fi
  done
}

cmd_tunnel() {
  head_ "SSH 로컬 터널 (Tailscale 이 없을 때) — 서버에는 아무것도 새로 열리지 않는다"
  info "에이전트는 127.0.0.1:$AGENT_PORT 에만 열려 있습니다. 맥에서 아래처럼 SSH 터널을 열면 맥의 localhost:$AGENT_PORT 로 접속됩니다."
  echo "  맥 터미널:  ssh -N -L ${AGENT_PORT}:127.0.0.1:${AGENT_PORT} $(whoami)@<서버 주소>"
  echo "  (Termius 를 쓴다면: 포트 포워딩 > Local, 로컬 포트 $AGENT_PORT, 목적지 127.0.0.1:$AGENT_PORT)"
  echo "  확인(맥):   curl http://127.0.0.1:$AGENT_PORT/api/health"
  echo "  프론트엔드: VITE_API_TARGET=http://127.0.0.1:$AGENT_PORT (기본값 그대로)"
  local cfg; cfg="$(grep -rhiE '^[[:space:]]*AllowTcpForwarding' /etc/ssh/sshd_config /etc/ssh/sshd_config.d 2>/dev/null | head -2 | tr '\n' ' ')"
  info "서버 SSH 설정(읽을 수 있을 때만): ${cfg:-읽을 수 없음. 'administratively prohibited' 가 나오면 관리자가 포워딩을 막은 것}"
}

cmd_expose() {
  head_ "외부 접속 설정 (Tailscale + 접근 토큰)"
  [ -f "$E2E_HOME/tailscale.env" ] && . "$E2E_HOME/tailscale.env"
  if [ -z "$(ts_bin)" ]; then
    res expose "외부 접속" "tailscale 가 없음. 관리자 설치 또는 'e2e.sh ts-setup'(sudo 없이 사용자 영역). 임시로는 'e2e.sh tunnel'(SSH 터널)" FAIL; return 1
  fi
  local ip; ip="$(ts_cli ip -4 2>/dev/null | head -1)"
  [ -n "$ip" ] || { res expose "외부 접속" "Tailscale 주소를 얻지 못함 (① 로그인: sudo tailscale up 또는 e2e.sh ts-setup  ② 권한 문제면 관리자가 'sudo tailscale set --operator=\$USER' 를 한 번 실행)" FAIL; return 1; }
  local token mode
  token="$(openssl rand -hex 24 2>/dev/null || python3 -c 'import secrets;print(secrets.token_hex(24))')"
  umask 077
  if ip_is_local "$ip"; then
    mode="바인딩: Tailscale 주소($ip)에 직접 열림"
    printf 'export AGENT_HOST=%s\nexport AGENT_TOKEN=%s\n' "$ip" "$token" > "$E2E_HOME/expose.env"
    AGENT_HOST="$ip"
  else
    mode="사용자 영역 모드: 서버에는 Tailscale 인터페이스가 없어 127.0.0.1 에 열고 tailnet 의 접속이 여기로 전달됨(토큰 필수)"
    printf 'export AGENT_TOKEN=%s\nexport AGENT_REQUIRE_TOKEN=1\n' "$token" > "$E2E_HOME/expose.env"
    unset AGENT_HOST
  fi
  AGENT_TOKEN="$token"; export AGENT_TOKEN; [ -n "${AGENT_HOST:-}" ] && export AGENT_HOST; [ -f "$E2E_HOME/expose.env" ] && grep -q REQUIRE "$E2E_HOME/expose.env" && export AGENT_REQUIRE_TOKEN=1
  info "$mode"
  if alive "$LOGS/agent.pid"; then
    info "에이전트를 새 설정으로 다시 시작합니다"
    stop_service agent || { res expose "외부 접속" "기존 에이전트가 종료되지 않음. 잠시 후 다시 실행하세요" FAIL; return 1; }
  fi
  start_agent || return 1
  N_TEST=1 cmd_test || true
  echo
  info "맥에서 접속하는 방법:"
  echo "  1) 맥의 frontend/.env 에  VITE_API_TARGET=http://$ip:$AGENT_PORT  를 넣고 npm run dev  (맥도 같은 Tailscale 계정에 로그인되어 있어야 함)"
  echo "  2) 브라우저에서 http://localhost:5173 을 열고 아래 토큰을 입력 (채팅에 붙이지 마세요)"
  echo "     토큰: $token"
  info "토큰은 $E2E_HOME/expose.env (권한 600) 에 있습니다. 해제: e2e.sh unexpose"
  res expose "외부 접속" "Tailscale 주소 $ip:$AGENT_PORT ($mode)" PASS
}

cmd_unexpose() {
  head_ "외부 접속 해제"
  rm -f "$E2E_HOME/expose.env"; unset AGENT_HOST AGENT_TOKEN AGENT_REQUIRE_TOKEN
  if alive "$LOGS/agent.pid"; then stop_service agent || wr "에이전트가 아직 종료되지 않았습니다. 잠시 후 e2e.sh status"; fi
  info "로컬 전용 설정으로 돌아갔습니다. 다시 시작: e2e.sh up"
}

# ------------------------------------------------------------------ report
cmd_report() {
  python3 - "$RESULTS" "$LOGS/report.md" "$E2E_ROOT" <<'PY'
import subprocess, sys, time
res, out, root = sys.argv[1:4]
rows = []
try:
    for line in open(res):
        p = line.rstrip("\n").split("\t")
        if len(p) == 4:
            rows.append(p)
except FileNotFoundError:
    pass
try:
    commit = subprocess.check_output(["git", "-C", root, "log", "--oneline", "-1"], text=True).strip()
except Exception:
    commit = "unknown"
mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "**FAIL**", "INFO": "-"}
lines = ["# 서버 통합 점검 결과", "", "- 시각: " + time.strftime("%Y-%m-%d %H:%M:%S"), "- 코드: " + commit, "",
         "| 항목 | 값 | 판정 |", "|---|---|---|"]
for _, label, value, status in rows:
    lines.append("| {} | {} | {} |".format(label, value.replace("|", "/"), mark.get(status, status)))
nf = sum(1 for r in rows if r[3] == "FAIL")
nw = sum(1 for r in rows if r[3] == "WARN")
lines += ["", "FAIL {}건, WARN {}건".format(nf, nw)]
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
  echo; info "결과표 파일: $LOGS/report.md  (이 내용을 그대로 전달하면 됩니다)"
}

# ------------------------------------------------------------------ all
cmd_all() {
  local steps=(init check params verify_mmr up test) s rc=0
  [ -f "$RESULTS" ] && mv "$RESULTS" "$RESULTS.prev"      # 이전 실행의 기록이 결과표에 섞이지 않게
  for s in "${steps[@]}"; do
    FAILS=0
    "cmd_$s"; rc=$?
    if [ $rc -ne 0 ] || [ "$FAILS" -gt 0 ]; then
      echo; bad "단계 '${s//_/-}' 에서 실패했습니다."
      if [ "${KEEP_GOING:-0}" != "1" ]; then echo; cmd_report; return 1; fi
    fi
  done
  echo; cmd_report
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

main() {
  local c="${1:-}"; shift || true
  case "$c" in
    init) cmd_init ;;
    check) cmd_check; cmd_report >/dev/null; [ "$FAILS" = 0 ] ;;
    params) cmd_params "${1:-}"; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] && [ "$FAILS" = 0 ] ;;
    verify-mmr) cmd_verify_mmr; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] && [ "$FAILS" = 0 ] ;;
    up) cmd_up; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] ;;
    test) cmd_test; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] ;;
    dashboard) cmd_dashboard "$@" ;;
    status) cmd_status ;;
    down) cmd_down ;;
    ts-setup) cmd_ts_setup; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] ;;
    ts-down) cmd_ts_down ;;
    tunnel) cmd_tunnel ;;
    expose) cmd_expose; local r=$?; cmd_report >/dev/null; [ $r -eq 0 ] ;;
    unexpose) cmd_unexpose ;;
    report) cmd_report ;;
    all) cmd_all ;;
    ""|-h|--help|help) usage ;;
    *) echo "알 수 없는 명령: $c"; usage; return 2 ;;
  esac
}

main "$@"
