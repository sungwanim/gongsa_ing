#!/usr/bin/env bash
# MMR(1차) -> Qwen(2차) -> 비교 지표를 tmux 한 세션에서 실행. GPU 서버에서는 git pull 후 실행만 하고 push 금지.
#   bash run_all.sh              tmux 세션을 만들고 백그라운드 실행 (Termius 끊겨도 계속 진행)
#   tmux attach -t mmr_qwen      진행 확인 (빠져나오기: Ctrl-b 누른 뒤 d)
# 환경변수:
#   MMR_ENV   MMR conda 환경 이름 (MMR을 새로 돌릴 때 필수)
#   VELM_ENV  velm conda 환경 이름 (기본 velm_qwen)
#   SKIP_MMR=1  MMR을 다시 돌리지 않고 기존 결과 csv 사용   MMR_OUT  csv 폴더 (기본 MMR_Test/log_MMR_AeBAD_S_54)
#   CKPT=경로   학습 없이 체크포인트로 MMR 평가만 (AeBAD_S_test.sh 사용)
#   SESSION(기본 mmr_qwen)
set -euo pipefail
cd "$(dirname "$0")"
SESSION="${SESSION:-mmr_qwen}"
if [[ "${1:-}" != "--inner" ]]; then
  command -v tmux >/dev/null || { echo "tmux 가 없습니다"; exit 1; }
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "이미 '$SESSION' 세션이 있습니다: tmux attach -t $SESSION"; exit 1; }
  ENVSTR=""; for v in MMR_ENV VELM_ENV SKIP_MMR MMR_OUT CKPT; do [[ -n "${!v:-}" ]] && ENVSTR+=" $v=$(printf '%q' "${!v}")"; done
  tmux new-session -d -s "$SESSION" -n pipeline "env $ENVSTR bash $(printf '%q' "$PWD/run_all.sh") --inner; echo; echo '[끝] Enter 를 누르면 창이 닫힙니다'; read"
  echo "시작했습니다. 확인: tmux attach -t $SESSION"
  exit 0
fi

MMR_TEST="$(cd ../../ModelA/MMR_Test && pwd)"
MMR_OUT="${MMR_OUT:-$MMR_TEST/log_MMR_AeBAD_S_54}"
VELM_ENV="${VELM_ENV:-velm_qwen}"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -z "${SKIP_MMR:-}" ]]; then
  [[ -n "${MMR_ENV:-}" ]] || { echo "MMR_ENV(MMR conda 환경 이름)를 지정하세요. 예: MMR_ENV=mmr bash run_all.sh"; exit 1; }
  echo "=== [1/3] MMR (1차) ==="
  conda activate "$MMR_ENV"
  if [[ -n "${CKPT:-}" ]]; then bash "$MMR_TEST/AeBAD_S_test.sh" "$CKPT"; else bash "$MMR_TEST/AeBAD_S_run.sh"; fi
  conda deactivate
else
  echo "=== [1/3] MMR 건너뜀 (기존 결과 사용: $MMR_OUT) ==="
fi

echo "=== [2/3] Qwen (2차) + [3/3] 지표 비교 ==="
conda activate "$VELM_ENV"
python qwen_eval.py --mmr-out "$MMR_OUT"
