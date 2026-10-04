#!/usr/bin/env bash
# 전체 평가 이미지(holdout 60장 제외)로 MMR 단독 / v6 / v7 / v8 비교.
#   v6 : 이미 돌린 결과 재사용 (없으면 같이 돌림)   v7, v8 : 아직 없으면 전체를 Qwen으로 돌림
# 끊겨도 이어서 실행됨(이미지별로 저장). velm_qwen 환경에서, tmux 안에서 실행 권장.
#   bash run_compare.sh
cd "$(dirname "$0")"
MMR_OUT="${MMR_OUT:-$HOME/gongsa_ing/src/ModelA/MMR_Test/log_MMR_AeBAD_S_54}"
for tag in v6 v7 v8; do
  echo "=== 프롬프트 $tag : Qwen 판정 (이미 끝난 이미지는 건너뜀) ==="
  python qwen_eval.py --mmr-out "$MMR_OUT" --prompt "$tag" --step qwen 2>&1 | grep --line-buffered -v MatMul8bitLt
done
echo "=== 비교 표 ==="
python qwen_eval.py --mmr-out "$MMR_OUT" --compare v6,v7,v8 2>&1 | grep --line-buffered -v MatMul8bitLt
echo "저장: results/comparison_all.txt, comparison_all.csv"
