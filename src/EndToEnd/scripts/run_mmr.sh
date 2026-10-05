#!/usr/bin/env bash
# MMR 서비스 실행 (conda 환경 mmr). 사용: MMR_CKPT=<MMR_*.pth> bash run_mmr.sh
# 선택: MMR_ENV(기본 mmr) MMR_RESIDENT(0=요청마다 로드·해제 [기본], 1=상주) MMR_PORT(8101) MMR_HOST(127.0.0.1) MMR_DEVICE(cuda:0)
set -euo pipefail
: "${MMR_CKPT:?MMR_CKPT(체크포인트 경로)를 지정하세요}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MMR_ENV:-mmr}"
cd "$(dirname "$0")/../mmr_service"
exec python server.py
