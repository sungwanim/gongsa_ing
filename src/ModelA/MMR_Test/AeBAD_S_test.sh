#!/bin/bash
# evaluation only from a trained checkpoint
# usage: bash AeBAD_S_test.sh <path/to/MMR_aebad_S_AeBAD_S.pth> [extra --opts]
cd "$(dirname "$0")"
CKPT="$1"; shift
python3 main.py --cfg method_config/AeBAD_S/MMR.yaml \
--opts NUM_GPUS 1 RNG_SEED 54 OUTPUT_DIR './log_MMR_AeBAD_S_eval' \
TRAIN.enable False TEST.enable True TEST.checkpoint "$CKPT" "$@"
