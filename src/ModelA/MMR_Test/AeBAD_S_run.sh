#!/bin/bash
# usage: bash AeBAD_S_run.sh [extra --opts, e.g. TRAIN_SETUPS.epochs 1]
# run from anywhere; relative paths (../../../AeBAD) resolve from this directory
cd "$(dirname "$0")"
python3 main.py --cfg method_config/AeBAD_S/MMR.yaml \
--opts NUM_GPUS 1 RNG_SEED 54 OUTPUT_DIR './log_MMR_AeBAD_S' "$@"
