#!/usr/bin/env bash
# 에이전트 서비스 실행 (conda 환경 velm_qwen, 설치 없음).
# 필수: AGENT_ARTIFACTS(기준값 폴더) AGENT_DATA_ROOT(AeBAD 폴더)
# 선택: AGENT_HOST(127.0.0.1) AGENT_PORT(8200) AGENT_TOKEN(로컬 주소가 아니면 필수) AGENT_MMR_URL(http://127.0.0.1:8101)
#       AGENT_DATA_DIR(SQLite 저장 폴더) AGENT_REFS_MANIFEST AGENT_REF_PX AGENT_ENV(기본 velm_qwen)
# 외부(Tailscale 등) 접속: AGENT_HOST 를 Tailscale 주소로 지정하고 AGENT_TOKEN 을 설정한다. 포트 포워딩은 쓰지 않는다.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AGENT_ENV:-velm_qwen}"
cd "$(dirname "$0")/../agent_service"
exec python server.py
