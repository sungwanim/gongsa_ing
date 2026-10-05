#!/usr/bin/env bash
# scripts/e2e.sh 의 up / status / test / down 흐름을 가짜 서비스로 검증한다 (서버·GPU·conda 불필요).
# 임시 폴더에 EndToEnd 를 복사하고 run_mmr.sh / run_agent.sh 를 가짜로 바꿔 끼운 뒤 e2e.sh 를 실제로 실행한다.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"
export E2E_HOME="$T/home" DATA_ROOT="$T/home/artifacts" MMR_PORT=18101 AGENT_PORT=18200
unset AGENT_HOST AGENT_TOKEN
mkdir -p "$T/EndToEnd/scripts" "$E2E_HOME/artifacts"
cp -r "$SRC/dev" "$SRC/agent_service" "$SRC/common" "$SRC/offline" "$SRC/mmr_service" "$T/EndToEnd/"
cp "$SRC/scripts/e2e.sh" "$T/EndToEnd/scripts/"
cp -r "$SRC/../ModelB" "$T/ModelB"
cat > "$T/EndToEnd/scripts/run_mmr.sh" <<'STUB'
#!/usr/bin/env bash
exec python3 -c '
import http.server, os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b"{\"status\": \"ok\", \"resident\": false, \"loaded\": false}"
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(os.environ["MMR_PORT"])), H).serve_forever()'
STUB
cat > "$T/EndToEnd/scripts/run_agent.sh" <<'STUB'
#!/usr/bin/env bash
cd "$(dirname "$0")/../dev" && MOCK_ARTIFACTS="$AGENT_ARTIFACTS" exec python3 run_mock_server.py "$AGENT_PORT"
STUB
python3 -c "import sys; sys.path.insert(0, '$T/EndToEnd/dev'); import mock_artifacts; mock_artifacts.make('$E2E_HOME/artifacts', 8)" >/dev/null
E="bash $T/EndToEnd/scripts/e2e.sh"
ok=0; total=0
chk() { total=$((total + 1)); if eval "$2"; then ok=$((ok + 1)); echo "PASS $1"; else echo "FAIL $1"; echo "     --- results.tsv ---"; sed 's/^/     /' "$E2E_HOME/logs/results.tsv" 2>/dev/null | cut -c1-150; fi; }

$E init > /dev/null
$E up > "$T/up.out" 2>&1; rc=$?
chk "up 성공(종료 코드 0)" "[ $rc -eq 0 ]"
chk "up: MMR/에이전트 PASS 기록" "grep -q '\[PASS\] MMR 서비스' $T/up.out && grep -q '\[PASS\] 에이전트 서비스' $T/up.out"
chk "pid 파일이 실제 서비스 프로세스를 가리킴(포트 점유 프로세스와 일치)" "[ \"\$(cat $E2E_HOME/logs/agent.pid)\" = \"\$(lsof -nP -iTCP:$AGENT_PORT -sTCP:LISTEN -t 2>/dev/null | head -1)\" ]"
$E up > "$T/up2.out" 2>&1
chk "up 을 다시 실행하면 이미 실행 중으로 처리(중복 시작 없음)" "grep -q '이미 실행 중' $T/up2.out"
$E status > "$T/status.out" 2>&1
chk "status: 두 서비스 실행 중" "[ \$(grep -c '실행 중' $T/status.out) -ge 2 ]"
$E test > "$T/test.out" 2>&1; rc=$?
chk "test 성공(종료 코드 0)" "[ $rc -eq 0 ]"
chk "test: 통합·안전 테스트 PASS 기록, 판정 칸에 FAIL 없음" "awk -F'\\t' '\$1==\"test_summary\" && \$4==\"PASS\"' $E2E_HOME/logs/results.tsv | grep -q . && ! awk -F'\\t' '\$4==\"FAIL\"' $E2E_HOME/logs/results.tsv | grep -q ."
chk "report.md 생성" "grep -q '통합·안전 테스트' $E2E_HOME/logs/report.md"
chk "결과표에 검사별 요약(test_runs)이 기록됨" "awk -F'\\t' '\$1==\"test_runs\" && \$3 ~ /read_map/' $E2E_HOME/logs/results.tsv | grep -q ."
$E params > "$T/params.out" 2>&1
chk "params: 산출물이 있으면 기존 산출물 사용으로 PASS 기록(이전 실패 기록 덮어씀)" "awk -F'\\t' '\$1==\"params\" && \$4==\"PASS\"' $E2E_HOME/logs/results.tsv | grep -q ."
$E down > "$T/down.out" 2>&1
sleep 1
chk "down: 에이전트 포트가 닫힘(프로세스가 실제로 종료됨)" "! lsof -nP -iTCP:$AGENT_PORT -sTCP:LISTEN -t >/dev/null 2>&1"
chk "down: MMR 포트가 닫힘" "! lsof -nP -iTCP:$MMR_PORT -sTCP:LISTEN -t >/dev/null 2>&1"
chk "down: pid 파일 정리" "[ ! -f $E2E_HOME/logs/agent.pid ] && [ ! -f $E2E_HOME/logs/mmr.pid ]"
$E test > "$T/test2.out" 2>&1; rc=$?
chk "서비스가 꺼져 있으면 test 는 실패로 보고" "[ $rc -ne 0 ] && grep -q '준비되어 있지 않음' $T/test2.out"

# ---- expose: 가짜 tailscale 로 두 방식(인터페이스에 직접 / 사용자 영역 모드) 검증 ----
mkdir -p "$T/bin"
mk_ts() { printf '#!/usr/bin/env bash\nargs=()\nfor a in "$@"; do case "$a" in --socket=*) ;; *) args+=("$a") ;; esac; done\n[ "${args[0]:-}" = "ip" ] && echo %s && exit 0\nexit 1\n' "$1" > "$T/bin/tailscale"; chmod +x "$T/bin/tailscale"; }
export PATH_ORIG="$PATH"
$E expose > "$T/expose0.out" 2>&1; rc=$?
chk "tailscale 가 없으면 expose 는 실패하고 대안(ts-setup/tunnel)을 안내" "[ $rc -ne 0 ] && grep -q 'ts-setup' $T/expose0.out && grep -q 'tunnel' $T/expose0.out"
$E tunnel > "$T/tunnel.out" 2>&1
chk "tunnel: ssh -L 안내에 에이전트 포트가 들어감" "grep -q 'ssh -N -L $AGENT_PORT:127.0.0.1:$AGENT_PORT' $T/tunnel.out"

mk_ts 100.100.100.1          # 이 서버의 인터페이스에 없는 주소 = 사용자 영역 모드
PATH="$T/bin:$PATH" $E expose > "$T/expose1.out" 2>&1; rc=$?
chk "사용자 영역 모드 expose 성공" "[ $rc -eq 0 ]"
chk "사용자 영역 모드: 127.0.0.1 에 열고 토큰을 강제(AGENT_HOST 없음, REQUIRE_TOKEN=1)" "grep -q 'AGENT_REQUIRE_TOKEN=1' $E2E_HOME/expose.env && ! grep -q 'AGENT_HOST' $E2E_HOME/expose.env"
chk "expose.env 권한 600" "[ \"\$(stat -f %Lp $E2E_HOME/expose.env 2>/dev/null || stat -c %a $E2E_HOME/expose.env)\" = 600 ]"
chk "expose 후 통합 테스트가 토큰으로 통과하고 토큰 없이는 401" "grep -q '토큰 없이 접근하면 401' $T/expose1.out && ! grep -q '\[FAIL\]' $T/expose1.out"
chk "맥 접속 안내에 tailnet 주소가 들어감" "grep -q 'VITE_API_TARGET=http://100.100.100.1:$AGENT_PORT' $T/expose1.out"
$E unexpose > "$T/unexpose.out" 2>&1
chk "unexpose: 설정 파일 삭제" "[ ! -f $E2E_HOME/expose.env ]"
$E down > /dev/null 2>&1

mk_ts 127.0.0.1              # 이 서버의 인터페이스에 있는 주소 = 정식 설치(커널 모드) 흉내
PATH="$T/bin:$PATH" $E expose > "$T/expose2.out" 2>&1; rc=$?
chk "정식 설치 모드 expose 성공, AGENT_HOST 가 tailscale 주소로 설정" "[ $rc -eq 0 ] && grep -q 'AGENT_HOST=127.0.0.1' $E2E_HOME/expose.env && ! grep -q REQUIRE $E2E_HOME/expose.env"
$E unexpose > /dev/null 2>&1; $E down > /dev/null 2>&1

echo; echo "e2e.sh 회귀 테스트: $ok / $total 통과"
pkill -f "$T/EndToEnd" 2>/dev/null; rm -rf "$T"
[ "$ok" = "$total" ]
