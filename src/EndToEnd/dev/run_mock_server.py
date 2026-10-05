"""[UI 개발 전용] GPU 없이 가짜 에이전트 서버를 띄운다 (맥에서 프론트엔드를 개발할 때).
결과는 무작위 값이다. 사용: python dev/run_mock_server.py [포트=8200]  -> 임시 폴더에 모의 기준값·갤러리를 만들어 사용"""
import os
import sys
import tempfile
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "agent_service"))
import mock_artifacts  # noqa: E402
import mock_backends   # noqa: E402
import server          # noqa: E402
from config import Settings  # noqa: E402


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8200
    tmp = tempfile.mkdtemp(prefix="e2e_mock_")
    art = os.environ.get("MOCK_ARTIFACTS") or os.path.join(tmp, "art")      # e2e.sh 회귀 테스트가 폴더를 지정한다
    mock_artifacts.make(art, n_gallery=8)
    s = Settings({"AGENT_ARTIFACTS": art, "AGENT_DATA_ROOT": art, "AGENT_DATA_DIR": os.environ.get("AGENT_DATA_DIR") or os.path.join(tmp, "db"),
                  "AGENT_PORT": str(port)})
    app = mock_backends.build_mock_app(s)
    print("[모의 서버 - 판정은 무작위] http://127.0.0.1:{}  (임시 폴더 {})".format(port, tmp), flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), server.make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
