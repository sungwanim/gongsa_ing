"""MMR 추론 HTTP 서비스 (표준 라이브러리만 사용 -> mmr 환경에 아무것도 설치하지 않는다).

  GET  /health  -> {"status": "ok", "resident": bool, "loaded": bool}
  POST /infer   -> 본문 = 이미지 파일 바이트(어떤 Content-Type 이든 상관없음)
                   {"score", "shape": [224, 224], "map_b64"(float32 리틀엔디안), "orig_size": [w, h], "timings"}

환경변수: MMR_CKPT(필수, MMR_*.pth), MMR_RESIDENT(0/1, 기본 0=요청마다 로드·해제), MMR_DEVICE(기본 cuda:0),
          MMR_HOST(기본 127.0.0.1), MMR_PORT(기본 8101)
"""
import base64
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024 * 1024


def make_handler(inferencer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "resident": inferencer.resident, "loaded": inferencer.loaded})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/infer":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > MAX_BODY:
                    return self._send(400, {"error": "본문 크기가 올바르지 않습니다 (1B ~ 64MB)"})
                data = self.rfile.read(n)
                out = inferencer.infer_bytes(data)
                amap = out["map"]
                self._send(200, {"score": out["score"], "shape": list(amap.shape),
                                 "map_b64": base64.b64encode(amap.astype("<f4").tobytes()).decode(),
                                 "orig_size": out["orig_size"], "timings": out["timings"]})
            except Exception as e:      # 이미지가 아니거나 추론 실패
                traceback.print_exc()
                self._send(400 if isinstance(e, OSError) else 500, {"error": "{}: {}".format(type(e).__name__, e)})

        def log_message(self, fmt, *args):
            sys.stderr.write("[mmr] " + fmt % args + "\n")

    return Handler


def main():
    from mmr_infer import MMRInferencer
    ckpt = os.environ.get("MMR_CKPT")
    if not ckpt:
        raise SystemExit("MMR_CKPT(체크포인트 경로)를 지정하세요.")
    inf = MMRInferencer(ckpt, device=os.environ.get("MMR_DEVICE", "cuda:0"), resident=os.environ.get("MMR_RESIDENT", "0") == "1")
    host, port = os.environ.get("MMR_HOST", "127.0.0.1"), int(os.environ.get("MMR_PORT", "8101"))
    print("MMR 서비스 시작 http://{}:{} (resident={})".format(host, port, inf.resident), flush=True)
    ThreadingHTTPServer((host, port), make_handler(inf)).serve_forever()


if __name__ == "__main__":
    main()
