"""에이전트 서비스 HTTP 서버 (표준 라이브러리만 사용 -> velm_qwen 환경에 아무것도 설치하지 않는다).

  GET  /api/health
  GET  /api/gallery                      갤러리 목록(id 만, 경로·정답 없음)
  GET  /api/gallery/<id>/thumb | image   갤러리 이미지
  GET  /api/dashboard[?include_test=1]   저장된 결과(최신순). 점검용 테스트 검사(X-E2E-Test: 1 헤더로 표시)는 기본 제외
  POST /api/inspect                      본문 = 이미지 바이트 -> SSE 스트림
  POST /api/inspect/gallery/<id>         갤러리 이미지로 같은 루프 -> SSE 스트림
SSE 이벤트: start, mmr, tool_start, thought, action, observation, final, done, error
한 번에 한 이미지만 처리한다(처리 중이면 409). 접근 토큰(AGENT_TOKEN)은 Authorization: Bearer 또는 ?token= 으로 받는다.
"""
import hashlib
import hmac
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import Settings   # noqa: E402  (VELM/offline/common 경로도 여기서 추가됨)
import separation as sep      # noqa: E402
from online_agent import OnlineAgent  # noqa: E402
from params import Params    # noqa: E402
from store import Store, downsample_map  # noqa: E402

GALLERY_ID = re.compile(r"^[0-9a-f]{12}$")
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}


class App:
    """서비스 상태: 설정, 파라미터, 갤러리, 에이전트, 저장소."""

    def __init__(self, settings, mmr_client, agent):
        self.s = settings
        self.mmr = mmr_client
        self.agent = agent
        self.params = agent.params
        self.store = Store(settings.data_dir)
        self.busy = threading.Lock()
        self.ready = True
        self._thumbs = {}
        sets, _ = sep.load(os.path.join(settings.artifacts, "separation.json"))
        sep.verify(os.path.join(settings.artifacts, "separation.json"))      # 겹침이 있으면 시작하지 않음
        self.sep_sets = sets
        with open(os.path.join(settings.artifacts, "gallery.json")) as f:
            self.gallery = json.load(f)["items"]
        self.gallery_by_id = {g["id"]: g for g in self.gallery}
        for g in self.gallery:        # 갤러리 파일이 오프라인 단계와 같은 파일인지(해시) 시작 때 확인
            if sep.sha256_file(os.path.join(settings.data_root, g["file"])) != g["sha256"]:
                raise SystemExit("갤러리 이미지가 변경되었습니다: {}".format(g["id"]))

    def gallery_path(self, gid):
        g = self.gallery_by_id.get(gid)
        if not g:
            return None
        return os.path.join(self.s.data_root, g["file"])

    def thumb(self, gid):
        if gid not in self._thumbs:
            img = Image.open(self.gallery_path(gid)).convert("RGB")
            img.thumbnail((320, 320))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            self._thumbs[gid] = buf.getvalue()
        return self._thumbs[gid]


def make_handler(app):
    s = app.s

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"      # SSE 는 연결을 닫아 끝낸다

        # ---------------------------------------------------------- 공통
        def _cors(self):
            if s.cors_origin:
                self.send_header("Access-Control-Allow-Origin", s.cors_origin)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, code, data, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            if not s.token:
                return True
            got = ""
            h = self.headers.get("Authorization", "")
            if h.startswith("Bearer "):
                got = h[7:]
            if not got:
                got = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
            return hmac.compare_digest(got.encode(), s.token.encode())

        def log_message(self, fmt, *args):
            sys.stderr.write("[agent] " + (fmt % args) + "\n")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        # ---------------------------------------------------------- GET
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":          # 인증 없이 상태만 (민감 정보 없음)
                return self._json(200, {"status": "ok", "ready": app.ready, "busy": app.busy.locked()})
            if not self._authorized():
                return self._json(401, {"error": "unauthorized"})
            if path == "/api/gallery":
                return self._json(200, {"items": [{"id": g["id"], "thumb": "/api/gallery/{}/thumb".format(g["id"])} for g in app.gallery]})
            m = re.match(r"^/api/gallery/([0-9a-f]{12})/(thumb|image)$", path)
            if m:
                gid, kind = m.groups()
                if gid not in app.gallery_by_id:
                    return self._json(404, {"error": "not found"})
                if kind == "thumb":
                    return self._bytes(200, app.thumb(gid), "image/jpeg")
                with open(app.gallery_path(gid), "rb") as f:
                    return self._bytes(200, f.read(), "application/octet-stream")
            if path == "/api/dashboard":
                inc = (parse_qs(urlparse(self.path).query).get("include_test") or ["0"])[0] == "1"
                return self._json(200, {"items": app.store.list(include_test=inc)})
            m = re.match(r"^/api/dashboard/(\d+)$", path)
            if m:
                rec = app.store.get(int(m.group(1)))
                return self._json(200, rec) if rec else self._json(404, {"error": "not found"})
            self._json(404, {"error": "not found"})

        # ---------------------------------------------------------- POST (추론)
        def do_POST(self):
            path = urlparse(self.path).path
            if not self._authorized():
                return self._json(401, {"error": "unauthorized"})
            if path == "/api/inspect":
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > s.max_upload:
                    return self._json(413 if n > 0 else 400, {"error": "이미지 크기는 1B ~ {}MB 여야 합니다".format(s.max_upload // 1024 // 1024)})
                return self._inspect(self.rfile.read(n), "upload", None)
            m = re.match(r"^/api/inspect/gallery/([0-9a-f]{12})$", path)
            if m:
                gid = m.group(1)
                if gid not in app.gallery_by_id:
                    return self._json(404, {"error": "not found"})
                with open(app.gallery_path(gid), "rb") as f:
                    return self._inspect(f.read(), "gallery", gid)
            self._json(404, {"error": "not found"})

        def _inspect(self, data, source, gid):
            try:
                img = Image.open(io.BytesIO(data))
                fmt = img.format
                img.verify()
            except Exception:
                return self._json(400, {"error": "이미지 파일이 아닙니다"})
            if fmt not in ALLOWED_FORMATS:
                return self._json(415, {"error": "지원하지 않는 형식입니다 ({})".format(fmt)})
            if not app.busy.acquire(blocking=False):
                return self._json(409, {"error": "다른 이미지를 처리 중입니다. 잠시 후 다시 시도하세요."})
            tmp = None
            is_test = self.headers.get("X-E2E-Test") == "1"      # 점검 스크립트(integration_test)의 검사는 대시보드에 보이지 않게 표시한다
            alive = [True]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()

            def emit(ev, obj):
                if not alive[0]:
                    return
                try:
                    self.wfile.write("event: {}\ndata: {}\n\n".format(ev, json.dumps(obj, ensure_ascii=False)).encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    alive[0] = False          # 화면이 닫혀도 판정은 끝까지 하고 저장한다

            t0 = time.time()
            try:
                sha = hashlib.sha256(data).hexdigest()
                flag = sep.classify_upload(sha, app.sep_sets) if source == "upload" else None
                emit("start", {"source": source, "size": list(Image.open(io.BytesIO(data)).size),
                               "warning": ("이 이미지는 {} 이미지와 같은 파일입니다. 데모에 쓰지 않는 것이 좋아요.".format(
                                   {"refs": "퓨샷 참고", "calib": "파라미터 계산", "gallery": "갤러리"}[flag])) if flag else None})
                tmp = os.path.join(tempfile.gettempdir(), "e2e_{}{}".format(uuid.uuid4().hex, ALLOWED_FORMATS[fmt]))
                with open(tmp, "wb") as f:
                    f.write(data)
                tm = time.time()
                mmr = app.mmr.infer(data)
                t_mmr = time.time() - tm
                b64, shp = downsample_map(mmr["map"])

                def agent_emit(ev, obj):
                    if ev == "mmr":
                        obj = dict(obj, map_b64=b64, map_shape=shp, sec=round(t_mmr, 2), timings=mmr.get("timings"))
                    emit(ev, obj)

                ta = time.time()
                res = app.agent.inspect(tmp, mmr, agent_emit)
                timings = {"mmr_s": round(t_mmr, 2), "agent_s": round(time.time() - ta, 2), "total_s": round(time.time() - t0, 2)}
                rec = {"source": source, "sha256": sha, "sep_flag": flag, "mmr_score": res["mmr_score"], "rscore": res["rscore"],
                       "zone": res["zone"], "decision": res["decision"], "defect_type": res["defect_type"], "why": res["why"],
                       "tools": res["tools"], "trace": res["trace"], "type_probs": res["type_probs"], "timings": timings,
                       "map_b64": b64, "map_shape": shp, "image_size": mmr["orig_size"], "is_test": is_test}
                rid = app.store.add(rec)
                emit("final", {"id": rid, "decision": res["decision"], "defect_type": res["defect_type"], "why": res["why"],
                               "zone": res["zone"], "tools": res["tools"], "type_probs": res["type_probs"], "timings": timings})
                emit("done", {"id": rid})
            except Exception as e:
                traceback.print_exc()
                emit("error", {"message": "{}: {}".format(type(e).__name__, str(e)[:300])})
            finally:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
                app.busy.release()

    return H


def build_app(settings):
    """운영용: MMR 서비스 + 실제 Qwen 에이전트. (가짜 구성요소는 dev/ 에만 있고 테스트가 App 에 직접 주입한다)"""
    settings.validate()
    params = Params(os.path.join(settings.artifacts, "params.json"))
    from mmr_client import MMRClient
    from qwen_backend import load_qwen
    mmr = MMRClient(settings.mmr_url)
    try:
        print("MMR 서비스:", mmr.health(), flush=True)
    except Exception as e:
        raise SystemExit("MMR 서비스에 연결할 수 없습니다 ({}): {}".format(settings.mmr_url, e))
    app = App(settings, mmr, load_qwen(settings, params))
    if app.gallery:                       # 시작할 때 한 번 돌려 둔다 (콜드 스타트 제거). 실패해도 서비스는 계속 시작한다
        try:
            t0 = time.time()
            app.agent.warmup(app.gallery_path(app.gallery[0]["id"]))
            print("예열 완료 (참고 이미지 캐시 준비, {:.1f}초)".format(time.time() - t0), flush=True)
        except Exception as e:
            print("[경고] 예열 실패(서비스는 계속 시작합니다): {}: {}".format(type(e).__name__, str(e)[:200]), flush=True)
    return app


def main():
    s = Settings()
    app = build_app(s)
    print("에이전트 서비스 시작 http://{}:{} (인증={})".format(s.host, s.port, "토큰" if s.token else "없음(로컬 전용)"), flush=True)
    ThreadingHTTPServer((s.host, s.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
