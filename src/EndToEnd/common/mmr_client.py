"""agent 서비스가 MMR 서비스를 부르는 클라이언트 (표준 라이브러리 + numpy)."""
import base64
import json
import urllib.error
import urllib.request

import numpy as np


class MMRClient:
    def __init__(self, base_url="http://127.0.0.1:8101", timeout=120):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=10) as r:
            return json.loads(r.read())

    def infer(self, image_bytes):
        """이미지 파일 바이트 -> {"score": float, "map": np.float32 (224,224), "orig_size": [w,h], "timings": {...}}"""
        req = urllib.request.Request(self.base_url + "/infer", data=image_bytes, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError("MMR 서비스 오류 {}: {}".format(e.code, e.read().decode(errors="replace")[:300]))
        amap = np.frombuffer(base64.b64decode(d["map_b64"]), dtype="<f4").reshape(d["shape"]).copy()
        return {"score": d["score"], "map": amap, "orig_size": d["orig_size"], "timings": d["timings"]}
