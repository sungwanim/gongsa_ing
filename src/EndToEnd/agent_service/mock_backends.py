"""GPU/모델 없이 흐름을 확인하는 모의 구성요소 (AGENT_MOCK=1). 실제 판정 능력은 없다."""
import base64
import hashlib

import numpy as np
from PIL import Image
import io

CLASSES = ["normal", "ablation", "breakdown", "fracture", "groove"]


def _seed(data):
    return int(hashlib.sha256(data).hexdigest()[:8], 16)


class FakeMMR:
    """이미지 바이트로 정해지는 가짜 이상 맵(224x224) + 점수. MMRClient.infer 와 같은 형태."""

    def infer(self, image_bytes):
        rnd = np.random.RandomState(_seed(image_bytes))
        size = Image.open(io.BytesIO(image_bytes)).size
        yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) / 223.0
        m = 0.2 + 0.05 * rnd.rand(224, 224).astype(np.float32)
        top = rnd.uniform(0.28, 0.72)
        for k in range(rnd.randint(1, 4)):
            cx, cy, r = rnd.uniform(0.2, 0.8), rnd.uniform(0.2, 0.8), rnd.uniform(0.04, 0.12)
            m += (top - 0.2) * (0.35 if k else 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        m = m.astype(np.float32)
        return {"score": float(m.max()), "map": m, "orig_size": list(size), "timings": {"mock": True}}


class FakeRefClassifier:
    """CachedRefClassifier 대역: classify(path) -> {"probs": 5-class 확률}"""

    def classify(self, path):
        data = open(path, "rb").read()
        rnd = np.random.RandomState(_seed(data) + 1)
        p = rnd.dirichlet(np.ones(5) * 0.6)
        if rnd.rand() < 0.5:
            p[0] += 1.2
        p = p / p.sum()
        probs = {k: float(v) for k, v in zip(CLASSES, p)}
        top = max(probs, key=probs.get)
        return {"label": top, "confidence": probs[top], "probs": probs, "class_mass": 1.0, "raw": top}


def fake_v7(path):
    return FakeRefClassifier().classify(path)["probs"]
