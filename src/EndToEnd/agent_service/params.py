"""고정 파라미터(params.json): 구간 기준, 임계값 t, 라우터 계수. 데이터셋 CSV/npz 는 읽지 않는다."""
import json

import numpy as np


class Params:
    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.raw = d
        self.tau_lo, self.tau_hi, self.t = float(d["tau_lo"]), float(d["tau_hi"]), float(d["t"])
        r = d["router"]
        self.mean, self.scale = np.array(r["mean"]), np.array(r["scale"])
        self.coef, self.intercept = np.array(r["coef"]), float(r["intercept"])

    def rscore(self, amap):
        """이상 맵 모양 특징 13개 -> 라우터 점수 (기존 attach_router_scores 와 같은 계산)."""
        import region_crop as rc
        x = np.array(rc.map_features(amap), dtype=np.float64)
        return float(((x - self.mean) / self.scale) @ self.coef + self.intercept)

    def zone(self, rs):
        return "clear_normal" if rs < self.tau_lo else ("confident" if rs >= self.tau_hi else "amb")
