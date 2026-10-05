"""대시보드 저장소(SQLite). 업로드 파일명은 저장하지 않는다(경로·파일명에 라벨이 들어 있을 수 있음)."""
import base64
import json
import os
import sqlite3
import threading
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  source TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  sep_flag TEXT,
  mmr_score REAL, rscore REAL, zone TEXT,
  decision TEXT, defect_type TEXT, why TEXT,
  tools TEXT, trace TEXT, type_probs TEXT, timings TEXT,
  map_b64 TEXT, map_shape TEXT, image_size TEXT
)"""


def downsample_map(amap, factor=2):
    """224x224 -> 112x112 (블록 평균), float16 base64. 대시보드 히트맵 표시용."""
    h, w = amap.shape
    m = amap[:h - h % factor, :w - w % factor].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return base64.b64encode(m.astype("<f2").tobytes()).decode(), list(m.shape)


class Store:
    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "dashboard.sqlite3")
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def add(self, rec):
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO inspections (created_at, source, sha256, sep_flag, mmr_score, rscore, zone, decision, defect_type, why,"
                " tools, trace, type_probs, timings, map_b64, map_shape, image_size) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), rec["source"], rec["sha256"], rec.get("sep_flag"), rec["mmr_score"], rec["rscore"],
                 rec["zone"], rec["decision"], rec.get("defect_type"), rec["why"], json.dumps(rec["tools"]),
                 json.dumps(rec["trace"], ensure_ascii=False), json.dumps(rec.get("type_probs")), json.dumps(rec["timings"]),
                 rec["map_b64"], json.dumps(rec["map_shape"]), json.dumps(rec.get("image_size"))))
            return cur.lastrowid

    @staticmethod
    def _row(r):
        d = dict(r)
        for k in ("tools", "trace", "type_probs", "timings", "map_shape", "image_size"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        d.pop("sha256", None)
        return d

    def list(self, limit=200):
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,))]

    def get(self, rid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM inspections WHERE id=?", (rid,)).fetchone()
            return self._row(r) if r else None
