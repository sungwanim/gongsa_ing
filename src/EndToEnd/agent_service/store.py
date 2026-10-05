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
  map_b64 TEXT, map_shape TEXT, image_size TEXT,
  is_test INTEGER NOT NULL DEFAULT 0,
  image_jpg BLOB
)"""

# 목록에는 이미지 본문(BLOB)을 싣지 않는다 (대시보드를 불러올 때마다 수 MB 를 읽지 않도록)
COLS = ("id, created_at, source, sha256, sep_flag, mmr_score, rscore, zone, decision, defect_type, why, tools, trace, type_probs, "
        "timings, map_b64, map_shape, image_size, is_test, (image_jpg IS NOT NULL) AS has_image")


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
            # 이전 버전으로 만든 DB 에는 is_test 열이 없다 -> 기록은 그대로 두고(is_test=0) 열만 추가한다
            cols = [r["name"] for r in c.execute("PRAGMA table_info(inspections)")]
            if "is_test" not in cols:
                c.execute("ALTER TABLE inspections ADD COLUMN is_test INTEGER NOT NULL DEFAULT 0")
            if "image_jpg" not in cols:
                c.execute("ALTER TABLE inspections ADD COLUMN image_jpg BLOB")       # 이전 기록은 이미지 없음(NULL) -> 화면은 히트맵만 표시

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def add(self, rec):
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO inspections (created_at, source, sha256, sep_flag, mmr_score, rscore, zone, decision, defect_type, why,"
                " tools, trace, type_probs, timings, map_b64, map_shape, image_size, is_test, image_jpg) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), rec["source"], rec["sha256"], rec.get("sep_flag"), rec["mmr_score"], rec["rscore"],
                 rec["zone"], rec["decision"], rec.get("defect_type"), rec["why"], json.dumps(rec["tools"]),
                 json.dumps(rec["trace"], ensure_ascii=False), json.dumps(rec.get("type_probs")), json.dumps(rec["timings"]),
                 rec["map_b64"], json.dumps(rec["map_shape"]), json.dumps(rec.get("image_size")), 1 if rec.get("is_test") else 0, rec.get("image_jpg")))
            return cur.lastrowid

    @staticmethod
    def _row(r):
        d = dict(r)
        d["has_image"] = bool(d.get("has_image"))
        for k in ("tools", "trace", "type_probs", "timings", "map_shape", "image_size"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        d.pop("sha256", None)
        return d

    def list(self, limit=200, include_test=False):
        """대시보드 목록(최신순). 점검용 테스트 검사(is_test=1)는 기본으로 제외한다."""
        q = "SELECT {} FROM inspections {} ORDER BY id DESC LIMIT ?".format(COLS, "" if include_test else "WHERE is_test = 0")
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q, (limit,))]

    def counts(self):
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*) AS n, COALESCE(SUM(is_test), 0) AS t FROM inspections").fetchone()
            return {"total": r["n"], "test": r["t"], "real": r["n"] - r["t"]}

    def backup(self):
        """DB 파일을 같은 폴더에 날짜가 붙은 복사본으로 저장하고 경로를 돌려준다 (삭제 전에 항상 호출)."""
        import shutil
        dst = os.path.join(os.path.dirname(self.path), "dashboard_backup_{}.sqlite3".format(time.strftime("%Y%m%d_%H%M%S")))
        with self._lock:
            src = sqlite3.connect(self.path)
            try:
                out = sqlite3.connect(dst)
                src.backup(out)          # 실행 중이어도 일관된 복사
                out.close()
            finally:
                src.close()
        return dst

    def purge(self, test_only=True):
        """기록 삭제. test_only=True 면 점검용 테스트 검사만, False 면 전부. 삭제한 건수를 돌려준다."""
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM inspections WHERE is_test = 1" if test_only else "DELETE FROM inspections")
            return cur.rowcount

    def get(self, rid):
        with self._conn() as c:
            r = c.execute("SELECT {} FROM inspections WHERE id=?".format(COLS), (rid,)).fetchone()
            return self._row(r) if r else None

    def get_image(self, rid):
        """저장된 미리보기 JPEG 바이트 (없으면 None)."""
        with self._conn() as c:
            r = c.execute("SELECT image_jpg FROM inspections WHERE id=?", (rid,)).fetchone()
            return bytes(r["image_jpg"]) if r and r["image_jpg"] is not None else None
