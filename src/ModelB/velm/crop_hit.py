"""크롭 전략별 '결함을 크롭 안에 담는 비율' 비교 (GPU 불필요).

MMR 이상 맵에서 점수가 높은 곳을 1곳 / 2곳 / 3곳 크롭했을 때, 정답 마스크(결함 위치)가
크롭 안에 들어오는 비율을 타입별로 보여 준다.
  중심 포함 : 결함 중심이 크롭 안에 있음
  일부 겹침 : 결함 네모와 크롭이 조금이라도 겹침
사용: python crop_hit.py [--mmr-out ...] [--n 20]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q          # noqa: E402
import region_crop as rc       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--n", type=int, default=20, help="타입별 이미지 수")
    p.add_argument("--crop-min", type=int, default=600)
    p.add_argument("--crop-max", type=int, default=900)
    a = p.parse_args()
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    mi = rc.MapIndex(a.mmr_out)
    rk = dict(min_side=a.crop_min, scale=1.5, max_side=a.crop_max)
    res = {}
    for r in q.quick_subset(rows, a.n, cond=lambda r: r["label"] == 1):
        path = q.resolve(r["path"])
        w, h = Image.open(path).size
        m = np.array(Image.open(path.replace("/test/", "/ground_truth/")).convert("L")) > 0
        ys, xs = np.where(m)
        mh, mw = m.shape
        sx, sy = w / float(mw), h / float(mh)
        mx0, my0, mx1, my1 = xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy
        cx, cy = xs.mean() * sx, ys.mean() * sy
        regs = [g for g, _ in rc.topk_regions(mi.get(path), w, h, 3, 0.0, rk)]   # rel=0: 곳 수 효과만 보기 위해 약한 곳도 포함
        for k in (1, 2, 3):
            sub = regs[:k]
            center = any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in sub)
            overlap = any(min(g[2], mx1) > max(g[0], mx0) and min(g[3], my1) > max(g[1], my0) for g in sub)
            d = res.setdefault(r["type"], {}).setdefault(k, [0, 0, 0])
            d[0] += int(center); d[1] += int(overlap); d[2] += 1
    print("크롭 영역이 결함을 담는 비율 (타입당 {}장, 크롭 한 변 {}~{}px)".format(a.n, a.crop_min, a.crop_max))
    print("{:>10} | {:^22} | {:^22} | {:^22}".format("타입", "1곳 (현재)", "2곳", "3곳"))
    print("{:>10} | ".format("") + " | ".join("{:>9} {:>11}".format("중심 포함", "일부 겹침") for _ in range(3)))
    for t in ["ablation", "breakdown", "fracture", "groove"]:
        if t not in res:
            continue
        cells = []
        for k in (1, 2, 3):
            c, o, n = res[t][k]
            cells.append("{:>9} {:>11}".format("{:.0%}".format(c / n), "{:.0%}".format(o / n)))
        print("{:>10} | ".format(t) + " | ".join(cells))
    print("\n읽는 법: 곳 수를 늘릴수록 비율이 올라가면 여러 곳을 크롭하는 방식이 도움이 된다는 뜻입니다 (대신 Qwen에 넘기는 이미지가 늘어남).")


if __name__ == "__main__":
    main()
