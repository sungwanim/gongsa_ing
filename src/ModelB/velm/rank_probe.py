"""정답 결함 종류가 Qwen 확률에서 몇 순위인지 본다 (GPU 불필요, 저장된 v8 확률 파일만 읽음).

보는 것
  [1] 불량 사진의 정답 종류가 4개(결함 종류만) 중 몇 순위인가 (타입별)
  [2] 1순위가 틀린 사진에서 정답이 2순위인 비율 / (1순위, 2순위) 조합 상위
사용: python rank_probe.py [--mmr-out ...]
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--prompt", default="v8_n12")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    q.TAG = a.prompt
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    pm = q.load_probs(a.out)
    defect = [r for r in rows if r["label"] == 1 and q.key(r) in pm]
    print("Qwen 확률이 있는 불량 사진 {}장".format(len(defect)))
    print("\n[1] 정답 종류의 순위 (결함 4종 중, normal 제외)")
    print("{:>10} | {:>5} | {:>6} {:>6} {:>6} {:>6}".format("타입", "장수", "1순위", "2순위", "3순위", "4순위"))
    pairs = {}
    for t in q.DEF4:
        rs = [r for r in defect if r["type"] == t]
        if not rs:
            continue
        cnt = Counter()
        for r in rs:
            pr = pm[q.key(r)]
            order = sorted(q.DEF4, key=lambda k: -pr.get(k, 0.0))
            cnt[order.index(t) + 1] += 1
            if order[0] != t:
                pairs.setdefault(t, Counter())[(order[0], order[1])] += 1
        n = len(rs)
        print("{:>10} | {:>5} | ".format(t, n) + " ".join("{:>6}".format("{:.0%}".format(cnt[k] / n)) for k in (1, 2, 3, 4)))
    print("\n[2] 1순위가 틀린 사진의 (1순위 -> 2순위) 조합 상위 3개")
    for t, c in pairs.items():
        tot = sum(c.values())
        print("  정답 {} (1순위 틀림 {}장)".format(t, tot))
        for (a1, a2), v in c.most_common(3):
            print("      1순위 {:>9}, 2순위 {:>9} : {}장".format(a1, a2, v))
    print("\n읽는 법: 정답이 2순위인 비율이 높으면 '두 개 중 하나' 재질문이 통하고, 낮으면 통하지 않습니다.")


if __name__ == "__main__":
    main()
