"""재관찰 효과 사전 시험 (GPU 불필요): 저장된 서로 다른 Qwen 결과(예: v8 전체 사진 + v10/v11 크롭)를 합치면 결함 종류 정확도가 오르는가?

같은 사진에 대해 두 결과가 모두 있는 불량 사진만 비교한다.
  v8 단독 / 두 번째 결과 단독 / 두 확률의 평균 (4개 결함 종류 확률만, normal 제외)
사용: python combine_probe.py [--tags v8_n12 v10_n15 v11_n15]   (첫 번째가 기준)
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q  # noqa: E402


def vec(pr):
    v = np.array([max(pr.get(k, 0.0), 1e-9) for k in q.DEF4], dtype=float)
    return v / v.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--tags", nargs="+", default=None)
    a = p.parse_args()
    found = sorted(os.path.basename(f)[len("qwen_results_"):-len(".jsonl")] for f in glob.glob(os.path.join(a.out, "qwen_results_*.jsonl")))
    print("서버에 있는 Qwen 결과 파일:", ", ".join(found) if found else "없음")
    tags = a.tags or found
    if "v8_n12" not in tags:
        tags = ["v8_n12"] + [t for t in tags if t != "v8_n12"]
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    maps = {}
    for t in tags:
        q.TAG = t
        maps[t] = q.load_probs(a.out)
    base = maps["v8_n12"]
    for t in tags[1:]:
        other = maps[t]
        rs = [r for r in rows if r["label"] == 1 and q.key(r) in base and q.key(r) in other]
        print("\n=== v8_n12 + {} : 두 결과가 모두 있는 불량 {}장 ===".format(t, len(rs)))
        if len(rs) < 20:
            print("  장수가 너무 적어서 비교하지 않습니다 (20장 미만)")
            continue
        print("{:>10} | {:>5} | {:>9} {:>9} {:>9}".format("타입", "장수", "v8 단독", t + " 단독", "평균"))
        tot = [0, 0, 0]
        for typ in q.DEF4:
            sub = [r for r in rs if r["type"] == typ]
            if not sub:
                continue
            c = [0, 0, 0]
            for r in sub:
                v1, v2 = vec(base[q.key(r)]), vec(other[q.key(r)])
                for i, v in enumerate((v1, v2, (v1 + v2) / 2)):
                    c[i] += int(q.DEF4[int(np.argmax(v))] == typ)
            for i in range(3):
                tot[i] += c[i]
            print("{:>10} | {:>5} | {:>9} {:>9} {:>9}".format(typ, len(sub), *["{:.0%}".format(x / len(sub)) for x in c]))
        print("{:>10} | {:>5} | {:>9} {:>9} {:>9}".format("전체", len(rs), *["{:.0%}".format(x / len(rs)) for x in tot]))
    print("\n읽는 법: '평균' 열이 v8 단독보다 전체/groove에서 올라가면 재관찰(합치기)이 도움이 된다는 뜻입니다.")


if __name__ == "__main__":
    main()
