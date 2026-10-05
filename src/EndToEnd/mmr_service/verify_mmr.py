"""온라인 추론이 기존 배치 평가와 같은 값을 내는지 확인한다 (개발 검증용. 앱 실행에는 쓰지 않는다).

기존 평가가 저장한 image_scores_*.csv 의 점수와 비교한다. mmr 환경 / GPU 에서 실행:
  python verify_mmr.py --ckpt <MMR_*.pth> --csv <log_MMR_AeBAD_S_54>/image_scores_..._same.csv --data-root <AeBAD> -n 10
"""
import argparse
import csv
import json
import os
import random
import statistics

from mmr_infer import MMRInferencer


def locate(data_root, path):
    p = path.replace("\\", "/")
    return os.path.join(data_root, p[p.index("AeBAD/") + len("AeBAD/"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--json-out", default=None, help="요약을 JSON 으로 저장 (e2e.sh 가 읽음)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, newline="")))
    random.Random(0).shuffle(rows)
    inf = MMRInferencer(a.ckpt, resident=a.resident)
    worst = 0.0
    loads, infers, peaks = [], [], []
    for r in rows[:a.n]:
        out = inf.infer_bytes(open(locate(a.data_root, r["image_path"]), "rb").read())
        d = abs(out["score"] - float(r["score"]))
        worst = max(worst, d)
        t = out["timings"]
        loads.append(t["load_s"])
        infers.append(t["infer_s"])
        peaks.append(t["gpu_peak_mb"])
        print("{}  online {:.6f}  csv {:.6f}  diff {:.2e}  {}".format(
            os.path.basename(r["image_path"]), out["score"], float(r["score"]), d, out["timings"]))
    print("최대 차이 {:.2e} -> {}".format(worst, "통과" if worst < 1e-3 else "불일치(확인 필요)"))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"n": len(infers), "max_diff": worst, "passed": worst < 1e-3, "load_s_median": statistics.median(loads),
                       "infer_s_median": statistics.median(infers), "gpu_peak_mb_max": max(peaks)}, f)


if __name__ == "__main__":
    main()
