"""12번 시험: MMR 이상 맵의 '모양' 특징을 더하면 정상/불량을 더 잘 가르는가? (GPU 불필요, 기존 코드는 건드리지 않음)

비교
  기준     : MMR 점수(이상 맵의 최댓값)만 사용 - 지금 방식
  라우터   : 점수 + 이상 맵 모양 특징(크기, 덩어리 수, 가늘기, 위치 ...)을 로지스틱 회귀로 학습 (보정용 절반으로 학습)
평가       : 보고용 절반에서 (1) AUROC (2) 같은 재현율에서 '정상 확정'으로 거를 수 있는 정상/전체 이미지 비율
사용: python router_probe.py [--mmr-out ...] [--recall 0.98]
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q          # noqa: E402
import region_crop as rc       # noqa: E402

features = rc.map_features
NAMES = rc.MAP_FEATURE_NAMES


def cleared_at_recall(score, y, recall):
    """불량의 recall 비율을 포함하는 가장 큰 임계값 -> (임계값 미만인 이미지 비율, 그중 정상 비율)."""
    d = np.sort(score[y == 1])
    k = int(np.floor((1.0 - recall) * len(d)))
    return float(d[min(k, len(d) - 1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--recall", type=float, default=0.98)
    a = p.parse_args()
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    mi = rc.MapIndex(a.mmr_out)
    X, y, sp, base = [], [], [], []
    for r in rows:
        try:
            m = mi.get(q.resolve(r["path"]))
        except KeyError:
            continue
        X.append(features(m)); y.append(r["label"]); sp.append(q.split_of(r)); base.append(r["score"])
    X, y, sp, base = np.array(X), np.array(y), np.array(sp), np.array(base)
    cal, rep = sp == "calib", sp == "report"
    print("이미지 {}장 (보정용 {} / 보고용 {})".format(len(y), int(cal.sum()), int(rep.sum())))

    sc = StandardScaler().fit(X[cal])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[cal]), y[cal])
    z = clf.decision_function(sc.transform(X))

    print("\n[1] 정상/불량 구분 AUROC (보고용 절반)")
    print("    MMR 점수만      : {:.3f}".format(roc_auc_score(y[rep], base[rep])))
    print("    점수 + 맵 모양  : {:.3f}".format(roc_auc_score(y[rep], z[rep])))

    print("\n[2] 같은 재현율 목표({:.0%})에서 '정상 확정'(Qwen에 안 보냄)으로 거르는 이미지 (보고용 절반)".format(a.recall))
    print("{:>14} | {:>12} | {:>16} | {:>14} | {:>14}".format("방식", "Qwen에 가는 비율", "정상 확정된 정상", "놓친 불량", "정상 확정 중 불량"))
    for name, s in (("MMR 점수만", base), ("점수 + 맵 모양", z)):
        thr = cleared_at_recall(s[cal], y[cal], a.recall)          # 임계값은 보정용 절반으로
        clear = s[rep] < thr
        yr = y[rep]
        sent = 1.0 - clear.mean()
        print("{:>14} | {:>12.1%} | {:>10}/{:<5} | {:>7}/{:<6} | {:>8}".format(
            name, sent, int((clear & (yr == 0)).sum()), int((yr == 0).sum()), int((clear & (yr == 1)).sum()),
            int((yr == 1).sum()), int((clear & (yr == 1)).sum())))

    print("\n[3] 로지스틱 회귀 계수 (절댓값이 클수록 구분에 중요, +는 불량 쪽)")
    for nm, c in sorted(zip(NAMES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print("    {:>20}: {:+.2f}".format(nm, c))
    print("\n읽는 법: [2]에서 '정상 확정된 정상'이 늘어나고 '놓친 불량'이 늘지 않으면 라우터가 도움이 됩니다.")


if __name__ == "__main__":
    main()
