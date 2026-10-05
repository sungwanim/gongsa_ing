"""에이전트 루프 사전 시험 (GPU 불필요): 저장된 Qwen 확률로 '애매하면 한 번 더 본다'는 2단계 루프를 시뮬레이션한다.

루프 (사진마다)
  구간(라우터): 정상 확정 -> 종료(정상) / 확실한 불량 -> 종료(불량) / 애매 -> 아래 루프
  1회차  : Qwen(v8) 의 normal 확률 pn1
           pn1 >= ta  -> 확신(정상) 종료
           pn1 <  tb  -> 확신(불량) 종료
           그 사이    -> 애매: 도구 사용(두 번째 의견, 기본 v7 프롬프트) 후 평균 pn = (pn1+pn2)/2
  2회차  : pn >= tc -> 정상, 아니면 불량.   (끝까지 애매하면 불량 = 재현율 보호)
두 번째 의견이 저장돼 있지 않은 사진은 1회차 기준(ta)만으로 판정한다.
기준값 ta/tb/tc 는 보정용 절반에서 '시스템 재현율 >= 목표' 를 만족하는 조합 중 오탐이 가장 적은 것으로 정하고, 숫자는 보고용 절반에서만 본다.
기준선(비교용): 루프 없이 v8 한 번만 보고 t 하나로 판정 (지금 최종 방식).
사용: python agent_loop_sim.py --mmr-out ... [--second v7] [--router] [--recall-target 0.98] [--target 0.978]
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q  # noqa: E402


def predict(rows, p1, p2, ta, tb, tc):
    """각 사진의 최종 판정 (1=불량, 0=정상)과 2회차를 쓴 사진 수."""
    out, used = [], 0
    for r in rows:
        z = r["zone10"]
        if z == "clear_normal":
            out.append(0); continue
        if z == "confident":
            out.append(1); continue
        a = p1.get(q.key(r))
        if a is None:
            out.append(1); continue
        pn1 = a.get("normal", 0.0)
        b = p2.get(q.key(r))
        if b is None:                                  # 두 번째 의견 없음 -> 1회차 기준만
            out.append(0 if pn1 >= ta else 1); continue
        if pn1 >= ta:
            out.append(0)
        elif pn1 < tb:
            out.append(1)
        else:
            used += 1
            out.append(0 if (pn1 + b.get("normal", 0.0)) / 2.0 >= tc else 1)
    return np.array(out), used


def summarize(rows, pred):
    y = np.array([r["label"] for r in rows])
    return q.metrics(y, pred)


def fmt(m):
    return "Recall {:.3f} (FN {:>3}) | FP {:>3} FPR {:.3f} | Acc {:.3f} Prec {:.3f} F1 {:.3f}".format(
        m["Recall"], m["FN"], m["FP"], m["FPR"], m["Accuracy"], m["Precision"], m["F1"])


def predict_rescue(rows, p1, p2, t0, tr):
    """v8 이 정상으로 인정(pn1>=t0)한 애매 구간 사진 중, 두 번째 의견의 normal 확률이 tr 미만이면 불량으로 되돌린다. (불량 쪽으로만 바꾼다)"""
    out, rescued = [], 0
    for r in rows:
        z = r["zone10"]
        if z == "clear_normal":
            out.append(0); continue
        if z == "confident":
            out.append(1); continue
        a = p1.get(q.key(r))
        if a is None or a.get("normal", 0.0) < t0:
            out.append(1); continue
        b = p2.get(q.key(r))
        if b is not None and b.get("normal", 0.0) < tr:
            out.append(1); rescued += 1
        else:
            out.append(0)
    return np.array(out), rescued


def rescue_table(rows, calib, report, p1, p2, a):
    t0 = q.choose_t_by_recall(calib, p1, None, 0.97)
    print("\n[구제 루프] v8 기준 t={} 로 정상 인정 후, 두 번째 의견({})의 normal 확률이 tr 미만이면 불량으로 되돌림".format(t0, a.second))
    print("{:>6} | {:^52} | {:^52}".format("tr", "보정용 절반", "보고용 절반"))
    for tr in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
        cm = summarize(calib, predict_rescue(calib, p1, p2, t0, tr)[0])
        pr, n = predict_rescue(report, p1, p2, t0, tr)
        rm = summarize(report, pr)
        print("{:>6} | Recall {:.3f} FN {:>3} FP {:>3} FPR {:.3f} F1 {:.3f} | Recall {:.3f} FN {:>3} FP {:>3} FPR {:.3f} F1 {:.3f} (되돌림 {}장)".format(
            tr, cm["Recall"], cm["FN"], cm["FP"], cm["FPR"], cm["F1"], rm["Recall"], rm["FN"], rm["FP"], rm["FPR"], rm["F1"], n))
    print("읽는 법: tr=0.0 이 구제 없음(기준선)입니다. tr을 올릴수록 FN이 줄고 FP가 늘어납니다. FN이 크게 줄면서 FP가 조금만 느는 구간이 있으면 구제 루프가 효과가 있는 것입니다.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--first", default="v8_n12")
    p.add_argument("--second", default="v7", help="두 번째 의견으로 쓸 저장된 Qwen 결과 이름")
    p.add_argument("--router", action="store_true")
    p.add_argument("--recall-target", type=float, default=0.98, help="구간 나누기(MMR/라우터 쪽)의 재현율 목표")
    p.add_argument("--target", type=float, default=0.978, help="시스템 전체 재현율 목표 (보정용 절반에서 이 이상이어야 후보)")
    p.add_argument("--fpr-hi", type=float, default=0.02)
    p.add_argument("--rescue", action="store_true", help="구제 루프 표: v8이 정상으로 인정한 사진을 두 번째 의견이 확실히 불량이라고 하면 불량으로 되돌림")
    a = p.parse_args()
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    calib = [r for r in rows if q.split_of(r) == "calib"]
    report = [r for r in rows if q.split_of(r) == "report"]
    field = "score"
    if a.router:
        q.attach_router_scores(rows, a.mmr_out)
        field = "rscore"
    lo, hi = q.calibrate_v10(calib, a.recall_target, a.fpr_hi, field)
    q.mark_zones_v10(rows, lo, hi, field)
    q.TAG = a.first
    p1 = q.load_probs(a.out)
    q.TAG = a.second
    p2 = q.load_probs(a.out)
    amb = [r for r in rows if r["zone10"] == "amb"]
    cov = sum(1 for r in amb if q.key(r) in p2)
    print("구간 나누기: {} | 애매 구간 {}장 중 두 번째 의견({}) 있는 사진 {}장 ({:.0%})".format(
        "라우터" if a.router else "MMR 점수", len(amb), a.second, cov, cov / max(len(amb), 1)))
    cov_r = sum(1 for r in report if r["zone10"] == "amb" and q.key(r) in p2)
    print("  (보고용 절반의 애매 구간 {}장 중 {}장)".format(sum(1 for r in report if r["zone10"] == "amb"), cov_r))

    if a.rescue:
        rescue_table(rows, calib, report, p1, p2, a)
        return

    # 기준선: v8 한 번 + t 하나 (지금 최종 방식, normal 재현율 목표 0.97 과 같은 방식으로 t 결정)
    t0 = q.choose_t_by_recall(calib, p1, None, 0.97)
    base_c = summarize(calib, predict(calib, p1, {}, t0, 0.0, 1.0)[0])
    base_r = summarize(report, predict(report, p1, {}, t0, 0.0, 1.0)[0])
    print("\n[기준선] v8 한 번 + t={} (루프 없음)".format(t0))
    print("  보정용: " + fmt(base_c))
    print("  보고용: " + fmt(base_r))

    best = None
    for ta, tb, tc in itertools.product((0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999), (0.0, 0.1, 0.3, 0.5, 0.7), (0.80, 0.90, 0.95, 0.97, 0.98, 0.99)):
        if tb >= ta:
            continue
        pr, used = predict(calib, p1, p2, ta, tb, tc)
        m = summarize(calib, pr)
        if m["Recall"] < a.target:
            continue
        key_ = (m["FP"], -m["Recall"])
        if best is None or key_ < best[0]:
            best = (key_, (ta, tb, tc), m, used)
    if best is None:
        print("\n[루프] 보정용 절반에서 재현율 {:.3f} 이상을 만족하는 조합이 없습니다. (--target 을 낮추거나 두 번째 의견을 바꿔 보세요)".format(a.target))
        return
    (_, (ta, tb, tc), m_c, used_c) = best
    pr, used_r = predict(report, p1, p2, ta, tb, tc)
    m_r = summarize(report, pr)
    print("\n[루프] 보정용에서 고른 기준: ta={} (이 이상이면 바로 정상), tb={} (이 미만이면 바로 불량), tc={} (재관찰 후 평균 normal 확률 기준)".format(ta, tb, tc))
    print("  보정용: " + fmt(m_c) + " | 2회차 사용 {}장".format(used_c))
    print("  보고용: " + fmt(m_r) + " | 2회차 사용 {}장".format(used_r))
    print("\n[비교, 보고용] 오탐(FP) {} -> {} ({:+d}),  놓친 불량(FN) {} -> {} ({:+d})".format(
        base_r["FP"], m_r["FP"], m_r["FP"] - base_r["FP"], base_r["FN"], m_r["FN"], m_r["FN"] - base_r["FN"]))
    print("읽는 법: 오탐이 줄고 놓친 불량이 늘지 않으면 루프(애매하면 한 번 더 본다)가 효과가 있는 것입니다. 보고용 절반 한 번의 결과라 몇 장 차이는 우연일 수 있습니다.")


if __name__ == "__main__":
    main()
