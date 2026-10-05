"""에이전트 판단 기록(agent_trace.jsonl) 분석 (GPU 불필요, 기록 파일만 읽음).
사용: python agent_report.py [results/agent_trace.jsonl]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "agent_trace.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()]
n = len(rows)
nd = sum(1 for x in rows if x["label"] == 1)
nn = n - nd
print("파일: {} | 사진 {}장 (정상 {} / 불량 {})".format(path, n, nn, nd))


def metrics(key):
    tp = sum(1 for x in rows if x["label"] == 1 and x[key] == 1)
    fn = nd - tp
    fp = sum(1 for x in rows if x["label"] == 0 and x[key] == 1)
    tn = nn - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / nd if nd else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"FN": fn, "FP": fp, "Recall": rec, "FPR": fp / nn if nn else 0.0, "Acc": (tp + tn) / n, "Prec": prec, "F1": f1}


print("\n[1] 성능 (이 {}장 기준)".format(n))
print("{:>8} | {:>3} {:>3} | {:>6} {:>6} {:>6} {:>6} {:>6}".format("", "FN", "FP", "Recall", "FPR", "Acc", "Prec", "F1"))
for name, key in (("기준선", "baseline_defect"), ("에이전트", "pred_defect")):
    m = metrics(key)
    print("{:>8} | {:>3} {:>3} | {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}".format(name, m["FN"], m["FP"], m["Recall"], m["FPR"], m["Acc"], m["Prec"], m["F1"]))

flips = [x for x in rows if x["pred_defect"] != x["baseline_defect"]]
print("\n[2] 판정이 바뀐 사진 {}장 (기준선 -> 에이전트)".format(len(flips)))
for x in flips:
    print("  {} | 구간 {} | 정답 {} | 기준선 {} -> 에이전트 {} | 이유: {} | 도구: {}".format(
        x["key"].split("/")[-1], x["zone"], x["true"], "불량" if x["baseline_defect"] else "정상",
        "불량" if x["pred_defect"] else "정상", x["why"], " > ".join(s["action"] for s in x["trace"])))
print("  불량->정상으로 바뀐 사진: {}장 (0이어야 정상)".format(sum(1 for x in flips if x["pred_defect"] == 0)))

print("\n[3] 도구 순서 분포")
seq = collections.Counter(" > ".join(s["action"] for s in x["trace"]) for x in rows)
for s, c in seq.most_common():
    print("  {:>3}장 ({:>3.0%}) {}".format(c, c / n, s))

print("\n[4] 구간별 도구 사용 평균 개수 (decide 제외)")
by = collections.defaultdict(list)
for x in rows:
    by[x["zone"]].append(sum(1 for s in x["trace"] if s["action"] != "decide"))
for z, v in by.items():
    print("  {:>13}: {:>3}장, 평균 {:.2f}개".format(z, len(v), sum(v) / len(v)))

use = collections.Counter(s["action"] for x in rows for s in x["trace"])
print("\n[5] 도구별 사용 횟수: " + ", ".join("{} {}".format(a, c) for a, c in use.most_common()))
skip = [x for x in rows if x["zone"] != "confident" and not any(s["action"] == "ask_whole" for s in x["trace"])]
print("    ask_whole 없이 끝난 사진(확실한 불량 제외): {}장".format(len(skip)))
for x in skip:
    print("      {} | 구간 {} | 정답 {} | 결과 {}".format(x["key"].split("/")[-1], x["zone"], x["true"], "불량" if x["pred_defect"] else "정상"))

print("\n[6] 생각 문장 샘플 (처음 8장, 각 사진의 첫 두 단계)")
for x in rows[:8]:
    print("  {} ({})".format(x["key"].split("/")[-1], x["zone"]))
    for s in x["trace"][:2]:
        print("     [{}] {}".format(s["action"], s["thought"][:140]))
print("\n[7] 생각 문장이 같은 사진 비율: 첫 문장 종류 {}가지 / {}장".format(len(set(x["trace"][0]["thought"] for x in rows)), n))
