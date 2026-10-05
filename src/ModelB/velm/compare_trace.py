"""에이전트 판단 기록(trace)에 있는 사진들로 MMR 단독 / 기준선(라우터+Qwen) / 에이전트를 같은 사진에서 비교한다 (GPU 불필요).
사용: python compare_trace.py results/agent_trace_free20b.jsonl [--mmr-out ...]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("trace")
p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
p.add_argument("--holdout", default=q.HOLDOUT)
a = p.parse_args()
rows = {q.key(r): r for r in q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)}
tr = [json.loads(l) for l in open(a.trace) if l.strip()]
miss = [x["key"] for x in tr if x["key"] not in rows]
if miss:
    sys.exit("MMR 결과에서 찾지 못한 사진이 있습니다: {}".format(miss[:3]))
y = [x["label"] for x in tr]
mmr = [rows[x["key"]]["pred"] for x in tr]
base = [x["baseline_defect"] for x in tr]
agent = [x["pred_defect"] for x in tr]
nd = sum(y)
print("사진 {}장 (정상 {} / 불량 {}) - {}".format(len(tr), len(tr) - nd, nd, a.trace))
print("{:>22} | {:>7} {:>4} {:>4} | {:>7} {:>7} {:>7} {:>7}".format("방식", "재현율", "FN", "FP", "오탐률", "정확도", "정밀도", "F1"))
for name, pr in (("MMR 단독", mmr), ("MMR+라우터+Qwen", base), ("에이전트", agent)):
    m = q.metrics(y, pr)
    print("{:>22} | {:>7.3f} {:>4} {:>4} | {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f}".format(
        name, m["Recall"], m["FN"], m["FP"], m["FPR"], m["Accuracy"], m["Precision"], m["F1"]))
print("불량->정상으로 바뀐 사진 (기준선 대비): {}장".format(sum(1 for b, g in zip(base, agent) if g < b)))
print("읽는 법: 사진 수가 적어서(정상/불량이 각각 몇 장) 재현율 1장 차이가 크게 보입니다. 829장 결과와 같이 보세요.")
