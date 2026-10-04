"""MMR(1차) 결과를 받아 Qwen(2차)로 normal/불량 4타입 중 하나를 고르게 하고, MMR vs MMR+Qwen 지표를 비교한다.

입력 : MMR이 저장한 image_scores_*.csv (image_path,label,score,prediction)  -> MMR 코드는 수정하지 않음
2차  : MMR이 불량(prediction=1)으로 본 이미지만 run_qwen.py 로 normal + 4개 타입 중 하나 선택
지표 : Accuracy / Precision / Recall / FPR / F1 / 혼동행렬 / Image AUROC (MMR만)
       전체(양품 vs 불량) 1회 + 불량 타입별(해당 타입 + 모든 양품) 각각
       * Pixel AUROC / PRO 는 Qwen이 이상 맵을 바꾸지 않으므로 MMR 로그 값을 그대로 사용
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
MMR_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "ModelA", "MMR_Test"))
TYPES = ["ablation", "breakdown", "fracture", "groove"]


def load_mmr_csv(mmr_out):
    rows = []
    files = sorted(glob.glob(os.path.join(mmr_out, "image_scores_*.csv")))
    if not files:
        raise SystemExit("MMR 결과 csv가 없습니다: {}/image_scores_*.csv".format(mmr_out))
    for f in files:
        domain = os.path.basename(f)[:-4].split("_")[-1]
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                p = r["image_path"].replace("\\", "/").split("/")
                anomaly = p[p.index("test") + 1] if "test" in p else "unknown"
                rows.append({"domain": domain, "path": r["image_path"], "label": int(float(r["label"])),
                             "score": float(r["score"]), "pred": int(float(r["prediction"])),
                             "type": "good" if int(float(r["label"])) == 0 else anomaly})
    return rows


def resolve(path):
    if os.path.isfile(path):
        return path
    alt = os.path.normpath(os.path.join(MMR_DIR, path))   # MMR은 MMR_Test 폴더 기준 상대경로로 저장함
    return alt if os.path.isfile(alt) else path


def key(r):
    return r["domain"] + "|" + r["path"]


def run_qwen(rows, out_dir, limit=None):
    from run_qwen import load_model, classify_image   # velm 코드 그대로 사용
    fpath = os.path.join(out_dir, "qwen_results_v6.jsonl")
    done = set()
    if os.path.exists(fpath):
        with open(fpath) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [r for r in rows if r["pred"] == 1 and key(r) not in done]
    if limit:
        todo = todo[:limit]
    print("Qwen 대상 {}장 (이미 완료 {}장)".format(len(todo), len(done)), flush=True)
    if not todo:
        return
    model, processor = load_model()
    with open(fpath, "a") as f:
        for n, r in enumerate(todo, 1):
            try:
                out = classify_image(resolve(r["path"]), model, processor)
            except Exception as e:
                out = {"label": "unknown", "confidence": 0.0, "probs": {}, "class_mass": 0.0, "raw": "ERROR: {}".format(e)}
            f.write(json.dumps({"key": key(r), "raw": out["raw"], "pred_type": out["label"],
                                "confidence": out["confidence"], "probs": out["probs"],
                                "class_mass": out["class_mass"]}, ensure_ascii=False) + "\n")
            f.flush()
            print("[{}/{}] {} -> {} ({:.1%})".format(n, len(todo), os.path.basename(r["path"]), out["label"], out["confidence"]), flush=True)


def metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    d = lambda a, b: a / b if b else 0.0
    p, r = d(tp, tp + fp), d(tp, tp + fn)
    auc = float("nan")
    if score is not None and len(set(y.tolist())) == 2:
        auc = float(roc_auc_score(y, score))
    return {"Accuracy": d(tp + tn, len(y)), "Precision": p, "Recall": r, "FPR": d(fp, fp + tn),
            "F1": d(2 * p * r, p + r), "TP": tp, "FN": fn, "FP": fp, "TN": tn, "ImageAUROC": auc}


def evaluate(rows, out_dir):
    qwen, conf = {}, {}
    fpath = os.path.join(out_dir, "qwen_results_v6.jsonl")
    if os.path.exists(fpath):
        with open(fpath) as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    qwen[d["key"]] = "good" if d["pred_type"] == "normal" else d["pred_type"]
                    conf[d["key"]] = d.get("confidence")
    final = []                       # MMR+Qwen 최종 라벨: good 또는 타입
    for r in rows:
        if r["pred"] == 0:
            final.append("good")
        else:
            if key(r) not in qwen:
                raise SystemExit("Qwen 결과 누락: {} -> 먼저 Qwen 단계를 실행하세요".format(key(r)))
            final.append(qwen[key(r)])
    final = np.array(final)
    y = np.array([r["label"] for r in rows]); pred1 = np.array([r["pred"] for r in rows])
    score = np.array([r["score"] for r in rows]); typ = np.array([r["type"] for r in rows])
    pred2 = (final != "good").astype(int)       # MMR이 불량으로 봤어도 Qwen이 normal이면 정상으로 뒤집음

    scopes = [("전체 (양품 vs 불량)", np.arange(len(rows)), None)]
    for t in sorted(set(typ.tolist()) - {"good"}):
        scopes.append(("타입: " + t, np.where((typ == "good") | (typ == t))[0], t))

    out = []
    for name, idx, t in scopes:
        m1 = metrics(y[idx], pred1[idx], score[idx])
        p2 = pred2[idx] if t is None else (final[idx] == t).astype(int)   # 타입별: Qwen이 그 타입이라고 맞혀야 양성
        m2 = metrics(y[idx], p2)
        m2["ImageAUROC"] = float("nan")
        for model, m in (("MMR", m1), ("MMR+Qwen", m2)):
            out.append(dict(scope=name, model=model, n_good=int((y[idx] == 0).sum()),
                            n_defect=int((y[idx] == 1).sum()), **m))

    cols = ["scope", "model", "n_good", "n_defect", "Accuracy", "Precision", "Recall", "FPR", "F1",
            "TP", "FN", "FP", "TN", "ImageAUROC"]
    f4 = lambda v: "-" if isinstance(v, float) and np.isnan(v) else ("{:.4f}".format(v) if isinstance(v, float) else str(v))
    w = [max(len(c), max(len(f4(r[c])) for r in out)) for c in cols]
    lines = [" | ".join(c.ljust(w[i]) for i, c in enumerate(cols)), "-+-".join("-" * x for x in w)]
    lines += [" | ".join(f4(r[c]).ljust(w[i]) for i, c in enumerate(cols)) for r in out]

    labels = ["good"] + TYPES + ["unknown"]
    cm = ["", "[MMR+Qwen 5-class 혼동행렬] 행=정답, 열=최종예측", "  ".join(["{:>10}".format("")] + ["{:>10}".format(c) for c in labels])]
    for t in ["good"] + TYPES:
        cm.append("  ".join(["{:>10}".format(t)] + ["{:>10}".format(int(((typ == t) & (final == c)).sum())) for c in labels]))
    dm = (y == 1) & (pred1 == 1)
    acc = float((final[dm] == typ[dm]).mean()) if dm.any() else 0.0
    unk = int((final == "unknown").sum())
    # Qwen 확신도 요약 (MMR이 불량으로 넘긴 이미지 기준, 정답 = 정상이면 good / 불량이면 해당 타입)
    flagged = [i for i in range(len(rows)) if pred1[i] == 1 and conf.get(key(rows[i])) is not None]
    ok = [conf[key(rows[i])] for i in flagged if final[i] == typ[i]]
    bad = [conf[key(rows[i])] for i in flagged if final[i] != typ[i]]
    m = lambda v: "{:.3f}".format(float(np.mean(v))) if v else "-"
    conf_line = "Qwen 확신도(선택한 클래스 확률) 평균: 맞힌 {}장 {} / 틀린 {}장 {}".format(len(ok), m(ok), len(bad), m(bad))
    with open(os.path.join(out_dir, "predictions_v6.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["domain", "image_path", "true_type", "mmr_score", "mmr_pred", "qwen_label", "confidence", "final"])
        for i, r in enumerate(rows):
            wr.writerow([r["domain"], r["path"], typ[i], "{:.6f}".format(r["score"]), pred1[i],
                         "-" if pred1[i] == 0 else ("normal" if final[i] == "good" else final[i]),
                         "" if conf.get(key(r)) is None else "{:.4f}".format(conf[key(r)]), final[i]])
    head = ("MMR 불량 판정 {}/{}장 -> Qwen 2차(normal 선택 가능, 정상으로 뒤집은 수는 아래 혼동행렬 good 열) | 불량 타입 분류 정확도(MMR이 잡은 진짜 불량 기준) {:.4f} | unknown {}장\n"
            "{}\n"
            "AUROC는 MMR 점수 기준. Pixel AUROC / PRO 는 Qwen이 이상 맵을 바꾸지 않으므로 MMR 로그 값과 같음.").format(
        int(pred1.sum()), len(rows), acc, unk, conf_line)
    text = head + "\n\n" + "\n".join(lines) + "\n" + "\n".join(cm) + "\n"
    print(text)
    with open(os.path.join(out_dir, "comparison_v6.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "comparison_v6.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(out)
    print("저장: {}/comparison_v6.txt, comparison_v6.csv, predictions_v6.csv".format(out_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(MMR_DIR, "log_MMR_AeBAD_S_54"),
                   help="MMR OUTPUT_DIR (image_scores_*.csv 가 있는 폴더)")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--step", default="all", choices=["all", "qwen", "eval"])
    p.add_argument("--limit", type=int, default=None, help="테스트용: Qwen 호출 장수 제한")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_mmr_csv(a.mmr_out)
    print("MMR 결과 {}장 로드".format(len(rows)))
    if a.step in ("all", "qwen"):
        run_qwen(rows, a.out, a.limit)
    if a.step in ("all", "eval"):
        evaluate(rows, a.out)


if __name__ == "__main__":
    main()
