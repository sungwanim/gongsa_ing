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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # GPU 메모리 조각 낭비 줄이기
HERE = os.path.dirname(os.path.abspath(__file__))
MMR_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "ModelA", "MMR_Test"))
TYPES = ["ablation", "breakdown", "fracture", "groove"]
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
HOLDOUT = os.path.join(HERE, "holdout_manifest.csv")   # 평가에서 제외할 60장 (test 이미지 중 따로 뺀 것)
TAG = "v6"      # 프롬프트 이름 (결과 파일 이름에 붙음, --prompt 로 변경)


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
                rows.append({"domain": domain, "path": r["image_path"], "folder_type": anomaly, "label": int(float(r["label"])),
                             "score": float(r["score"]), "pred": int(float(r["prediction"])),
                             "type": "good" if int(float(r["label"])) == 0 else anomaly})
    return rows


def load_holdout(path):
    s = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s.add((r["domain"], r["defect"], os.path.basename(r["original_path"])))
    return s


def apply_holdout(rows, path):
    """제외 목록(도메인, 폴더의 결함 타입, 파일 이름)에 있는 이미지를 평가 대상에서 뺀다. 데이터 파일은 지우지 않는다."""
    hold = load_holdout(path)
    kept = [r for r in rows if (r["domain"], r["folder_type"], os.path.basename(r["path"])) not in hold]
    removed = len(rows) - len(kept)
    print("제외 목록 {}장 중 평가 대상에서 뺀 이미지 {}장 (남은 {}장)".format(len(hold), removed, len(kept)))
    if removed != len(hold):
        print("[경고] 제외 목록과 MMR 결과가 {}장 맞지 않습니다. 파일 이름/도메인을 확인하세요.".format(len(hold) - removed))
    return kept


def build_refs(manifest, per_group):
    """holdout 60장을 (촬영 조건 + 결함 타입) 라벨이 붙은 참고 이미지 목록으로 만든다. 그룹(조건,타입)당 per_group장, 고정 시드로 섞음."""
    import random
    groups = {}
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            path = os.path.join(REPO_ROOT, r["original_path"])
            groups.setdefault((r["domain"], r["defect"]), []).append(
                {"path": path, "label": r["defect"], "condition": r["domain"]})
    refs = []
    for k in sorted(groups):
        refs += sorted(groups[k], key=lambda x: x["path"])[:per_group]
    missing = [r["path"] for r in refs if not os.path.isfile(r["path"])]
    if missing:
        raise SystemExit("참고 이미지 {}장을 찾을 수 없습니다 (예: {}). AeBAD가 {} 아래에 있는지 확인하세요.".format(
            len(missing), missing[0], REPO_ROOT))
    random.Random(0).shuffle(refs)
    print("참고 이미지 {}장 사용 (라벨: 촬영 조건 + 결함 타입)".format(len(refs)), flush=True)
    return refs


def resolve(path):
    if os.path.isfile(path):
        return path
    alt = os.path.normpath(os.path.join(MMR_DIR, path))   # MMR은 MMR_Test 폴더 기준 상대경로로 저장함
    return alt if os.path.isfile(alt) else path


def key(r):
    return r["domain"] + "|" + r["path"]


def estimate_threshold(rows):
    """MMR이 쓴 불량 판정 임계값 추정: CSV의 prediction(=score>=임계값)에서 역산. (정상 판정 중 최고점, 불량 판정 중 최저점) 사이."""
    s0 = [r["score"] for r in rows if r["pred"] == 0]
    s1 = [r["score"] for r in rows if r["pred"] == 1]
    lo, hi = max(s0), min(s1)
    if lo >= hi:
        print("[경고] 정상 판정 최고점({:.5f}) >= 불량 판정 최저점({:.5f}): 임계값이 하나가 아닐 수 있습니다".format(lo, hi))
    return hi          # 불량 판정 중 최저 점수 = 임계값 바로 위 (임계값은 lo 초과, hi 이하)


def mark_zones(rows, thr, pct):
    """임계값 위아래 pct% 를 '애매한 구간'으로 표시. 각 행에 zone 저장:
       clear_normal(확실한 정상) / lower_amb(정상 판정이지만 임계값 바로 아래) /
       upper_amb(불량 판정이지만 임계값 바로 위) / confident(확실한 불량)"""
    up, down = thr * (1 + pct / 100.0), thr * (1 - pct / 100.0)
    for r in rows:
        if r["pred"] == 1:
            r["zone"] = "upper_amb" if r["score"] < up else "confident"
        else:
            r["zone"] = "lower_amb" if r["score"] >= down else "clear_normal"
    return up, down


def band_info(rows):
    thr = estimate_threshold(rows)
    print("MMR 불량 판정 임계값(추정) = {:.5f}".format(thr))
    print("이미지 수: 정상 {} / 불량 {}  (MMR 판정 불량 {}장)".format(
        sum(1 for r in rows if r["label"] == 0), sum(1 for r in rows if r["label"] == 1), sum(1 for r in rows if r["pred"] == 1)))
    print("\n임계값 위아래 N% 구간별 이미지 수 (Qwen이 볼 대상)")
    print("{:>6} | {:>20} | {:>12} {:>12} | {:>8}".format("N(%)", "점수 구간", "정상 이미지", "불량 이미지", "합계"))
    for pct in (5, 10, 20, 30, 50):
        up, down = thr * (1 + pct / 100.0), thr * (1 - pct / 100.0)
        inb = [r for r in rows if down <= r["score"] < up]
        g = sum(1 for r in inb if r["label"] == 0)
        print("{:>6} | {:>9.4f} ~ {:<9.4f} | {:>12} {:>12} | {:>8}".format(pct, down, up, g, len(inb) - g, len(inb)))
    print("\n(참고) 구간이 넓을수록 Qwen이 볼 이미지가 많아집니다. 이 숫자를 보고 --band-pct N 을 정하세요.")


def run_qwen(rows, out_dir, limit=None, subset=None, fname=None, refs=None, ref_px=None, use_cache=True, view_fn=None, ref_views=None):
    from run_qwen import load_model, classify_image   # velm 코드 그대로 사용
    fpath = os.path.join(out_dir, fname or "qwen_results_{}.jsonl".format(TAG))
    done = set()
    if os.path.exists(fpath):
        with open(fpath) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [r for r in (subset if subset is not None else rows)
            if (r["pred"] == 1 or r.get("zone") == "lower_amb") and key(r) not in done]
    if limit:
        todo = todo[:limit]
    print("Qwen 대상 {}장 (이미 완료 {}장)".format(len(todo), len(done)), flush=True)
    if not todo:
        return
    model, processor = load_model()
    clf = None
    if view_fn is not None:
        from run_qwen import CachedViewClassifier
        clf = CachedViewClassifier(model, processor, refs, ref_views, view_fn)
    elif refs and use_cache:
        from run_qwen import CachedRefClassifier
        clf = CachedRefClassifier(model, processor, refs, ref_px)
    with open(fpath, "a") as f:
        for n, r in enumerate(todo, 1):
            try:
                if clf is not None:
                    out = clf.classify(resolve(r["path"]))
                else:
                    out = classify_image(resolve(r["path"]), model, processor, prompt=TAG, refs=refs, ref_max_pixels=ref_px)
            except Exception as e:
                # 오류는 결과로 저장하지 않는다 (저장하면 '끝난 이미지'로 취급되어 다시 판정되지 않음)
                msg = str(e)
                print("[오류] {} 처리 실패, 저장하지 않고 건너뜀: {}: {}".format(os.path.basename(r["path"]), type(e).__name__, msg[:200]), flush=True)
                if "out of memory" in msg.lower():
                    raise SystemExit("GPU 메모리 부족으로 중단합니다. 같은 GPU를 쓰는 다른 작업이 있는지 확인하세요. (저장된 결과는 이어서 할 수 있습니다)")
                continue
            f.write(json.dumps({"key": key(r), "raw": out["raw"], "pred_type": out["label"],
                                "confidence": out["confidence"], "probs": out["probs"],
                                "class_mass": out["class_mass"]}, ensure_ascii=False) + "\n")
            f.flush()
            print("[{}/{}] {} -> {} ({:.1%})".format(n, len(todo), os.path.basename(r["path"]), out["label"], out["confidence"]), flush=True)
    if clf is not None:
        print("캐시 사용 {}장 / 전체 다시 계산 {}장 (캐시 상태: {})".format(clf.n_cached, clf.n_fallback, clf.state), flush=True)


def quick_subset(rows, n):
    """MMR이 불량으로 넘긴 이미지 중 (정상 포함) 타입별 n장을 고정 시드로 뽑는다 -> 프롬프트끼리 같은 이미지로 비교."""
    import random
    rnd = random.Random(0)
    by = {}
    for r in rows:
        if r["pred"] == 1:
            by.setdefault(r["type"], []).append(r)
    sub = []
    for t in ["good"] + TYPES:
        lst = by.get(t, [])
        sub += rnd.sample(lst, min(n, len(lst)))
    return sub


def quick_report(sub, out_dir, fname):
    res = {}
    with open(os.path.join(out_dir, fname)) as f:
        for l in f:
            if l.strip():
                d = json.loads(l); res[d["key"]] = d
    cols = ["normal"] + TYPES + ["unknown"]
    lines = ["[빠른 시험] 프롬프트 {} / MMR이 불량으로 넘긴 이미지 중 타입별 샘플".format(TAG), "",
             "{:>10} {:>5} | ".format("정답", "장수") + " ".join("{:>9}".format(c) for c in cols) + " | 맞힘  정상으로답함"]
    for t in ["good"] + TYPES:
        rs = [r for r in sub if r["type"] == t and key(r) in res]
        if not rs:
            continue
        cnt = {c: sum(1 for r in rs if res[key(r)]["pred_type"] == c) for c in cols}
        right = cnt["normal"] if t == "good" else cnt.get(t, 0)
        lines.append("{:>10} {:>5} | ".format("정상" if t == "good" else t, len(rs)) + " ".join("{:>9}".format(cnt[c]) for c in cols)
                     + " | {:>4.0%}  {:>6.0%}".format(right / len(rs), cnt["normal"] / len(rs)))
    lines += ["", "해석: 정상 행은 '정상으로답함'이 높을수록 좋고, 불량 행은 '맞힘'이 높고 '정상으로답함'이 낮을수록 좋음."]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(out_dir, "quick_{}.txt".format(TAG)), "w") as f:
        f.write(text)


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


def evaluate(rows, out_dir, quiet=False, band_pct=None):
    out_tag = TAG + ("_b{:g}".format(band_pct) if band_pct else "")
    qwen, conf, probs_map = {}, {}, {}
    fpath = os.path.join(out_dir, "qwen_results_{}.jsonl".format(TAG))
    if os.path.exists(fpath):
        with open(fpath) as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    qwen[d["key"]] = "good" if d["pred_type"] == "normal" else d["pred_type"]
                    conf[d["key"]] = d.get("confidence")
                    probs_map[d["key"]] = d.get("probs") or {}
    final = []                       # MMR+Qwen 최종 라벨: good 또는 타입
    for r in rows:
        zone = r.get("zone") if band_pct else None
        if band_pct is None:
            # 기본: MMR이 불량으로 본 것은 전부 Qwen이 정상/타입 판정
            if r["pred"] == 0:
                final.append("good")
            else:
                if key(r) not in qwen:
                    raise SystemExit("Qwen 결과 누락: {} -> 먼저 Qwen 단계를 실행하세요".format(key(r)))
                final.append(qwen[key(r)])
            continue
        # 애매한 구간만 Qwen이 정상/불량 판정, 확실한 불량은 Qwen이 타입만 분류
        if zone == "clear_normal":
            final.append("good")
            continue
        if key(r) not in qwen:
            raise SystemExit("Qwen 결과 누락: {} -> 먼저 같은 --band-pct 로 Qwen 단계(--step qwen)를 실행하세요".format(key(r)))
        q = qwen[key(r)]
        if zone == "confident" and q == "good":          # 확실한 불량은 정상으로 못 뒤집음 -> 불량 타입 4개 중 확률 최대
            pr = {k: v for k, v in probs_map.get(key(r), {}).items() if k != "normal"}
            q = max(pr, key=pr.get) if pr else "unknown"
        final.append(q)                                   # lower_amb: Qwen이 불량이라고 하면 불량으로 바뀜(구출), upper_amb: Qwen이 정상이라 하면 정상
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
    with open(os.path.join(out_dir, "predictions_{}.csv".format(out_tag)), "w", newline="") as f:
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
    if not quiet:
        print(text)
    with open(os.path.join(out_dir, "comparison_{}.txt".format(out_tag)), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "comparison_{}.csv".format(out_tag)), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(out)
    if not quiet:
        print("저장: {0}/comparison_{1}.txt, comparison_{1}.csv, predictions_{1}.csv".format(out_dir, out_tag))
    return out, {"type_acc": acc, "unknown": unk, "conf_line": conf_line}


def sweep_normal(rows, out_dir):
    """저장된 확률로 'normal 확률 기준값(t)'을 바꿔 가며 결과를 비교한다 (GPU 불필요).
    규칙: MMR이 불량으로 본 이미지에서 Qwen의 normal 확률 >= t 이면 정상, 아니면 불량.
          불량 타입은 normal을 뺀 4개 중 확률이 가장 큰 것. (t > 1 이면 MMR 불량 판정을 전부 유지하고 타입만 Qwen이 정함)"""
    fpath = os.path.join(out_dir, "qwen_results_{}.jsonl".format(TAG))
    if not os.path.exists(fpath):
        raise SystemExit("결과 파일이 없습니다: {}".format(fpath))
    pm = {}
    with open(fpath) as f:
        for l in f:
            if l.strip():
                d = json.loads(l)
                if d.get("probs"):
                    pm[d["key"]] = d["probs"]
    flagged = [r for r in rows if r["pred"] == 1]
    miss = [r for r in flagged if key(r) not in pm]
    if miss:
        raise SystemExit("확률이 없는 이미지 {}장 (예: {}). 먼저 Qwen 단계를 끝내세요.".format(len(miss), key(miss[0])))
    y = np.array([r["label"] for r in rows]); pred1 = np.array([r["pred"] for r in rows])
    typ = np.array([r["type"] for r in rows])
    pn = np.zeros(len(rows)); best = np.array(["good"] * len(rows), dtype=object)
    for i, r in enumerate(rows):
        if r["pred"] == 1:
            pr = pm[key(r)]
            pn[i] = pr.get("normal", 0.0)
            dp = {k: v for k, v in pr.items() if k != "normal"}
            best[i] = max(dp, key=dp.get) if dp else "unknown"
    n_def = int((y == 1).sum())
    lines = ["MMR 불량 판정 {}장 중 Qwen 확률로 재판정. 평가 대상 {}장 (정상 {} / 불량 {}).".format(len(flagged), len(rows), int((y == 0).sum()), n_def),
             "규칙: normal 확률 >= t 이면 정상, 아니면 불량(타입 = normal 제외 4개 중 확률 최대). t가 낮을수록 불량으로 더 많이 판정.", "",
             "{:>7} | {:>8} {:>9} {:>7} {:>7} {:>7} | {:>5} {:>5} {:>5} {:>5} | {:>20}".format(
                 "t", "Accuracy", "Precision", "Recall", "FPR", "F1", "TN", "FP", "FN", "TP", "타입 맞힘(불량 기준)")]
    m = metrics(y, pred1)
    lines.append("{:>7} | {:>8.4f} {:>9.4f} {:>7.4f} {:>7.4f} {:>7.4f} | {:>5} {:>5} {:>5} {:>5} | {:>20}".format(
        "MMR단독", m["Accuracy"], m["Precision"], m["Recall"], m["FPR"], m["F1"], m["TN"], m["FP"], m["FN"], m["TP"], "-"))
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01):
        final = np.where((pred1 == 1) & (pn < t), best, "good")
        pred2 = (final != "good").astype(int)
        m = metrics(y, pred2)
        ok = int(((y == 1) & (final == typ)).sum())
        lines.append("{:>7} | {:>8.4f} {:>9.4f} {:>7.4f} {:>7.4f} {:>7.4f} | {:>5} {:>5} {:>5} {:>5} | {:>13}/{:<6}".format(
            ("%g" % t) if t <= 1 else "항상불량", m["Accuracy"], m["Precision"], m["Recall"], m["FPR"], m["F1"],
            m["TN"], m["FP"], m["FN"], m["TP"], ok, n_def))
    lines += ["", "참고: t는 이 평가 이미지로 고르면 결과가 실제보다 좋게 보입니다(낙관적). 보고서에는 t를 어떻게 정했는지 같이 적으세요.",
              "      타입 맞힘 = 불량 이미지 중 최종 타입이 정답과 같은 수. '항상불량' 줄은 MMR이 불량이라 한 것을 전부 불량으로 두고 타입만 Qwen이 정한 경우."]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(out_dir, "sweep_{}.txt".format(TAG)), "w") as f:
        f.write(text)
    print("저장: {}/sweep_{}.txt".format(out_dir, TAG))


def compare_all(rows, out_dir, tags):
    """MMR 단독 + 프롬프트별(MMR+Qwen) 결과를 한 표로 비교. 각 프롬프트의 qwen_results_<tag>.jsonl 이 필요."""
    global TAG
    res = {}
    for t in tags:
        TAG = t
        res[t] = evaluate(rows, out_dir, quiet=True)
    cols = ["scope", "model", "n_good", "n_defect", "Accuracy", "Precision", "Recall", "FPR", "F1", "TP", "FN", "FP", "TN"]
    first = res[tags[0]][0]
    table = []
    scopes = []
    for r in first:
        if r["scope"] not in scopes:
            scopes.append(r["scope"])
    for sc in scopes:
        table.append(next(r for r in first if r["scope"] == sc and r["model"] == "MMR"))
        for t in tags:
            row = dict(next(r for r in res[t][0] if r["scope"] == sc and r["model"] == "MMR+Qwen"))
            row["model"] = "MMR+Qwen({})".format(t)
            table.append(row)
    f4 = lambda v: "{:.4f}".format(v) if isinstance(v, float) else str(v)
    w = [max(len(c), max(len(f4(r[c])) for r in table)) for c in cols]
    lines = [" | ".join(c.ljust(w[i]) for i, c in enumerate(cols)), "-+-".join("-" * x for x in w)]
    prev = None
    for r in table:
        if prev is not None and r["scope"] != prev:
            lines.append("")
        prev = r["scope"]
        lines.append(" | ".join(f4(r[c]).ljust(w[i]) for i, c in enumerate(cols)))
    foot = ["", "[프롬프트별 불량 타입 분류 정확도(MMR이 잡은 진짜 불량 기준) / unknown 수 / Qwen 확신도]"]
    for t in tags:
        x = res[t][1]
        foot.append("  {}: 타입 정확도 {:.4f}, unknown {}장 | {}".format(t, x["type_acc"], x["unknown"], x["conf_line"]))
    head = "평가 대상 {}장 (holdout 60장 제외). 프롬프트: {}\n".format(len(rows), ", ".join(tags)).replace("\\n", "\n")
    text = head + "\n" + "\n".join(lines) + "\n" + "\n".join(foot) + "\n"
    print(text)
    with open(os.path.join(out_dir, "comparison_all.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out_dir, "comparison_all.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(table)
    print("저장: {0}/comparison_all.txt, comparison_all.csv".format(out_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(MMR_DIR, "log_MMR_AeBAD_S_54"),
                   help="MMR OUTPUT_DIR (image_scores_*.csv 가 있는 폴더)")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--step", default="all", choices=["all", "qwen", "eval"])
    p.add_argument("--limit", type=int, default=None, help="테스트용: Qwen 호출 장수 제한")
    p.add_argument("--prompt", default="v6", help="프롬프트 이름: v6(짧은 원본+normal), v7(타입 특징 설명), v8(라벨 붙은 참고 이미지), v9(v8 + MMR 위치 박스/확대). 결과 파일 이름에 붙음")
    p.add_argument("--holdout", default=HOLDOUT, help="평가에서 제외할 이미지 목록 csv (기본: holdout_manifest.csv)")
    p.add_argument("--no-holdout", action="store_true", help="제외 없이 전체로 평가")
    p.add_argument("--ref-per-group", type=int, default=5, help="v8: (촬영 조건, 결함 타입) 그룹당 참고 이미지 수. 제외 목록(holdout_manifest.csv)에 있는 만큼까지 사용 (지금은 그룹당 1장 = 12장)")
    p.add_argument("--ref-px", type=int, default=128, help="v8: 참고 이미지 한 장당 토큰 수 (128이면 약 128*28*28 화소). 메모리 부족이면 줄이기")
    p.add_argument("--compare", default=None, metavar="TAGS",
                   help="저장된 결과로 MMR 단독 + 여러 프롬프트를 한 표로 비교. 예: v6,v7,v8")
    p.add_argument("--band-pct", type=float, default=None, metavar="N",
                   help="MMR 불량 판정 임계값 위아래 N%% 안의 애매한 이미지만 Qwen이 정상/불량 판정(확실한 불량은 타입만 분류). 예: --band-pct 20")
    p.add_argument("--sweep-normal", action="store_true", help="저장된 확률로 normal 확률 기준값(t)을 바꿔 가며 정확도/재현율/F1/타입 맞힘을 표로 출력 (GPU 불필요)")
    p.add_argument("--band-info", action="store_true", help="MMR 점수 분포와 N%%별 애매한 이미지 수를 출력하고 종료 (GPU 불필요)")
    p.add_argument("--no-cache", action="store_true", help="v8: 참고 이미지 캐시를 쓰지 않고 매번 전부 계산 (느림, 비교/확인용)")
    p.add_argument("--dump-views", type=int, default=None, metavar="N",
                   help="v9: 박스/확대 사진이 제대로 만들어지는지 눈으로 보기 위해 타입(정상 포함)별 N장을 results/views/ 에 저장하고 종료 (GPU 불필요)")
    p.add_argument("--quick", type=int, default=None, metavar="N",
                   help="빠른 프롬프트 시험: MMR이 불량으로 넘긴 이미지 중 타입별(정상 포함) N장만 돌려서 정답 분포를 출력")
    a = p.parse_args()
    global TAG
    TAG = a.prompt
    os.makedirs(a.out, exist_ok=True)
    rows = load_mmr_csv(a.mmr_out)
    print("MMR 결과 {}장 로드 / 프롬프트 {}".format(len(rows), TAG))
    if not a.no_holdout:
        if not os.path.exists(a.holdout):
            raise SystemExit("제외 목록 파일이 없습니다: {} (제외 없이 평가하려면 --no-holdout)".format(a.holdout))
        rows = apply_holdout(rows, a.holdout)
    if a.band_info:
        band_info(rows)
        return
    if a.band_pct:
        thr = estimate_threshold(rows)
        up, down = mark_zones(rows, thr, a.band_pct)
        z = {}
        for r in rows:
            z[r["zone"]] = z.get(r["zone"], 0) + 1
        print("애매한 구간: 임계값 {:.5f} 기준 위아래 {:g}% = 점수 {:.5f} ~ {:.5f} | 구간별 이미지 수 {}".format(thr, a.band_pct, down, up, z), flush=True)
    if a.sweep_normal:
        sweep_normal(rows, a.out)
        return
    if a.compare:
        compare_all(rows, a.out, [t.strip() for t in a.compare.split(",") if t.strip()])
        return
    refs = ref_px = None
    view_fn = ref_views = None
    if a.prompt in ("v8", "v9"):
        if a.no_holdout:
            raise SystemExit("v8은 holdout 60장을 참고 이미지로 쓰므로 --no-holdout 과 함께 쓸 수 없습니다 (평가 오염).")
        refs = build_refs(a.holdout, a.ref_per_group)
        ref_px = a.ref_px * 28 * 28
        # 결과 파일 이름: 기본(참고 이미지 60장, 128토큰)과 다르면 장수/크기를 이름에 붙여 기존 결과와 섞이지 않게 한다
        suffix = ""
        if len(refs) != 60:
            suffix += "_n{}".format(len(refs))
        if a.ref_px != 128:
            suffix += "_p{}".format(a.ref_px)
        if a.prompt == "v9":
            import region_crop
            maps = region_crop.MapIndex(a.mmr_out)
            print("이상 맵 {}장 로드, 참고 이미지 {}장의 위치는 정답 마스크로 계산".format(len(maps.maps), len(refs)), flush=True)
            ref_views = [region_crop.views_from_mask(r["path"]) for r in refs]
            view_fn = lambda path: region_crop.views_from_map(resolve(path), maps)
        if a.dump_views and a.prompt == "v9":
            vdir = os.path.join(a.out, "views")
            os.makedirs(vdir, exist_ok=True)
            for i, (r, (ov, cr)) in enumerate(zip(refs, ref_views), 1):
                ov.save(os.path.join(vdir, "ref_{:02d}_{}_{}_box.jpg".format(i, r["condition"], r["label"])), quality=92)
                cr.save(os.path.join(vdir, "ref_{:02d}_{}_{}_crop.jpg".format(i, r["condition"], r["label"])), quality=92)
            for r in quick_subset(rows, a.dump_views):
                ov, cr = view_fn(r["path"])
                base = "q_{}_{}_{}".format(r["type"], r["domain"], os.path.basename(r["path"])[:-4])
                ov.save(os.path.join(vdir, base + "_box.jpg"), quality=92)
                cr.save(os.path.join(vdir, base + "_crop.jpg"), quality=92)
            print("저장: {} (참고 이미지 {}장 x 2, 검사 이미지 타입별 {}장 x 2). 열어서 박스가 결함 위치에 있는지 확인하세요.".format(vdir, len(refs), a.dump_views))
            return
        if suffix:
            TAG = a.prompt + suffix
            print("결과 파일 이름에 붙는 이름: {}".format(TAG), flush=True)
        # 안전장치: 판정 대상에 참고 이미지가 하나라도 남아 있으면 중단
        ref_keys = {(r["condition"], r["label"], os.path.basename(r["path"])) for r in refs}
        overlap = [r for r in rows if (r["domain"], r["folder_type"], os.path.basename(r["path"])) in ref_keys]
        print("판정 대상 {}장 / 참고 이미지와 겹치는 이미지 {}장".format(len(rows), len(overlap)), flush=True)
        if overlap:
            raise SystemExit("참고 이미지가 판정 대상에 섞여 있습니다. 중단합니다.")
    if a.quick:
        sub = quick_subset(rows, a.quick)
        fname = "qwen_quick_{}.jsonl".format(TAG)
        run_qwen(rows, a.out, subset=sub, fname=fname, refs=refs, ref_px=ref_px, use_cache=not a.no_cache, view_fn=view_fn, ref_views=ref_views)
        quick_report(sub, a.out, fname)
        return
    if a.step in ("all", "qwen"):
        run_qwen(rows, a.out, a.limit, refs=refs, ref_px=ref_px, use_cache=not a.no_cache, view_fn=view_fn, ref_views=ref_views)
    if a.step in ("all", "eval"):
        evaluate(rows, a.out, band_pct=a.band_pct)


if __name__ == "__main__":
    main()
