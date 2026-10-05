"""오프라인 1회 실행 (서버, velm_qwen 환경): 실시간 앱이 쓸 고정 파라미터와 이미지 분리 목록을 만든다.

만드는 것 (모두 --out 폴더. 앱은 이 폴더만 읽는다):
  params.json      구간 기준(tau_lo/tau_hi), 임계값 t, 라우터(StandardScaler + 로지스틱) 계수
  separation.json  세 집합(참고 / 보정 / 갤러리)의 해시 목록과 겹침 검사 결과
  gallery.json     데모 갤러리(보고용 절반 - 모든 참고·제외 목록), 정답 라벨 없음

기존 코드(qwen_eval / region_crop / agent)는 수정하지 않고 import 해서 같은 계산을 재현한다.
이 스크립트만 데이터셋 CSV/npz/저장된 Qwen 결과를 읽는다. 앱이 실행되는 동안에는 읽지 않는다.
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
VELM = os.path.normpath(os.path.join(HERE, "..", "..", "ModelB", "velm"))
sys.path.insert(0, VELM)
sys.path.insert(0, HERE)
import qwen_eval as q      # noqa: E402
import region_crop as rc   # noqa: E402
import separation as sep   # noqa: E402


def keyset(manifest):
    return q.load_holdout(manifest)        # {(domain, defect, 파일이름)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    ap.add_argument("--results-dir", required=True, help="팀원의 qwen_results_v8_n12.jsonl 이 있는 폴더 (예: .../velm/results)")
    ap.add_argument("--data-root", default=os.path.join(q.REPO_ROOT, "AeBAD"), help="AeBAD 폴더 (그 아래 AeBAD_S/...)")
    ap.add_argument("--holdout", default=q.HOLDOUT, help="agent.py 가 평가에서 제외하는 목록(기본 holdout_manifest.csv)")
    ap.add_argument("--refs-manifest", default=os.path.join(VELM, "holdout_manifest_12.csv"), help="ask_whole 이 쓰는 참고 이미지 목록")
    ap.add_argument("--manifests-dir", default=VELM, help="holdout_manifest*.csv 가 있는 폴더(제외 목록 전체). 기본: velm 폴더")
    ap.add_argument("--recall-target", type=float, default=0.98)
    ap.add_argument("--fpr-hi", type=float, default=0.02)
    ap.add_argument("--normal-recall", type=float, default=0.97)
    ap.add_argument("--gallery-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="새 폴더 (기존 결과 폴더가 아닌 곳)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- 1) agent.py 와 같은 행 구성 / 분할 / 구간 / 임계값 -------------------------------------------------
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    def locate(path):
        """CSV 의 ../../../AeBAD/AeBAD_S/test/... 를 --data-root 아래 실제 파일로 바꾼다 (다른 폴더에서 실행해도 동작)."""
        p = path.replace("\\", "/")
        return os.path.join(a.data_root, p[p.index("AeBAD/") + len("AeBAD/"):])

    for r in rows:
        r["file"] = locate(r["path"])
        r["split"] = q.split_of(r)
    absent = [r["file"] for r in rows if not os.path.isfile(r["file"])]
    assert not absent, "이미지 파일 {}장을 찾을 수 없음 (예: {}). --data-root 를 확인하세요.".format(len(absent), absent[0])
    calib = [r for r in rows if r["split"] == "calib"]
    report = [r for r in rows if r["split"] == "report"]
    print("행 {}장 (보정용 {} / 보고용 {})".format(len(rows), len(calib), len(report)))

    mi = rc.MapIndex(a.mmr_out)
    X = np.array([rc.map_features(mi.get(r["file"])) for r in rows])
    y = np.array([r["label"] for r in rows])
    cal = np.array([r["split"] == "calib" for r in rows])
    scaler = StandardScaler().fit(X[cal])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(scaler.transform(X[cal]), y[cal])
    z = clf.decision_function(scaler.transform(X))
    for r, v in zip(rows, z):
        r["rscore"] = float(v)
    # 기존 함수와 같은 결과인지 확인 (다르면 중단)
    ref_rows = copy.deepcopy(rows)
    q.attach_router_scores(ref_rows, a.mmr_out)
    diff = max(abs(r["rscore"] - s["rscore"]) for r, s in zip(rows, ref_rows))
    assert diff < 1e-9, "라우터 재현이 기존 attach_router_scores 와 다릅니다 (차이 {})".format(diff)

    lo, hi = q.calibrate_v10([r for r in rows if r["split"] == "calib"], a.recall_target, a.fpr_hi, "rscore")
    q.mark_zones_v10(rows, lo, hi, "rscore")      # agent.py 와 같은 순서: 구간 표시(zone10)를 붙여야 임계값 t 를 구할 수 있다
    q.TAG = "v8_n12"
    p1 = q.load_probs(a.results_dir)
    have = sum(1 for r in calib if q.key(r) in p1)
    print("저장된 Qwen v8_n12 결과가 보정용 {}장 중 {}장에 있음".format(len(calib), have))
    if have < len(calib):
        print("[경고] 보정용 이미지 일부에 Qwen 결과가 없습니다. t 가 기존 최종값과 다를 수 있습니다.")
    missing = [r for r in calib if q.key(r) not in p1]
    # clear_normal 구간 이미지는 Qwen 확률을 보지 않고 정상으로 처리하므로(v10_label) 누락이 t 계산에 영향을 주지 않는다
    miss_by_zone = {z: sum(1 for r in missing if r["zone10"] == z) for z in ("clear_normal", "amb", "confident")}
    print("Qwen 결과가 없는 보정용 이미지 {}장의 구간: {}".format(len(missing), miss_by_zone))
    t = q.choose_t_by_recall(calib, p1, None, a.normal_recall)
    print("구간 기준 tau_lo={:.4f} tau_hi={:.4f} (rscore) | 임계값 t={}".format(lo, hi, t))

    # ---- 2) 세 집합과 분리 검사 ------------------------------------------------------------------------------
    manifests = sorted(glob.glob(os.path.join(a.manifests_dir, "holdout_manifest*.csv")))
    excluded = set()
    for m in manifests:
        excluded |= keyset(m)
    print("제외 목록 {}개 파일, 합계 {}장: {}".format(len(manifests), len(excluded), [os.path.basename(m) for m in manifests]))

    refs = []
    import csv
    with open(a.refs_manifest, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            path = os.path.join(a.data_root, r["original_path"].split("AeBAD/", 1)[1])
            refs.append({"path": path, "label": r["defect"], "condition": r["domain"]})
    missing = [r["path"] for r in refs if not os.path.isfile(r["path"])]
    assert not missing, "참고 이미지를 찾을 수 없음: {}".format(missing[:2])
    refs_sha = {sep.sha256_file(r["path"]) for r in refs}

    t0 = time.time()
    calib_sha = {sep.sha256_file(r["file"]) for r in calib}
    cand = [r for r in report if (r["domain"], r["folder_type"], os.path.basename(r["path"])) not in excluded]
    cand_sha = {r["file"]: sep.sha256_file(r["file"]) for r in cand}
    cand = [r for r in cand if cand_sha[r["file"]] not in refs_sha and cand_sha[r["file"]] not in calib_sha]
    print("갤러리 후보 {}장 (보고용 {}장에서 제외 목록·참고·보정과 같은 파일 제거), 해시 계산 {:.0f}초".format(len(cand), len(report), time.time() - t0))

    rng = random.Random(a.seed)
    strata = {}
    for r in sorted(cand, key=lambda r: r["file"]):
        strata.setdefault((r["domain"], r["type"]), []).append(r)
    chosen = [rng.choice(v) for _, v in sorted(strata.items())]
    rest = [r for r in sorted(cand, key=lambda r: r["file"]) if r not in chosen]
    rng.shuffle(rest)
    chosen += rest[:max(0, a.gallery_size - len(chosen))]
    chosen = chosen[:a.gallery_size] if len(chosen) > a.gallery_size else chosen
    rng.shuffle(chosen)       # 순서만으로 종류를 추측하지 못하게 섞음
    gallery = []
    for r in chosen:
        gid = hashlib.sha1(("gallery" + str(a.seed) + cand_sha[r["file"]]).encode()).hexdigest()[:12]
        rel = os.path.relpath(r["file"], a.data_root).replace(os.sep, "/")
        gallery.append({"id": gid, "file": rel, "sha256": cand_sha[r["file"]]})
    gallery_sha = {g["sha256"] for g in gallery}

    sets = {"refs": refs_sha, "calib": calib_sha, "gallery": gallery_sha}
    bad = sep.check_disjoint(sets)
    assert not bad, "이미지 분리 위반: {}".format({"{}-{}".format(*k): v for k, v in bad.items()})
    print("분리 검사 통과: 참고 {} / 보정 {} / 갤러리 {} (서로 겹침 0)".format(len(refs_sha), len(calib_sha), len(gallery_sha)))

    # ---- 3) 저장 ---------------------------------------------------------------------------------------------
    try:
        commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    params = {
        "field": "rscore", "tau_lo": lo, "tau_hi": hi, "t": t,
        "targets": {"recall_target": a.recall_target, "fpr_hi": a.fpr_hi, "normal_recall": a.normal_recall},
        "router": {"feature_names": rc.MAP_FEATURE_NAMES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])},
        "n_calib": len(calib), "refs_manifest": os.path.basename(a.refs_manifest), "n_refs": len(refs),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "code_commit": commit,
    }
    with open(os.path.join(a.out, "params.json"), "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "separation.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in sets.items()}, f)
    with open(os.path.join(a.out, "gallery.json"), "w") as f:
        json.dump({"items": gallery, "refs": [{"file": os.path.relpath(r["path"], a.data_root).replace(os.sep, "/"),
                                              "label": r["label"], "condition": r["condition"]} for r in refs]}, f, ensure_ascii=False, indent=2)
    summary = {"n_rows": len(rows), "n_calib": len(calib), "n_report": len(report), "qwen_calib_have": have,
               "qwen_calib_total": len(calib), "qwen_missing_by_zone": miss_by_zone,
               "qwen_missing_effective": miss_by_zone["amb"] + miss_by_zone["confident"], "tau_lo": lo, "tau_hi": hi, "t": t, "n_refs": len(refs_sha),
               "n_gallery": len(gallery_sha), "n_calib_hashes": len(calib_sha), "n_excluded": len(excluded),
               "manifests": [os.path.basename(m) for m in manifests], "refs_manifest": os.path.basename(a.refs_manifest),
               "overlap": 0, "created_at": params["created_at"], "code_commit": commit}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("저장:", a.out, "->", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
