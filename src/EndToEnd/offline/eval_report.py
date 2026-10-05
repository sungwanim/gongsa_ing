"""보고용 이미지(보고용 절반, 제외 목록 빼고 829장)를 실행 중인 에이전트 서비스에 한 장씩 보내 최종 성능을 잰다.

- 실제 데모와 같은 경로(MMR 서비스 -> 에이전트 서비스 -> 실시간 Qwen)를 쓰므로, 서비스가 떠 있어야 한다 (e2e.sh up / expose).
- 정답 라벨은 이 스크립트(클라이언트)만 안다. 서비스에는 이미지만 보낸다.
- 점검용 표시(X-E2E-Test)를 붙여 대시보드에는 쌓이지 않는다.
- 결과는 --out 폴더의 eval_trace.jsonl(한 장당 한 줄, 이어서 하기 가능)과 eval_summary.md/json.
- 파라미터(구간 기준, t)는 보정용 절반으로만 정해졌고, 여기서 보는 이미지는 그것과 겹치지 않는다.
사용 (서버, velm_qwen 환경): python eval_report.py --data-root ~/end2end/gongsa_ing/AeBAD --out ~/end2end/eval [--limit N] [--resume]
"""
import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VELM = os.path.normpath(os.path.join(HERE, "..", "..", "ModelB", "velm"))
sys.path.insert(0, VELM)
import qwen_eval as q  # noqa: E402

DEF4 = ["ablation", "breakdown", "fracture", "groove"]


def inspect(url, data, token, timeout=900):
    """이미지 한 장을 검사하고 (이벤트 dict, 클라이언트 소요시간) 을 돌려준다. 서비스가 바쁘면(409) 잠시 기다렸다 다시 시도한다."""
    for _ in range(60):
        req = urllib.request.Request(url + "/api/inspect", data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream", "X-E2E-Test": "1"})
        if token:
            req.add_header("Authorization", "Bearer " + token)
        t0 = time.time()
        try:
            events, ev = {}, None
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode().rstrip("\n")
                    if line.startswith("event: "):
                        ev = line[7:]
                    elif line.startswith("data: "):
                        events[ev] = json.loads(line[6:])
            return events, time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code == 409:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("서비스가 계속 바쁩니다 (409)")


def summarize(res, rows_all, thr):
    """res: 처리된 한 장당 기록 목록. 마크다운 요약 문자열과 dict 를 돌려준다."""
    y = np.array([r["label"] for r in res])
    ag = np.array([1 if r["decision"] == "defect" else 0 for r in res])
    sc = np.array([r["mmr_score"] for r in res])
    mmr = (sc >= thr).astype(int)
    m_ag, m_mmr = q.metrics(y, ag), q.metrics(y, mmr, sc)
    out = {"n": len(res), "n_normal": int((y == 0).sum()), "n_defect": int((y == 1).sum()), "mmr_threshold": thr,
           "agent": m_ag, "mmr_only": m_mmr}

    def row(name, m):
        return "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} |".format(
            name, m["Accuracy"], m["Recall"], m["FPR"], m["Precision"], m["F1"], m["TP"], m["FN"], m["FP"], m["TN"])

    L = ["# 보고용 이미지 평가 결과", "",
         "- 처리한 이미지 {}장 (정상 {} / 불량 {}), 보정용 이미지와 겹치지 않음".format(out["n"], out["n_normal"], out["n_defect"]),
         "- MMR 단독 임계값 {:.6f} (MMR 결과 csv 의 판정 경계에서 역산한 추정값)".format(thr),
         "- MMR 단독 이미지 AUROC {:.4f}".format(m_mmr["ImageAUROC"]), "",
         "| 방식 | Accuracy | Recall | FPR | Precision | F1 | TP | FN | FP | TN |", "|---|---|---|---|---|---|---|---|---|---|",
         row("MMR 단독", m_mmr), row("MMR + Qwen 에이전트", m_ag), ""]

    # 구간별
    L += ["## 구간별 (에이전트)", "", "| 구간 | 장수 | 정상 | 불량 | 불량으로 판정 | 놓친 불량(FN) | 오경보(FP) |", "|---|---|---|---|---|---|---|"]
    for z in ("clear_normal", "amb", "confident"):
        s = [r for r in res if r["zone"] == z]
        if not s:
            continue
        yy = np.array([r["label"] for r in s]); pp = np.array([1 if r["decision"] == "defect" else 0 for r in s])
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(z, len(s), int((yy == 0).sum()), int((yy == 1).sum()), int(pp.sum()),
                                                              int(((yy == 1) & (pp == 0)).sum()), int(((yy == 0) & (pp == 1)).sum())))
    L.append("")

    # 불량 종류 (정답이 불량이고 불량으로 판정된 것)
    hit = [r for r in res if r["label"] == 1 and r["decision"] == "defect"]
    ok = sum(1 for r in hit if r["defect_type"] == r["type"])
    out["type_accuracy"] = ok / len(hit) if hit else None
    L += ["## 불량 종류 (불량을 불량으로 잡은 {}장 중)".format(len(hit)), "", "정확도 {:.3f} ({}/{})".format(out["type_accuracy"] or 0, ok, len(hit)), "",
          "| 정답 \\ 예측 | " + " | ".join(DEF4 + ["unknown"]) + " |", "|---|" + "---|" * (len(DEF4) + 1)]
    for t in DEF4:
        c = collections.Counter(r["defect_type"] for r in hit if r["type"] == t)
        L.append("| {} | ".format(t) + " | ".join(str(c.get(k, 0)) for k in DEF4 + ["unknown"]) + " |")
    L.append("")

    # 도구와 시간
    n_tools = [len(r["tools"]) for r in res]
    use = collections.Counter(t for r in res for t in r["tools"])
    tot = np.array([r["timings"]["total_s"] for r in res])
    out["tools_mean"] = float(np.mean(n_tools)); out["sec_median"] = float(np.median(tot)); out["sec_p95"] = float(np.percentile(tot, 95))
    L += ["## 도구 호출과 시간", "", "- 이미지당 평균 도구 호출 {:.2f}회 (decide 포함)".format(out["tools_mean"]),
          "- 도구별 사용 횟수: " + ", ".join("{} {}".format(k, v) for k, v in sorted(use.items())),
          "- 이미지당 소요 시간: 중앙값 {:.1f}초, 95% {:.1f}초, 최대 {:.1f}초".format(out["sec_median"], out["sec_p95"], tot.max()), ""]

    # 안전장치 점검: MMR 단독이 불량이라 한 것을 에이전트가 정상으로 바꾼 적이 있는지는 구간 규칙상 'confident' 에서 0 이어야 한다
    bad = [r for r in res if r["zone"] == "confident" and r["decision"] == "normal"]
    L.append("- 안전장치 점검: 확실한 불량 구간인데 정상으로 판정된 이미지 {}장 (0이어야 정상)".format(len(bad)))
    out["confident_but_normal"] = len(bad)
    return "\n".join(L) + "\n", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--token", default=os.environ.get("AGENT_TOKEN", ""), help="기본: 환경변수 AGENT_TOKEN (화면·기록에 출력하지 않음)")
    ap.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    ap.add_argument("--holdout", default=q.HOLDOUT)
    ap.add_argument("--data-root", default=os.path.join(q.REPO_ROOT, "AeBAD"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만 (0 이면 전부). 순서는 고정된 섞임 순서라 일부만 해도 편향이 없다")
    ap.add_argument("--resume", action="store_true", help="이미 처리한 이미지는 건너뛰고 이어서")
    ap.add_argument("--summary-only", action="store_true", help="서비스를 부르지 않고 저장된 기록으로 요약만 다시 만든다")
    a = ap.parse_args()

    import hashlib
    rows_all = q.load_mmr_csv(a.mmr_out)
    thr = q.estimate_threshold(rows_all)
    rows = q.apply_holdout(rows_all, a.holdout)
    report = sorted([r for r in rows if q.split_of(r) == "report"], key=lambda r: hashlib.md5(("eval" + q.key(r)).encode()).hexdigest())
    if a.limit:
        report = report[:a.limit]
    print("보고용 {}장 (정상 {} / 불량 {})".format(len(report), sum(r["label"] == 0 for r in report), sum(r["label"] == 1 for r in report)), flush=True)

    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "eval_trace.jsonl")
    done = {}
    if (a.resume or a.summary_only) and os.path.exists(fp):
        for l in open(fp):
            if l.strip():
                x = json.loads(l)
                done[x["key"]] = x
        print("이미 처리한 이미지 {}장".format(len(done)), flush=True)
    if not a.summary_only:
        with open(fp, "a" if a.resume else "w") as f:
            for i, r in enumerate(report, 1):
                k = q.key(r)
                if k in done:
                    continue
                p = r["path"].replace("\\", "/")
                path = os.path.join(a.data_root, p[p.index("AeBAD/") + len("AeBAD/"):])
                with open(path, "rb") as g:
                    data = g.read()
                try:
                    ev, sec = inspect(a.url, data, a.token)
                except urllib.error.HTTPError as e:
                    # 참고·보정 이미지(422)나 서버 오류는 기록하고 다음으로 (보고용 이미지는 422 가 나오지 않아야 한다)
                    print("[{}/{}] {} HTTP {} -> 건너뜀".format(i, len(report), os.path.basename(path), e.code), flush=True)
                    continue
                fin = ev.get("final")
                if not fin:
                    print("[{}/{}] {} 결과 없음: {}".format(i, len(report), os.path.basename(path), ev.get("error")), flush=True)
                    continue
                rec = {"key": k, "file": os.path.basename(path), "domain": r["domain"], "type": r["type"], "label": r["label"],
                       "mmr_score": ev["mmr"]["score"], "zone": fin["zone"], "decision": fin["decision"], "defect_type": fin["defect_type"],
                       "tools": fin["tools"], "type_probs": fin["type_probs"], "why": fin["why"], "timings": fin["timings"], "client_s": round(sec, 2)}
                done[k] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print("[{}/{}] {} 정답 {} -> {} ({}) {:.1f}초".format(i, len(report), rec["file"], r["type"], rec["decision"], rec["zone"], sec), flush=True)
    keys = {q.key(r) for r in report}
    res = [x for k, x in done.items() if k in keys]
    if not res:
        raise SystemExit("처리된 이미지가 없습니다")
    md, js = summarize(res, rows, thr)
    if len(res) < len(report):
        md = "> 주의: 보고용 {}장 중 {}장만 처리됨 (--resume 으로 이어서 실행)\n\n".format(len(report), len(res)) + md
    open(os.path.join(a.out, "eval_summary.md"), "w").write(md)
    json.dump(js, open(os.path.join(a.out, "eval_summary.json"), "w"), ensure_ascii=False, indent=2)
    print("\n" + md)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
