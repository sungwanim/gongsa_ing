"""offline/build_params.py 를 합성 데이터로 끝까지 실행해 검증한다 (서버·실제 데이터 불필요).

가짜 AeBAD 이미지, MMR 점수 csv/npz, Qwen 저장 결과, 제외/참고 목록을 만들고 build_params.py 를 실제로 실행한 뒤
  - 산출물(params / separation / gallery / summary)이 만들어지는지
  - 갤러리 이미지가 참고·보정·제외 목록과 겹치지 않는지
  - 만들어진 산출물로 에이전트 서비스(모의 백엔드)가 시작되고 검사가 되는지
  - 참고 이미지가 제외 목록에 없어 보정용에 섞이면 분리 검사가 중단시키는지(부정 테스트)
를 확인한다.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(E2E, "agent_service"))
TYPES = ["good", "ablation", "breakdown", "fracture", "groove"]
DOMAINS = ["same", "background", "illumination", "view"]
DEFECTS = TYPES[1:]
PER = 14
ok = []


def check(name, cond, extra=""):
    ok.append(bool(cond))
    print("{} {} {}".format("PASS" if cond else "FAIL", name, extra))


def make_fixture(tmp, refs_in_holdout=True):
    data = os.path.join(tmp, "data")                    # = --data-root (AeBAD 폴더)
    mmr = os.path.join(tmp, "mmr_out")
    res = os.path.join(tmp, "velm_results")
    man = os.path.join(tmp, "manifests")
    for d in (data, mmr, res, man):
        os.makedirs(d, exist_ok=True)
    rnd = np.random.RandomState(0)
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32) / 223.0
    probs = {}
    for dom in DOMAINS:
        rows, maps = [], []
        for t in TYPES:
            os.makedirs(os.path.join(data, "AeBAD_S", "test", t, dom), exist_ok=True)
            for k in range(PER):
                rel = "AeBAD_S/test/{}/{}/IMG_{}.png".format(t, dom, k)
                Image.fromarray(rnd.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(os.path.join(data, rel), "JPEG")
                top = rnd.normal(0.42 if t == "good" else 0.56, 0.06)
                m = 0.2 + 0.03 * rnd.rand(224, 224).astype(np.float32)
                cx, cy = rnd.uniform(0.2, 0.8, 2)
                m += (top - 0.2) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.08 ** 2))
                path = "../../../AeBAD/" + rel
                rows.append((path, 0 if t == "good" else 1, float(m.max())))
                maps.append(m.astype(np.float32))
                pn = float(np.clip(rnd.normal(0.9 if t == "good" else 0.3, 0.1), 0.01, 0.99))
                probs[dom + "|" + path] = {"normal": pn, **{d: (1 - pn) / 4 for d in DEFECTS}}
        with open(os.path.join(mmr, "image_scores_aebad_S_AeBAD_S_{}.csv".format(dom)), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "label", "score", "prediction"])
            for p, l, s in rows:
                w.writerow([p, l, s, int(s >= 0.43)])
        np.savez_compressed(os.path.join(mmr, "anomaly_maps_aebad_S_AeBAD_S_{}.npz".format(dom)),
                            image_paths=np.array([r[0] for r in rows]), labels=np.array([r[1] for r in rows]),
                            scores=np.array([r[2] for r in rows]), anomaly_maps=np.stack(maps))
    with open(os.path.join(res, "qwen_results_v8_n12.jsonl"), "w") as f:
        for k, pr in probs.items():
            f.write(json.dumps({"key": k, "probs": pr}) + "\n")

    def write_manifest(name, rows):
        with open(os.path.join(man, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "defect", "original_path"])
            w.writerows(rows)
    refs = [(dom, d, "AeBAD/AeBAD_S/test/{}/{}/IMG_0.png".format(d, dom)) for d in DEFECTS for dom in DOMAINS[:3]]      # 12장
    extra = [(dom, d, "AeBAD/AeBAD_S/test/{}/{}/IMG_1.png".format(d, dom)) for d in DEFECTS[:3] for dom in DOMAINS[:1]]
    write_manifest("holdout_manifest_12.csv", refs)
    write_manifest("holdout_manifest.csv", (refs if refs_in_holdout else extra) + extra)
    write_manifest("holdout_manifest_60.csv", refs + extra + [(dom, "ablation", "AeBAD/AeBAD_S/test/ablation/{}/IMG_{}.png".format(dom, k)) for dom in DOMAINS for k in (2, 3)])
    return data, mmr, res, man


def run_build(tmp, data, mmr, res, man, out, extra=()):
    cmd = [sys.executable, os.path.join(E2E, "offline", "build_params.py"), "--mmr-out", mmr, "--results-dir", res, "--data-root", data,
           "--holdout", os.path.join(man, "holdout_manifest.csv"), "--refs-manifest", os.path.join(man, "holdout_manifest_12.csv"),
           "--manifests-dir", man, "--gallery-size", "12", "--out", out, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def main():
    tmp = tempfile.mkdtemp()
    data, mmr, res, man = make_fixture(tmp)
    out = os.path.join(tmp, "art")
    r = run_build(tmp, data, mmr, res, man, out)
    print(r.stdout[-1500:])
    check("build_params 정상 종료", r.returncode == 0, (r.stderr or "")[-400:])
    if r.returncode != 0:
        sys.exit(1)
    for f in ("params.json", "separation.json", "gallery.json", "summary.json"):
        check("산출물 " + f, os.path.isfile(os.path.join(out, f)))
    p = json.load(open(os.path.join(out, "params.json")))
    sm = json.load(open(os.path.join(out, "summary.json")))
    check("구간 기준과 임계값이 숫자", all(isinstance(p[k], float) for k in ("tau_lo", "tau_hi", "t")) and p["tau_lo"] <= p["tau_hi"],
          "tau_lo={:.3f} tau_hi={:.3f} t={}".format(p["tau_lo"], p["tau_hi"], p["t"]))
    check("라우터 계수가 13개 특징", len(p["router"]["coef"]) == 13 and len(p["router"]["mean"]) == 13)
    check("Qwen 저장 결과가 보정용 이미지를 모두 덮음", sm["qwen_calib_have"] == sm["qwen_calib_total"], "{}/{}".format(sm["qwen_calib_have"], sm["qwen_calib_total"]))
    check("참고 12장 / 갤러리 12장", sm["n_refs"] == 12 and sm["n_gallery"] == 12, "{} / {}".format(sm["n_refs"], sm["n_gallery"]))
    sep = json.load(open(os.path.join(out, "separation.json")))
    check("세 집합이 서로 겹치지 않음", not (set(sep["refs"]) & set(sep["calib"])) and not (set(sep["refs"]) & set(sep["gallery"])) and not (set(sep["calib"]) & set(sep["gallery"])))
    gal = json.load(open(os.path.join(out, "gallery.json")))
    names = [g["file"] for g in gal["items"]]
    manifest_names = set()
    for fn in os.listdir(man):
        for row in csv.DictReader(open(os.path.join(man, fn))):
            manifest_names.add(row["original_path"].split("AeBAD/", 1)[1])
    check("갤러리가 제외·참고 목록의 이미지를 포함하지 않음", not (set(names) & manifest_names))
    check("갤러리에 정답 라벨 필드가 없음", all(set(g) == {"id", "file", "sha256"} for g in gal["items"]))
    check("갤러리 id 가 서로 다름", len({g["id"] for g in gal["items"]}) == len(gal["items"]))

    # 같은 입력을 다시 실행하면 같은 결과 (재현성)
    out2 = os.path.join(tmp, "art2")
    r2 = run_build(tmp, data, mmr, res, man, out2)
    p2 = json.load(open(os.path.join(out2, "params.json")))
    check("같은 입력이면 같은 기준값(재현성)", r2.returncode == 0 and (p["tau_lo"], p["tau_hi"], p["t"]) == (p2["tau_lo"], p2["tau_hi"], p2["t"])
          and [g["sha256"] for g in gal["items"]] == [g["sha256"] for g in json.load(open(os.path.join(out2, "gallery.json")))["items"]])

    # 산출물로 에이전트 서비스(모의 백엔드)가 시작되고 갤러리 이미지가 검사된다
    import mock_backends
    import urllib.request
    import threading
    from http.server import ThreadingHTTPServer
    import server
    from config import Settings
    s = Settings({"AGENT_ARTIFACTS": out, "AGENT_DATA_ROOT": data, "AGENT_DATA_DIR": os.path.join(tmp, "db")})
    app = mock_backends.build_mock_app(s)       # 갤러리 해시 재검증, 분리 검증 포함
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:{}".format(srv.server_address[1])
    items = json.loads(urllib.request.urlopen(base + "/api/gallery").read())["items"]
    body = urllib.request.urlopen(urllib.request.Request(base + "/api/inspect/gallery/" + items[0]["id"], method="POST")).read().decode()
    check("만들어진 산출물로 서비스 시작 + 갤러리 검사 완료", "event: done" in body and len(items) == 12)
    srv.shutdown()

    # 부정 테스트: 참고 이미지가 제외 목록에 없으면 보정용에 섞여 분리 검사가 중단시켜야 한다
    tmp2 = tempfile.mkdtemp()
    d2, m2, r2_, man2 = make_fixture(tmp2, refs_in_holdout=False)
    rb = run_build(tmp2, d2, m2, r2_, man2, os.path.join(tmp2, "art"))
    check("참고 이미지가 보정용과 겹치면 중단(종료 코드≠0, 분리 위반 메시지)", rb.returncode != 0 and "분리 위반" in (rb.stderr + rb.stdout), (rb.stderr or "")[-200:])

    print("\nbuild_params 테스트: {} / {} 통과".format(sum(ok), len(ok)))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
