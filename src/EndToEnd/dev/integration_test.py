"""실행 중인 에이전트 서비스에 대한 통합·안전 테스트 (표준 라이브러리만 사용, 서버/맥 어디서든 실행).

  1) health / 갤러리 응답 형태(경로·정답 비노출) / 썸네일
  2) 갤러리 이미지 N장 검사: 이벤트 순서, 판정 결과, 소요 시간
  3) 안전: 참고 이미지 업로드 시 경고, 이미지가 아닌 파일 거절, 대시보드·갤러리 응답에 경로·해시 비노출
  4) (--expect-auth) 토큰 없이 접근하면 401
결과는 PASS/FAIL/WARN 줄과 --json-out 요약으로 낸다. 종료 코드: FAIL 이 있으면 1.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

RESULTS = []


def rec(level, name, detail=""):
    RESULTS.append((level, name))
    print("[{}] {} {}".format(level, name, detail), flush=True)


def check(name, ok, detail="", warn=False):
    rec("PASS" if ok else ("WARN" if warn else "FAIL"), name, detail)
    return ok


def request(url, token, data=None, method=None):
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    r.add_header("X-E2E-Test", "1")        # 이 점검의 검사는 대시보드에 보이지 않게 표시(점검 후 자동 정리됨)
    if token:
        r.add_header("Authorization", "Bearer " + token)
    if data is not None:
        r.add_header("Content-Type", "application/octet-stream")
    return r


def get(url, token):
    with urllib.request.urlopen(request(url, token), timeout=30) as r:
        return r.status, r.read()


def sse(url, token, data=None):
    """검사를 실행하고 (이벤트 목록, 클라이언트 측 소요 시간) 을 돌려준다. HTTP 오류는 예외."""
    t0 = time.time()
    events, ev = [], None
    with urllib.request.urlopen(request(url, token, data=data, method="POST"), timeout=900) as resp:
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                events.append((ev, json.loads(line[6:])))
    return events, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--token", default="")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--latency-warn", type=float, default=30.0)
    ap.add_argument("--expect-auth", action="store_true")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    base, tok = a.url.rstrip("/"), a.token

    # 1) health / 갤러리
    try:
        st, body = get(base + "/api/health", "")
    except Exception as e:
        rec("FAIL", "서비스 연결", "{} ({})".format(type(e).__name__, e))
        return finish(a, [])
    h = json.loads(body)
    check("health ready", st == 200 and h.get("ready") is True, str(h))
    st, body = get(base + "/api/gallery", tok)
    items = json.loads(body)["items"]
    leak = ("AeBAD" in body.decode()) or (".png" in body.decode())
    check("갤러리 응답에는 id/thumb 만 있고 경로·파일명이 없음", items and all(set(i) == {"id", "thumb"} for i in items) and not leak,
          "{}개".format(len(items)))
    if items:
        st, th = get(base + items[0]["thumb"], tok)
        check("썸네일 제공", st == 200 and len(th) > 100)

    # 2) 검사 N장
    runs = []
    for i in range(min(a.n, len(items))):
        try:
            events, client_s = sse("{}/api/inspect/gallery/{}".format(base, items[i]["id"]), tok)
        except urllib.error.HTTPError as e:
            check("검사 #{}".format(i), False, "HTTP {} {}".format(e.code, e.read().decode(errors="replace")[:200]))
            continue
        names = [e for e, _ in events]
        d = dict(events)
        ok_order = names[:2] == ["start", "mmr"] and names[-2:] == ["final", "done"] and "error" not in names
        if "error" in names:
            check("검사 #{}".format(i), False, "error 이벤트: {}".format(d["error"]["message"]))
            continue
        fin = d.get("final", {})
        tm = fin.get("timings", {})
        runs.append({"total_s": tm.get("total_s", client_s), "mmr_s": tm.get("mmr_s"), "agent_s": tm.get("agent_s"),
                     "tools": fin.get("tools"), "zone": fin.get("zone"), "decision": fin.get("decision"), "defect_type": fin.get("defect_type")})
        check("검사 #{} 이벤트 순서(start→mmr→…→final→done)".format(i), ok_order, ">".join(fin.get("tools", [])) + " → {}/{}".format(fin.get("decision"), fin.get("defect_type")))
        check("검사 #{} 소요 시간 {:.1f}초 (기준 {:.0f}초)".format(i, runs[-1]["total_s"], a.latency_warn), runs[-1]["total_s"] <= a.latency_warn, warn=True)

    # 3) 안전
    try:
        refs = json.load(open(os.path.join(a.artifacts, "gallery.json"))).get("refs", [])
    except Exception:
        refs = []
    ref_path = os.path.join(a.data_root, refs[0]["file"]) if refs else None
    if ref_path and os.path.isfile(ref_path):
        try:
            events, _ = sse(base + "/api/inspect", tok, data=open(ref_path, "rb").read())
            w = dict(events).get("start", {}).get("warning")
            check("참고 이미지를 올리면 경고", bool(w) and "참고" in w, str(w))
        except Exception as e:
            check("참고 이미지 업로드", False, str(e))
    else:
        check("참고 이미지 업로드 경고 점검", False, "참고 이미지 파일을 찾지 못해 건너뜀", warn=True)
    try:
        sse(base + "/api/inspect", tok, data=b"this is not an image")
        check("이미지가 아닌 파일은 거절(400)", False, "거절되지 않음")
    except urllib.error.HTTPError as e:
        check("이미지가 아닌 파일은 거절(400)", e.code == 400, "HTTP {}".format(e.code))
    st, body = get(base + "/api/dashboard?include_test=1", tok)
    txt = body.decode()
    dash = json.loads(txt)["items"]
    check("대시보드에 결과 저장", len(dash) >= len(runs), "{}건".format(len(dash)))
    check("대시보드 응답에 경로·파일명·해시가 없음", "AeBAD" not in txt and "sha256" not in txt and ".png" not in txt)
    st, body = get(base + "/api/dashboard", tok)
    shown = [i for i in json.loads(body)["items"] if i.get("is_test")]
    check("점검용 검사는 기본 대시보드 목록에 보이지 않음", not shown, "{}건 노출".format(len(shown)))

    # 4) 인증
    if a.expect_auth:
        try:
            get(base + "/api/dashboard", "")
            check("토큰 없이 접근하면 401", False, "접근됨")
        except urllib.error.HTTPError as e:
            check("토큰 없이 접근하면 401", e.code == 401, "HTTP {}".format(e.code))
    return finish(a, runs)


def finish(a, runs):
    n = {k: sum(1 for l, _ in RESULTS if l == k) for k in ("PASS", "FAIL", "WARN")}
    totals = [r["total_s"] for r in runs]
    summary = {"pass": n["PASS"], "fail": n["FAIL"], "warn": n["WARN"], "runs": runs,
               "first_total_s": totals[0] if totals else None,
               "rest_median_s": statistics.median(totals[1:]) if len(totals) > 1 else None,
               "max_total_s": max(totals) if totals else None}
    print("\n통합 테스트: PASS {} / FAIL {} / WARN {}".format(n["PASS"], n["FAIL"], n["WARN"]))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(summary, f, ensure_ascii=False)
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
