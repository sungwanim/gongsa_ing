"""서버 통합 점검용 클라이언트 (표준 라이브러리만 사용, 어느 conda 환경에서든 실행 가능).

  python3 sse_probe.py inspect --gallery N      N번째(0부터) 갤러리 이미지로 검사하고 이벤트를 시간과 함께 출력
  python3 sse_probe.py inspect --file 경로      파일을 업로드해 검사
  python3 sse_probe.py dashboard                저장된 결과 요약(구간, 판정, 도구, 소요 시간)
  python3 sse_probe.py health
공통 옵션: --url (기본 http://127.0.0.1:8200)  --token (AGENT_TOKEN 이 있을 때)
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def req(url, token, data=None, method=None):
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    if token:
        r.add_header("Authorization", "Bearer " + token)
    if data is not None:
        r.add_header("Content-Type", "application/octet-stream")
    return r


def get_json(url, token):
    with urllib.request.urlopen(req(url, token), timeout=30) as r:
        return json.loads(r.read())


def inspect(a):
    if a.file:
        r = req(a.url + "/api/inspect", a.token, data=open(a.file, "rb").read())
    else:
        items = get_json(a.url + "/api/gallery", a.token)["items"]
        if not items:
            sys.exit("갤러리가 비어 있습니다")
        gid = items[a.gallery]["id"]
        r = req(a.url + "/api/inspect/gallery/" + gid, a.token, method="POST")
    t0 = time.time()
    final = None
    try:
        resp = urllib.request.urlopen(r, timeout=900)
    except urllib.error.HTTPError as e:
        sys.exit("HTTP {}: {}".format(e.code, e.read().decode(errors="replace")[:300]))
    ev = None
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            d = json.loads(line[6:])
            if ev == "mmr":
                d = {k: v for k, v in d.items() if k != "map_b64"}
            print("{:6.1f}s  {:<11} {}".format(time.time() - t0, ev, json.dumps(d, ensure_ascii=False)[:170]), flush=True)
            if ev == "final":
                final = d
            if ev == "error":
                sys.exit(1)
    if final:
        print("\n요약: 판정={} 종류={} 구간={} 도구={} 총 {:.1f}초 (MMR {:.1f}s / 에이전트 {:.1f}s)".format(
            final["decision"], final["defect_type"], final["zone"], ">".join(final["tools"]),
            final["timings"]["total_s"], final["timings"]["mmr_s"], final["timings"]["agent_s"]))


def dashboard(a):
    items = get_json(a.url + "/api/dashboard", a.token)["items"]
    print("저장된 결과 {}건".format(len(items)))
    for it in items:
        print("#{:<3} {:<8} {:<12} {:<7} {:<10} {:>6.1f}s  {}{}".format(
            it["id"], it["source"], it["zone"], it["decision"], it["defect_type"] or "-", it["timings"]["total_s"],
            ">".join(it["tools"]), "  [경고: {}]".format(it["sep_flag"]) if it["sep_flag"] else ""))
    if items:
        ts = [i["timings"]["total_s"] for i in items]
        print("평균 {:.1f}초, 최대 {:.1f}초".format(sum(ts) / len(ts), max(ts)))


def main():
    try:
        _main()
    except urllib.error.URLError as e:
        sys.exit("서비스에 연결할 수 없습니다: {} (서비스가 떠 있는지, --url 이 맞는지 확인)".format(getattr(e, "reason", e)))
    except urllib.error.HTTPError as e:
        sys.exit("HTTP {}: {}".format(e.code, e.read().decode(errors="replace")[:200]))


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "dashboard", "health"])
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--token", default="")
    ap.add_argument("--gallery", type=int, default=0)
    ap.add_argument("--file")
    a = ap.parse_args()
    if a.cmd == "health":
        print(get_json(a.url + "/api/health", a.token))
    elif a.cmd == "dashboard":
        dashboard(a)
    else:
        inspect(a)


if __name__ == "__main__":
    main()
