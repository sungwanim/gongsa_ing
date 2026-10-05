"""모의 모드 통합 테스트 (GPU 불필요): python test_local.py"""
import io
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent_service"))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402

import mock_artifacts  # noqa: E402
import mock_backends  # noqa: E402
import server  # noqa: E402
from config import Settings  # noqa: E402

ok = []


def check(name, cond, extra=""):
    ok.append(bool(cond))
    print("{} {} {}".format("PASS" if cond else "FAIL", name, extra))


def sse(url, data=None, token=None):
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/octet-stream"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode()
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def start(settings):
    app = mock_backends.build_mock_app(settings)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return app, srv, "http://127.0.0.1:{}".format(srv.server_address[1])


def main():
    tmp = tempfile.mkdtemp()
    info = mock_artifacts.make(os.path.join(tmp, "art"))
    env = {"AGENT_ARTIFACTS": os.path.join(tmp, "art"), "AGENT_DATA_ROOT": os.path.join(tmp, "art"), "AGENT_DATA_DIR": os.path.join(tmp, "db")}
    app, srv, base = start(Settings(env))

    st, body = get(base + "/api/health")
    check("health", st == 200 and json.loads(body)["status"] == "ok")
    gal = json.loads(get(base + "/api/gallery")[1])["items"]
    check("gallery 목록: id/thumb 만 노출", len(gal) == 6 and set(gal[0]) == {"id", "thumb"}, str(gal[0]))
    st, th = get(base + gal[0]["thumb"])
    check("썸네일 JPEG", st == 200 and Image.open(io.BytesIO(th)).size[0] <= 320)

    # 갤러리 이미지 추론: 이벤트 순서
    ev = sse(base + "/api/inspect/gallery/" + gal[0]["id"])
    names = [e for e, _ in ev]
    check("이벤트 순서 start->mmr->...->final->done", names[0] == "start" and names[1] == "mmr" and names[-2:] == ["final", "done"], str(names))
    mmr = dict(ev)["mmr"]
    check("mmr 이벤트에 점수/구간/히트맵", {"score", "rscore", "zone", "map_b64", "map_shape"} <= set(mmr) and mmr["map_shape"] == [112, 112])
    check("thought/action/observation 포함", {"thought", "action", "observation"} <= set(names) or dict(ev)["final"]["tools"] == ["decide"], str(names))
    fin = dict(ev)["final"]
    check("final 결과", fin["decision"] in ("defect", "normal") and "timings" in fin, json.dumps(fin, ensure_ascii=False)[:160])

    # 여러 업로드: 모든 구간을 한 번씩 보도록 서로 다른 이미지
    zones, decisions = set(), []
    for i in range(12):
        buf = io.BytesIO()
        mock_artifacts.synth_image(500 + i).save(buf, "PNG")
        e = sse(base + "/api/inspect", buf.getvalue())
        d = dict(e)
        zones.add(d["mmr"]["zone"])
        decisions.append((d["mmr"]["zone"], d["final"]["decision"], d["final"]["tools"]))
        assert e[-1][0] == "done", e[-1]
    check("업로드 12장 처리, 구간 2종 이상 관찰", len(zones) >= 2, str(sorted(zones)))
    # 안전장치: 확실한 불량 구간은 항상 불량
    check("확실한 불량 구간은 항상 불량", all(d == "defect" for z, d, _ in decisions if z == "confident"))
    # 안전장치: 애매 구간에서 ask_whole 없이 정상이 되면 안 됨
    check("애매 구간은 ask_whole 없이 정상 불가", all(not (z == "amb" and d == "normal" and "ask_whole" not in t) for z, d, t in decisions))

    # 참고/보정 이미지 해시 경고
    ev = sse(base + "/api/inspect", open(info["calib_file"], "rb").read())
    w = dict(ev)["start"]["warning"]
    check("보정 이미지와 같은 파일이면 경고", w and "파라미터 계산" in w, str(w))

    # 오류 처리
    for name, data, code in [("이미지 아님", b"not an image", 400)]:
        try:
            sse(base + "/api/inspect", data)
            check(name, False)
        except urllib.error.HTTPError as e:
            check(name + " -> {}".format(code), e.code == code)
    # 동시 처리 제한
    app.busy.acquire()
    try:
        sse(base + "/api/inspect/gallery/" + gal[1]["id"])
        check("처리 중이면 409", False)
    except urllib.error.HTTPError as e:
        check("처리 중이면 409", e.code == 409)
    finally:
        app.busy.release()

    # 대시보드
    items = json.loads(get(base + "/api/dashboard")[1])["items"]
    check("대시보드에 모두 저장(최신순)", len(items) == 14 and items[0]["id"] > items[-1]["id"], str(len(items)))
    blob = json.dumps(items)
    check("대시보드에 파일명/경로/해시 없음", "sha256" not in blob and "gallery_imgs" not in blob and ".png" not in blob)
    check("기록에 trace 포함", all(isinstance(i["trace"], list) and i["trace"] for i in items))

    # 서버 재시작 후에도 유지(SQLite)
    srv.shutdown()
    app2, srv2, base2 = start(Settings(env))
    check("재시작 후 대시보드 유지", len(json.loads(get(base2 + "/api/dashboard")[1])["items"]) == 14)
    srv2.shutdown()

    # 토큰 인증
    env_t = dict(env, AGENT_TOKEN="secret-token", AGENT_DATA_DIR=os.path.join(tmp, "db2"))
    app3, srv3, base3 = start(Settings(env_t))
    try:
        get(base3 + "/api/dashboard")
        check("토큰 없으면 401", False)
    except urllib.error.HTTPError as e:
        check("토큰 없으면 401", e.code == 401)
    check("토큰 있으면 200", get(base3 + "/api/dashboard", token="secret-token")[0] == 200)
    check("health 는 인증 없이", get(base3 + "/api/health")[0] == 200)
    srv3.shutdown()

    # 로컬이 아닌 주소 + 토큰 없음 -> 시작 거부
    try:
        Settings(dict(env, AGENT_HOST="0.0.0.0")).validate()
        check("외부 주소는 토큰 없이 시작 거부", False)
    except SystemExit:
        check("외부 주소는 토큰 없이 시작 거부", True)

    # 분리 위반 시 시작 거부
    bad = os.path.join(tmp, "art")
    d = json.load(open(os.path.join(bad, "separation.json")))
    d["calib"].append(d["gallery"][0])
    json.dump(d, open(os.path.join(bad, "separation.json"), "w"))
    try:
        mock_backends.build_mock_app(Settings(dict(env, AGENT_DATA_DIR=os.path.join(tmp, "db3"))))
        check("분리 위반이면 시작 거부", False)
    except RuntimeError as e:
        check("분리 위반이면 시작 거부", "분리 위반" in str(e))

    # ---- 점검용 검사는 대시보드에서 제외 / 정리 / 이전 DB 마이그레이션 ----
    def inspect_with(base_, gid, test):
        r = urllib.request.Request("{}/api/inspect/gallery/{}".format(base_, gid), method="POST")
        if test:
            r.add_header("X-E2E-Test", "1")
        urllib.request.urlopen(r, timeout=60).read()

    mock_artifacts.make(os.path.join(tmp, "art"))        # 앞의 '분리 위반' 테스트가 망가뜨린 기준값을 다시 깨끗하게 만든다 (같은 내용이라 id 도 같음)
    tmp3 = tempfile.mkdtemp()
    env3 = dict(env, AGENT_DATA_DIR=os.path.join(tmp3, "db"))
    app4, srv4, base4 = start(Settings(env3))
    inspect_with(base4, gal[0]["id"], True)
    inspect_with(base4, gal[1]["id"], True)
    inspect_with(base4, gal[2]["id"], False)
    default_items = json.loads(get(base4 + "/api/dashboard")[1])["items"]
    all_items = json.loads(get(base4 + "/api/dashboard?include_test=1")[1])["items"]
    check("X-E2E-Test 검사는 기본 대시보드에서 제외(실제 1건만 보임)", len(default_items) == 1 and len(all_items) == 3, "{}건 / 전체 {}건".format(len(default_items), len(all_items)))
    c = app4.store.counts()
    check("counts: 실제 1 / 테스트 2", (c["real"], c["test"]) == (1, 2), str(c))
    bak = app4.store.backup()
    check("삭제 전 백업 파일 생성", os.path.isfile(bak) and os.path.getsize(bak) > 0, os.path.basename(bak))
    check("purge(test_only): 테스트 2건만 삭제하고 실제 1건은 유지", app4.store.purge(test_only=True) == 2 and app4.store.counts()["real"] == 1)
    srv4.shutdown()
    # 이전 버전 DB(is_test 열 없음): 기록은 그대로 보이고 열만 추가된다
    import sqlite3
    old_dir = os.path.join(tmp3, "old")
    os.makedirs(old_dir)
    con = sqlite3.connect(os.path.join(old_dir, "dashboard.sqlite3"))
    con.execute("""CREATE TABLE inspections (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, source TEXT NOT NULL, sha256 TEXT NOT NULL,
        sep_flag TEXT, mmr_score REAL, rscore REAL, zone TEXT, decision TEXT, defect_type TEXT, why TEXT, tools TEXT, trace TEXT, type_probs TEXT,
        timings TEXT, map_b64 TEXT, map_shape TEXT, image_size TEXT)""")
    con.execute("INSERT INTO inspections (created_at, source, sha256, mmr_score, rscore, zone, decision, why, tools, trace, timings, map_b64, map_shape) "
                "VALUES ('2026-10-05 00:00:00','gallery','x',0.5,0.1,'amb','defect','old','[]','[]','{\"total_s\": 1.0}','AA','[1,1]')")
    con.commit()
    con.close()
    from store import Store
    migrated = Store(old_dir)
    items = migrated.list()
    check("이전 DB 마이그레이션: 기존 기록 보존(is_test=0)", len(items) == 1 and items[0]["is_test"] == 0 and items[0]["why"] == "old")

    # agent.py 오프라인 평가 방식(저장된 결과 조회 함수 주입)이 그대로 동작하는지
    import agent as A
    import numpy as np
    stored_whole = {"k1": {"normal": 0.5, "ablation": 0.2, "breakdown": 0.1, "fracture": 0.1, "groove": 0.1}}
    tools = A.ToolBox(map_of=lambda r: np.random.RandomState(0).rand(224, 224), whole_of=lambda r: stored_whole.get(r["domain"]),
                      second_of=lambda r: None, tau_lo=-0.5)
    row = {"domain": "k1", "path": "x", "score": 0.5, "rscore": 0.3, "zone10": "amb", "type": "t", "label": 1}
    seen = []
    res = A.run_agent(row, A.MockBrain(), tools, 0.98, on_event=lambda n, d: seen.append(n))
    check("오프라인 방식 ToolBox/run_agent 동작 + 이벤트 콜백", res["pred_defect"] == 1 and "thought" in seen and res["type_probs"] is not None, str(seen))

    print("\n{} / {} 통과".format(sum(ok), len(ok)))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
