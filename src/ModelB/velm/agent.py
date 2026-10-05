"""검사 에이전트 (ReAct 방식): 생각(Thought) -> 행동(Action: 도구 선택) -> 관찰(Observation) 을 반복해서 정상/불량을 정한다.

설계 원칙
  - 사람이 주는 것은 목표뿐: "불량을 정상으로 판정하지 않는다. 오탐은 근거가 확실할 때만 줄인다."
  - 어떤 도구를 쓸지, 몇 번 볼지, 언제 멈출지는 에이전트(Qwen)가 스스로 고른다.
  - 행동 선택은 글이 아니라 '도구 이름의 첫 토큰 확률'로 한다 (엉뚱한 글이 판정을 흔들지 않게). 생각(Thought)은 짧게 쓰게 하고, 기록으로 남긴다.
  - 안전장치(고정): 에이전트의 추가 확인은 기준선(라우터+Qwen 현재 최종) 판정을 '불량 쪽으로만' 바꿀 수 있다.
        -> 같은 t 로 비교하면 재현율이 기준선보다 낮아질 수 없다. 확신이 없거나 도구를 안 쓰면 불량.
도구
  read_map      : MMR 점수 / 라우터 점수 / 이상 맵 모양을 글로 요약 (GPU 불필요)
  ask_whole     : 전체 사진 + 참고 불량 12장으로 Qwen 5-class 확률 (v8, 저장된 결과가 있으면 그것을 사용)
  second_prompt : 다른 프롬프트(v7)로 같은 사진을 다시 판정 (저장된 결과가 있으면 사용)
  zoom_check    : 이상 맵이 가리킨 부위를 원본 해상도로 확대해서 '진짜 손상인가, 반사/그림자인가' 를 Qwen에게 물음 (GPU 필요)
  decide        : 지금까지의 증거로 확정
사용
  python agent.py --mmr-out ... --router --eval --limit 40            (GPU, Qwen 에이전트)
  python agent.py --mmr-out ... --router --eval --limit 40 --mock     (GPU 없이 흐름만 확인하는 규칙 기반 두뇌)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_eval as q  # noqa: E402

TOOLS = {
    "read_map": "MMR 점수, 라우터 점수, 이상 맵 모양(크기/가늘기/위치)을 요약해서 읽는다",
    "ask_whole": "전체 사진을 참고 불량 12장과 비교해 Qwen이 5개 중 무엇인지 확률로 답한다",
    "second_prompt": "다른 프롬프트로 같은 사진을 한 번 더 판정한다",
    "zoom_check": "이상 맵이 가리킨 부위를 원본 해상도로 확대해 진짜 손상인지 반사/그림자인지 확인한다",
    "decide": "지금까지의 증거로 최종 판정을 확정한다",
}
GOAL = ("Goal: never call a defective blade normal. Call a blade normal only when the evidence clearly supports it. "
        "Use as few checks as needed, but if you are unsure, check more.")


# ---------------------------------------------------------------- 판정 규칙 (고정 안전장치)
def final_decision(zone, ev, t):
    """증거 -> (불량 여부 1/0, 이유). 추가 확인은 기준선 판정을 불량 쪽으로만 바꾼다."""
    if zone == "confident":
        return 1, "MMR/라우터가 확실한 불량"
    pn = ev.get("pn_whole")
    if zone == "clear_normal":
        if pn is None:
            return 0, "정상 확정 구간이고 더 확인하지 않음 (기준선과 같음)"
        if pn < t:
            return 1, "정상 확정 구간이지만 Qwen 정상 확률 {:.3f} < t={}".format(pn, t)
        return 0, "Qwen도 정상 (확률 {:.3f})".format(pn)
    # amb
    if pn is None:
        return 1, "전체 사진 판정을 하지 않아 정상으로 인정할 수 없음 (불량 우선)"
    if pn < t:
        return 1, "Qwen 정상 확률 {:.3f} < t={}".format(pn, t)
    if ev.get("zoom_defect") is not None and ev["zoom_defect"] >= 0.5:
        return 1, "확대해 보니 손상 확률 {:.3f}".format(ev["zoom_defect"])
    if ev.get("pn_v7") is not None and ev["pn_v7"] < 0.5:
        return 1, "두 번째 프롬프트가 불량 쪽 (정상 확률 {:.3f})".format(ev["pn_v7"])
    return 0, "Qwen 정상 확률 {:.3f} >= t 이고 추가 확인에서도 반대 근거 없음".format(pn)


def defect_type(ev):
    pr = ev.get("probs_whole")
    if not pr:
        return "unknown"
    return max(q.DEF4, key=lambda k: pr.get(k, 0.0))


# ---------------------------------------------------------------- 도구
class ToolBox:
    def __init__(self, p1, p2, maps=None, tau_lo=None, model=None, processor=None, ref_clf=None, zoom_fn=None):
        self.p1, self.p2, self.maps, self.tau_lo = p1, p2, maps, tau_lo
        self.model, self.processor, self.ref_clf, self.zoom_fn = model, processor, ref_clf, zoom_fn

    def available(self, ev, used):
        """아직 안 쓴 도구만 + 필요한 자원이 있는 도구만."""
        names = ["read_map", "ask_whole", "second_prompt", "zoom_check", "decide"]
        out = []
        for n in names:
            if n in used and n != "decide":
                continue
            if n == "zoom_check" and self.zoom_fn is None:
                continue
            if n == "second_prompt" and self.p2 is None:
                continue
            out.append(n)
        return out

    def run(self, name, r, ev):
        k = q.key(r)
        if name == "read_map":
            txt = "MMR 점수 {:.3f}".format(r["score"])
            if "rscore" in r:
                txt += ", 라우터 점수 {:.2f} (정상 확정 기준선 {:.2f}보다 {})".format(
                    r["rscore"], self.tau_lo, "낮음 -> 정상 확정 구간" if r["zone10"] == "clear_normal" else "높음")
                ev["rscore"] = r["rscore"]
            txt += ", 구간 {}".format({"clear_normal": "정상 확정", "amb": "애매", "confident": "확실한 불량"}[r["zone10"]])
            txt += ". 안내: " + {
                "confident": "이미 불량으로 확정된 구간이라 정상으로 바뀌지 않음, 불량 종류가 필요하면 ask_whole",
                "clear_normal": "더 확인하지 않으면 정상으로 남음, 정상 확정이 틀렸는지 보려면 ask_whole",
                "amb": "정상으로 인정하려면 ask_whole 결과가 반드시 필요함"}[r["zone10"]]
            if self.maps is not None:
                import region_crop as rc
                f = rc.map_features(self.maps.get(q.resolve(r["path"])))
                txt += "; 이상 부위 면적비율 {:.3f}, 덩어리 {}개, 가늘기 {:.1f}, 가장자리까지 거리 {:.2f}".format(f[5], int(f[7]), f[8], f[12])
            return txt
        if name == "ask_whole":
            pr = self.p1.get(k)
            if pr is None and self.ref_clf is not None:
                pr = self.ref_clf.classify(q.resolve(r["path"]))["probs"]
            if pr is None:
                return "결과 없음 (이 사진은 전체 판정이 저장돼 있지 않음)"
            ev["pn_whole"], ev["probs_whole"] = pr.get("normal", 0.0), pr
            top = max(pr, key=pr.get)
            return "ask_whole 결과 (Qwen이 참고 불량 사진 12장과 비교한 판정, 이상 탐지기 아님): 정상 확률 {:.3f}, 가장 높은 항목 {} ({:.3f})".format(pr.get("normal", 0.0), top, pr[top])
        if name == "second_prompt":
            pr = self.p2.get(k) if self.p2 is not None else None
            if pr is None:
                return "결과 없음 (이 사진은 두 번째 프롬프트 결과가 저장돼 있지 않음)"
            ev["pn_v7"] = pr.get("normal", 0.0)
            return "두 번째 프롬프트 결과: 정상 확률 {:.3f}".format(ev["pn_v7"])
        if name == "zoom_check":
            d = self.zoom_fn(r)
            ev["zoom_defect"] = d
            return "확대 확인: 실제 손상일 확률 {:.2f}".format(d)
        return "확정"


# ---------------------------------------------------------------- 두뇌
class MockBrain:
    """GPU 없이 흐름만 확인하는 규칙 기반 두뇌 (실제 에이전트가 아님). 생각 글도 규칙으로 만든다."""

    def step(self, r, log, ev, options):
        if "read_map" in options and "read_map" not in [a for a, _ in log]:
            return "먼저 지도 정보를 읽어 보자.", "read_map"
        if "ask_whole" in options and ev.get("pn_whole") is None:
            return "전체 사진을 Qwen에게 물어보자.", "ask_whole"
        pn = ev.get("pn_whole")
        if pn is not None and pn < 0.995 and "zoom_check" in options:
            return "정상 확률이 애매하니 확대해서 확인하자.", "zoom_check"
        return "증거가 충분하다.", "decide"


TOOL_HELP = {
    "read_map": "Read a summary of the anomaly detector output for this photo (scores, zone, anomaly shape).",
    "ask_whole": "A Qwen vision-language model compares the whole photo with 12 labeled defect reference photos and returns P(normal) and the likely defect type "
                 "(this is NOT the anomaly detector). A blade can only be judged NORMAL if this check was done.",
    "second_prompt": "Ask the same photo again with a different prompt for a second opinion. Optional; it can only raise suspicion, never clear a blade.",
    "decide": "Finish and give the final verdict. Use it when you have enough evidence; extra checks cost time.",
}


class QwenBrain:
    """Qwen2-VL 에이전트: 짧은 생각을 쓰고, 도구 이름의 첫 토큰 확률로 행동을 고른다. 선택 확률은 self.last_probs 에 남는다."""

    def __init__(self, model, processor, t=0.98):
        import torch
        self.torch, self.model, self.processor, self.t = torch, model, processor, t
        self.last_probs = None
        self.device = next(p.device for p in model.parameters() if p.device.type != "meta")
        tok = processor.tokenizer
        self.ids = {}
        for n in TOOLS:
            self.ids[n] = tok.encode(n, add_special_tokens=False)[0]
        if len(set(self.ids.values())) != len(self.ids):
            raise RuntimeError("도구 이름의 첫 토큰이 겹칩니다: {}".format(self.ids))

    def _prompt(self, r, log, ev, options, thought=None):
        tools = "\n".join("- {}: {}".format(n, TOOL_HELP.get(n, TOOLS[n])) for n in options)
        rules = ("Inspection rules:\n"
                 "- A blade is judged NORMAL only if the ask_whole check gives P(normal) >= {t} and no other check disagrees. Otherwise it is judged DEFECTIVE.\n"
                 "- Without ask_whole the blade cannot be judged normal.\n"
                 "- Extra checks cost time, so stop with decide once you have enough evidence.").format(t=self.t)
        hist = "\n".join("Action: {}\nObservation: {}".format(a, o) for a, o in log) or "(no checks yet)"
        user = ("You are an aircraft engine blade inspection agent.\n{}\n\n{}\n\nAvailable tools:\n{}\n\nSo far:\n{}\n\n"
                "Write one short sentence of reasoning about what to do next, then choose the tool.\nThought:").format(GOAL, rules, tools, hist)
        msgs = [{"role": "user", "content": [{"type": "text", "text": user}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + ("" if thought is None else " {}\nAction:".format(thought))

    def step(self, r, log, ev, options):
        torch, tok = self.torch, self.processor.tokenizer
        text = self._prompt(r, log, ev, options)
        inp = tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=60, do_sample=False, stop_strings=["\n"], tokenizer=tok)
        thought = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        inp2 = tok(self._prompt(r, log, ev, options, thought), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inp2).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        sel = torch.softmax(torch.stack([lp[self.ids[n]] for n in options]), 0)
        self.last_probs = {n: round(float(p), 3) for n, p in zip(options, sel)}
        best = max(options, key=lambda n: self.last_probs[n])
        return thought, best


# ---------------------------------------------------------------- 루프
def run_agent(r, brain, tools, t, max_steps=5, auto_map=False, auto_whole=False, skip_confident=False):
    """한 사진에 대해 생각 -> 행동 -> 관찰 루프를 돈다.
    보조 옵션(컨트롤러가 대신 해 주는 것, 기록에 auto=True 로 표시):
      auto_map       : 시작할 때 read_map 을 자동 실행 (에이전트는 첫 관찰을 받은 상태에서 시작)
      skip_confident : 확실한 불량 구간은 에이전트가 고민하지 않고 종류 확인(ask_whole)만 자동 실행 후 확정
      auto_whole     : 에이전트가 ask_whole 없이 decide 하려 하면 컨트롤러가 먼저 ask_whole 을 실행 (확인 누락 보완)"""
    ev, log, trace, used = {}, [], [], set()

    def auto(action, thought):
        obs = tools.run(action, r, ev)
        used.add(action)
        log.append((action, obs))
        trace.append({"thought": thought, "action": action, "observation": obs, "auto": True})

    if auto_map or (skip_confident and r["zone10"] == "confident"):
        auto("read_map", "(자동) 시작할 때 이상 맵 요약을 읽는다")
    if skip_confident and r["zone10"] == "confident":
        auto("ask_whole", "(자동) 확실한 불량 구간이라 고민 없이 종류만 확인한다")
        trace.append({"thought": "(자동) 확실한 불량이라 추가 확인 없이 확정", "action": "decide", "observation": "확정", "auto": True})
    else:
        for _ in range(max_steps):
            options = tools.available(ev, used)
            thought, action = brain.step(r, log, ev, options)
            cp = getattr(brain, "last_probs", None)
            if action not in options:
                action = "decide"
            if action == "decide":
                if auto_whole and "ask_whole" not in used:
                    auto("ask_whole", "(자동 보완) 전체 사진 판정이 빠져 있어 먼저 실행한다")
                trace.append({"thought": thought, "action": "decide", "observation": "확정", "choice_probs": cp})
                break
            obs = tools.run(action, r, ev)
            used.add(action)
            log.append((action, obs))
            trace.append({"thought": thought, "action": action, "observation": obs, "choice_probs": cp})
    dec, why = final_decision(r["zone10"], ev, t)
    return {"key": q.key(r), "zone": r["zone10"], "true": r["type"], "label": r["label"], "pred_defect": dec,
            "type": defect_type(ev) if dec else "good", "why": why, "evidence": {k: v for k, v in ev.items() if k != "probs_whole"},
            "trace": trace}


def baseline_pred(r, p1, t):
    """기준선(라우터+Qwen 현재 최종)의 판정: 정상 확정 구간 -> 정상, 확실한 불량 -> 불량, 애매 -> pn>=t 면 정상."""
    if r["zone10"] == "clear_normal":
        return 0
    if r["zone10"] == "confident":
        return 1
    pr = p1.get(q.key(r))
    return 0 if (pr is not None and pr.get("normal", 0.0) >= t) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr-out", default=os.path.join(q.MMR_DIR, "log_MMR_AeBAD_S_54"))
    p.add_argument("--holdout", default=q.HOLDOUT)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--router", action="store_true")
    p.add_argument("--recall-target", type=float, default=0.98)
    p.add_argument("--normal-recall", type=float, default=0.97)
    p.add_argument("--fpr-hi", type=float, default=0.02)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--limit", type=int, default=40, help="에이전트를 돌릴 보고용 사진 수 (정해진 무작위 순서)")
    p.add_argument("--mock", action="store_true", help="GPU 없이 규칙 기반 두뇌로 흐름만 확인")
    p.add_argument("--assist", action="store_true", help="보조 옵션 전부 켜기: --auto-map --auto-whole --skip-confident")
    p.add_argument("--auto-map", action="store_true", help="시작할 때 read_map 자동 실행")
    p.add_argument("--auto-whole", action="store_true", help="에이전트가 ask_whole 없이 decide 하면 컨트롤러가 대신 실행")
    p.add_argument("--skip-confident", action="store_true", help="확실한 불량 구간은 고민 없이 종류 확인만 하고 확정")
    p.add_argument("--resume", action="store_true", help="이미 처리한 사진은 건너뛰고 이어서 (기록 파일에 덧붙임)")
    p.add_argument("--trace-name", default="agent_trace", help="results/ 아래 판단 기록 파일 이름(확장자 제외)")
    p.add_argument("--no-zoom", action="store_true", help="zoom_check 도구를 쓰지 않음 (GPU 없이도 실제 저장 결과로 에이전트 흐름 확인 가능)")
    a = p.parse_args()
    if a.assist:
        a.auto_map = a.auto_whole = a.skip_confident = True

    import hashlib
    import region_crop as rc
    rows = q.apply_holdout(q.load_mmr_csv(a.mmr_out), a.holdout)
    calib = [r for r in rows if q.split_of(r) == "calib"]
    report = [r for r in rows if q.split_of(r) == "report"]
    field = "score"
    if a.router:
        q.attach_router_scores(rows, a.mmr_out)
        field = "rscore"
    lo, hi = q.calibrate_v10(calib, a.recall_target, a.fpr_hi, field)
    q.mark_zones_v10(rows, lo, hi, field)
    q.TAG = "v8_n12"
    p1 = q.load_probs(a.out)
    q.TAG = "v7"
    p2 = q.load_probs(a.out)
    t = q.choose_t_by_recall(calib, p1, None, a.normal_recall)
    maps = rc.MapIndex(a.mmr_out)
    print("기준 t={} | 구간 선 {:.3f}/{:.3f} | 보고용 {}장".format(t, lo, hi, len(report)))

    zoom_fn = None
    model = processor = None
    if not a.no_zoom and not a.mock:
        print("zoom_check 도구는 아직 GPU 구현 전입니다 (--no-zoom 으로 실행하세요)")
        a.no_zoom = True
    tools = ToolBox(p1, p2, maps, lo, zoom_fn=zoom_fn)
    if a.mock:
        brain = MockBrain()
    else:
        from run_qwen import load_model
        model, processor = load_model()
        brain = QwenBrain(model, processor, t=t)

    if not a.eval:
        return
    order = sorted(report, key=lambda r: hashlib.md5(("agent" + q.key(r)).encode()).hexdigest())[:a.limit]
    results = []
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "{}{}.jsonl".format(a.trace_name, "_mock" if a.mock else ""))
    done = {}
    if a.resume and os.path.exists(fp):
        for l in open(fp):
            if l.strip():
                x = json.loads(l)
                done[x["key"]] = x
        print("이어서 하기: 이미 처리한 사진 {}장".format(len(done)))
    with open(fp, "a" if a.resume else "w") as f:
        for i, r in enumerate(order, 1):
            k = q.key(r)
            if k in done:
                results.append(done[k])
                continue
            res = run_agent(r, brain, tools, t, auto_map=a.auto_map, auto_whole=a.auto_whole, skip_confident=a.skip_confident)
            res["baseline_defect"] = baseline_pred(r, p1, t)
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print("[{}/{}] {} 구간 {} -> {} ({}) | 도구 {}".format(
                i, len(order), os.path.basename(r["path"]), r["zone10"], "불량" if res["pred_defect"] else "정상", res["why"],
                " > ".join(s["action"] + ("*" if s.get("auto") else "") for s in res["trace"])), flush=True)
    y = np.array([x["label"] for x in results])
    ag = np.array([x["pred_defect"] for x in results])
    bs = np.array([x["baseline_defect"] for x in results])
    ma, mb = q.metrics(y, ag), q.metrics(y, bs)
    print("\n[에이전트가 본 {}장] 기준선: FN {} FP {} | 에이전트: FN {} FP {}".format(len(results), mb["FN"], mb["FP"], ma["FN"], ma["FP"]))
    flipped = int((ag != bs).sum())
    toward_normal = int(((ag == 0) & (bs == 1)).sum())
    print("판정이 바뀐 사진 {}장 (불량->정상으로 바뀐 사진 {}장: 안전장치상 0이어야 정상)".format(flipped, toward_normal))
    print("trace 저장: {}".format(fp))


if __name__ == "__main__":
    main()
