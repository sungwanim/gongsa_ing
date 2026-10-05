"""기존 에이전트(ModelB/velm/agent.py)를 수정하지 않고 '온라인' 입력으로 돌리는 어댑터.

run_agent() 는 도구가 같은 인터페이스의 객체를 받으면 그대로 동작한다.
  저장된 이상 맵      -> SingleMap(방금 받은 맵 1장)
  저장된 Qwen v8 확률 -> p1 을 비워 두면 ToolBox 가 ref_clf(CachedRefClassifier)로 Qwen 을 직접 호출
  저장된 Qwen v7 확률 -> LazyV7 (v7 프롬프트로 즉석 판정)
  두뇌               -> QwenBrain (모의 모드는 MockBrain)
실시간 진행 표시는 두뇌/도구를 감싸는 래퍼가 emit(event, data) 로 내보낸다.
"""
import time

import agent as A      # ModelB/velm/agent.py (그대로 사용)


class SingleMap:
    def __init__(self, m):
        self.m = m

    def get(self, path):
        return self.m


class LazyV7:
    """second_prompt 도구용: 현재 이미지에 대해 v7 프롬프트로 Qwen 을 즉석 호출."""

    def __init__(self, fn):
        self.fn, self.path = fn, None

    def get(self, key):
        return self.fn(self.path) if self.path else None


class OnlineToolBox(A.ToolBox):
    def __init__(self, emit, p2, maps, tau_lo, ref_clf):
        super().__init__({}, p2, maps, tau_lo, ref_clf=ref_clf, zoom_fn=None)
        self.emit, self.probs_whole = emit, None

    def run(self, name, r, ev):
        self.emit("tool_start", {"tool": name})
        t0 = time.time()
        if self.p2 is not None:
            self.p2.path = r["path"]
        obs = super().run(name, r, ev)
        if name == "ask_whole":
            self.probs_whole = ev.get("probs_whole")
        self.emit("observation", {"tool": name, "text": obs, "sec": round(time.time() - t0, 2)})
        return obs


class EmittingBrain:
    def __init__(self, inner, emit):
        self.inner, self.emit = inner, emit

    @property
    def last_probs(self):
        return getattr(self.inner, "last_probs", None)

    def step(self, r, log, ev, options):
        t0 = time.time()
        thought, action = self.inner.step(r, log, ev, options)
        self.emit("thought", {"text": thought, "options": list(options)})
        self.emit("action", {"tool": action, "choice_probs": self.last_probs, "sec": round(time.time() - t0, 2)})
        return thought, action


class OnlineAgent:
    def __init__(self, params, brain, ref_clf, v7_fn):
        self.params, self.brain, self.ref_clf, self.v7_fn = params, brain, ref_clf, v7_fn

    def inspect(self, image_path, mmr, emit, max_steps=5):
        """mmr: {"score", "map"} (MMR 서비스 응답). 반환: 최종 결과 dict (emit 으로 진행 이벤트도 보냄)."""
        rs = self.params.rscore(mmr["map"])
        zone = self.params.zone(rs)
        r = {"domain": "online", "path": image_path, "score": float(mmr["score"]), "rscore": rs, "zone10": zone,
             "type": "unknown", "label": -1}
        emit("mmr", {"score": r["score"], "rscore": rs, "zone": zone})
        tools = OnlineToolBox(emit, LazyV7(self.v7_fn) if self.v7_fn else None, SingleMap(mmr["map"]), self.params.tau_lo, self.ref_clf)
        res = A.run_agent(r, EmittingBrain(self.brain, emit), tools, self.params.t, max_steps=max_steps)
        return {"decision": "defect" if res["pred_defect"] else "normal",
                "defect_type": res["type"] if res["pred_defect"] else None,
                "why": res["why"], "zone": zone, "mmr_score": r["score"], "rscore": rs,
                "tools": [s["action"] for s in res["trace"]], "trace": res["trace"],
                "evidence": res["evidence"], "type_probs": tools.probs_whole}
