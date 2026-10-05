"""온라인(실시간) 입력으로 에이전트(ModelB/velm/agent.py)를 실행한다.

agent.ToolBox 는 증거의 출처를 함수로 주입받는다. 여기서는 저장된 결과 대신 즉석 계산 함수를 넣는다.
  이상 맵            -> 방금 MMR 서비스가 낸 맵
  ask_whole (v8)     -> whole_fn(이미지 경로): Qwen 이 참고 이미지와 비교해 5-class 확률을 즉석 계산
  second_prompt (v7) -> second_fn(이미지 경로): v7 프롬프트로 즉석 판정 (None 이면 이 도구는 제공하지 않음)
진행 이벤트는 run_agent(on_event=emit) 로 받는다.
"""
import agent as A      # ModelB/velm/agent.py


class OnlineAgent:
    def __init__(self, params, brain, whole_fn, second_fn=None):
        self.params, self.brain, self.whole_fn, self.second_fn = params, brain, whole_fn, second_fn

    def inspect(self, image_path, mmr, emit, max_steps=5):
        """mmr: {"score", "map"} (MMR 서비스 응답). 최종 결과 dict 를 돌려주고, 진행은 emit(이벤트, dict) 로 보낸다."""
        p = self.params
        rs = p.rscore(mmr["map"])
        zone = p.zone(rs)
        r = {"domain": "online", "path": image_path, "score": float(mmr["score"]), "rscore": rs, "zone10": zone,
             "type": "unknown", "label": -1}          # 온라인에는 정답이 없다
        emit("mmr", {"score": r["score"], "rscore": rs, "zone": zone})
        amap = mmr["map"]
        tools = A.ToolBox(map_of=lambda row: amap,
                          whole_of=lambda row: self.whole_fn(row["path"]),
                          second_of=(lambda row: self.second_fn(row["path"])) if self.second_fn else None,
                          tau_lo=p.tau_lo)
        res = A.run_agent(r, self.brain, tools, p.t, max_steps=max_steps, on_event=emit)
        return {"decision": "defect" if res["pred_defect"] else "normal",
                "defect_type": res["type"] if res["pred_defect"] else None,
                "why": res["why"], "zone": zone, "mmr_score": r["score"], "rscore": rs,
                "tools": [s["action"] for s in res["trace"]], "trace": res["trace"],
                "evidence": res["evidence"], "type_probs": res["type_probs"]}
