"""실제 Qwen 두뇌/도구 구성 (GPU, velm_qwen 환경). 기존 velm 코드를 그대로 import 해서 쓴다.

- 모델: run_qwen.load_model()  (Qwen2-VL-7B, 8-bit)
- ask_whole: CachedRefClassifier(참고 이미지 부분 캐시) -> 참고 이미지는 manifest 의 이미지(데모 이미지와 겹치지 않음)
- second_prompt: classify_image(prompt="v7")
- 두뇌: QwenBrain(t = params.t)
"""
import csv
import os
import random
import re

import agent as A
from online_agent import OnlineAgent


def build_refs(manifest, data_root):
    """qwen_eval.build_refs 와 같은 구성(그룹별 정렬 + 고정 시드 섞기)을 --data-root 기준 경로로 재현한다."""
    groups = {}
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            path = os.path.join(data_root, r["original_path"].split("AeBAD/", 1)[1])
            groups.setdefault((r["domain"], r["defect"]), []).append({"path": path, "label": r["defect"], "condition": r["domain"]})
    refs = []
    for k in sorted(groups):
        refs += sorted(groups[k], key=lambda x: x["path"])[:5]
    missing = [r["path"] for r in refs if not os.path.isfile(r["path"])]
    if missing:
        raise SystemExit("참고 이미지를 찾을 수 없습니다: {}".format(missing[0]))
    random.Random(0).shuffle(refs)
    return refs


class KoreanThoughtBrain:
    """QwenBrain 을 감싸 화면에 보여 줄 '생각'만 한국어로 옮긴다.
    도구 선택(첫 토큰 확률)은 원래 영어 생각으로 이미 끝난 뒤라 판정에는 영향이 없다. 번역이 실패하면 영어 원문을 그대로 쓴다."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):          # last_probs 등은 안쪽 두뇌 것을 그대로 쓴다
        return getattr(self.inner, name)

    def step(self, r, log, ev, options):
        thought, action = self.inner.step(r, log, ev, options)
        return self.to_korean(thought), action

    def to_korean(self, text):
        if not text or re.search(r"[가-힣]", text):
            return text
        try:
            torch, proc, model = self.inner.torch, self.inner.processor, self.inner.model
            msgs = [{"role": "user", "content": [{"type": "text", "text":
                     "다음 영어 문장을 자연스러운 한국어 한 문장으로 번역하세요. 번역문만 출력하고 설명은 쓰지 마세요. "
                     "도구 이름(read_map, ask_whole, second_prompt, decide)은 번역하지 말고 그대로 두세요.\n\n" + text}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc.tokenizer(prompt, return_tensors="pt").to(self.inner.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=120, do_sample=False)
            ko = proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            return ko if re.search(r"[가-힣]", ko) else text
        except Exception as e:             # 번역 실패가 검사를 막으면 안 된다
            print("[경고] 생각 번역 실패, 영어 원문 사용: {}".format(e), flush=True)
            return text


def load_qwen(settings, params):
    from run_qwen import CachedRefClassifier, classify_image, load_model
    model, processor = load_model()
    refs = build_refs(settings.refs_manifest, settings.data_root)
    print("참고 이미지 {}장 (ask_whole)".format(len(refs)), flush=True)
    ref_clf = CachedRefClassifier(model, processor, refs, settings.ref_px)
    brain = KoreanThoughtBrain(A.QwenBrain(model, processor, t=params.t))
    whole_fn = lambda path: ref_clf.classify(path)["probs"]
    v7_fn = lambda path: classify_image(path, model, processor, prompt="v7")["probs"]
    return OnlineAgent(params, brain, whole_fn, v7_fn)
