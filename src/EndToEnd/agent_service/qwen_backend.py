"""실제 Qwen 두뇌/도구 구성 (GPU, velm_qwen 환경). 기존 velm 코드를 그대로 import 해서 쓴다.

- 모델: run_qwen.load_model()  (Qwen2-VL-7B, 8-bit)
- ask_whole: CachedRefClassifier(참고 이미지 부분 캐시) -> 참고 이미지는 manifest 의 이미지(데모 이미지와 겹치지 않음)
- second_prompt: classify_image(prompt="v7")
- 두뇌: QwenBrain(t = params.t)
"""
import csv
import os
import random

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


def load_qwen(settings, params):
    from run_qwen import CachedRefClassifier, classify_image, load_model
    model, processor = load_model()
    refs = build_refs(settings.refs_manifest, settings.data_root)
    print("참고 이미지 {}장 (ask_whole)".format(len(refs)), flush=True)
    ref_clf = CachedRefClassifier(model, processor, refs, settings.ref_px)
    brain = A.QwenBrain(model, processor, t=params.t)
    v7_fn = lambda path: classify_image(path, model, processor, prompt="v7")["probs"]
    return OnlineAgent(params, brain, ref_clf, v7_fn)
