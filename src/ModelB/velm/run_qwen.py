import argparse
import os

import torch

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

DEFECT_CLASSES = [
    "ablation",
    "breakdown",
    "fracture",
    "groove",
]

# MMR(1차)이 애매하게 불량으로 넘긴 이미지도 다시 검증하므로 "normal"(정상)도 선택지에 포함
CLASSES = ["normal"] + DEFECT_CLASSES


# ============================================================
# Load Qwen2-VL
# ============================================================

def load_model():
    print("Qwen2-VL 모델 로딩 중...")

    # 이미지 해상도가 너무 크면 GPU 메모리 사용량이 증가하므로 제한
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # 8-bit quantization
    # 16GB MIG GPU에서도 Qwen2-VL-7B를 실행할 수 있도록
    # 모델 weight 메모리 사용량을 줄임
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("모델 로딩 완료 (8-bit)")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU 메모리 사용량:",
            round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
            "GB",
        )

    return model, processor


# ============================================================
# Prompts (이름으로 선택: classify_image(..., prompt="v6"))
#  v6 : 원본 짧은 프롬프트 + 선택지에 normal 추가 (기존 결과와 동일한 문구)
#  v7 : 타입별 특징과 판단 순서를 구체적으로 적은 프롬프트 (정상 쏠림을 줄이기 위한 개선안)
#       특징은 MMR 저장소의 AeBAD-S 샘플 그림(assets/image/dataset_s.jpg)을 보고 작성:
#       breakdown=동그란 구멍/점, ablation=검게 탄 자국, fracture=일부가 잘려 나감, groove=가는 틈/작은 홈
# 확신도는 모델 확률로 계산 (classify_image 참고)
# ============================================================

PROMPTS = {}

PROMPTS["v6"] = f"""
You are inspecting an aircraft engine blade.

The blade in the image may be normal or may have one defect.

Classify the image into exactly ONE of the following five classes:

{chr(10).join(f"- {name}" for name in CLASSES)}

Rules:
1. Select exactly one class.
2. Return only the class name.
3. Do not provide an explanation.
4. Do not return any class other than the five listed above.
"""

PROMPTS["v7"] = """
You are inspecting an aircraft engine blade.

This image was flagged as suspicious by an automatic anomaly detector. Most flagged images contain a real defect, but some are false alarms caused only by lighting, reflection, shadow, background or viewpoint changes.

Classify the image into exactly ONE of the following five classes:

- normal: the blade is intact. The surface is smooth and the outline is continuous, with no holes, burn marks, cuts or missing pieces. Differences in brightness, color, reflection, shadow, background, position, size or viewing angle alone are NOT defects.
- ablation: dark, black or scorched burn marks, soot-like discoloration or burnt patches on the blade surface. The surface looks burnt or eroded by heat, while the blade outline is mostly intact.
- breakdown: small round holes, pits or punctures in the blade surface, either a single hole or several small holes or dots, like perforations caused by impact.
- fracture: a piece of the blade is broken off or cut away. The blade end, tip or edge is missing, so the overall outline is clearly changed and a flat or jagged broken edge is visible.
- groove: a thin narrow slit, crack line or tiny notch cut into the blade edge or surface. It is very small, and the rest of the blade looks intact.

How to decide:
1. Look at the whole blade outline. Is a piece missing or cut off? If yes, choose fracture.
2. Look at the surface for dark burn marks or scorched patches. If yes, choose ablation.
3. Look for small round holes or dots in the surface. If yes, choose breakdown.
4. Look at the edges and surface for a thin slit, crack line or tiny notch. If yes, choose groove.
5. Do not answer normal just because the damage is small or subtle. Choose normal only if none of the above is present.

Rules:
1. Select exactly one class.
2. Return only the class name.
3. Do not provide an explanation.
4. Do not return any class other than the five listed above.
"""


# v8 : 라벨이 붙은 참고 이미지(결함 60장: 촬영 조건 + 결함 타입)를 먼저 보여주고 판정시키는 방식
#      (참고 이미지는 평가에서 제외된 holdout 60장. classify_image(..., refs=[...]) 로 전달)
V8_HEAD = """You are inspecting an aircraft engine blade.

First you will see labeled reference examples of DEFECTIVE blades. Each example states how the photo was taken (different background, different lighting, or different camera view) and which defect it shows.
The same defect type can look different under different conditions and on different blades, so learn what each defect looks like instead of matching exact pixels.
"""

V8_TAIL = """Now classify the inspection image that follows into exactly ONE of the following five classes:

{class_list}

normal means the blade has none of the defects shown in the reference examples. Differences in brightness, color, reflection, shadow or background alone are not defects.

Rules:
1. Select exactly one class.
2. Return only the class name.
3. Do not provide an explanation.
4. Do not return any class other than the five listed above.
"""

CONDITION_TEXT = {
    "background": "different background",
    "illumination": "different lighting",
    "view": "different camera view",
}


def build_ref_content(refs, ref_max_pixels, images=None):
    """refs: [{"path", "label", "condition"}]  ->  메시지 content 조각 리스트"""
    content = [{"type": "text", "text": V8_HEAD}]
    for i, r in enumerate(refs, 1):
        content.append({
            "type": "text",
            "text": "Reference example {}: {}, defect = {}".format(
                i, CONDITION_TEXT.get(r["condition"], r["condition"]), r["label"]),
        })
        if images is not None:
            item = {"type": "image", "image": images[i - 1]}     # 미리 줄여 둔 PIL 이미지 (매번 디스크에서 읽지 않음)
        else:
            item = {"type": "image", "image": r["path"]}
            if ref_max_pixels:
                item["max_pixels"] = ref_max_pixels      # 예시 이미지는 작게 (토큰/메모리 절약)
        content.append(item)
    content.append({"type": "text", "text": V8_TAIL.format(class_list="\n".join("- " + c for c in CLASSES))})
    content.append({"type": "text", "text": "Inspection image:"})
    return content


# ============================================================
# Defect classification (label + confidence)
# ============================================================

_LABEL_TOKEN_IDS = None


def _label_first_token_ids(processor):
    """각 클래스 이름의 첫 토큰 id (소문자/첫글자 대문자 두 가지 표기). 클래스끼리 겹치면 오류."""
    global _LABEL_TOKEN_IDS
    if _LABEL_TOKEN_IDS is not None:
        return _LABEL_TOKEN_IDS

    tok = processor.tokenizer
    ids = {}
    for name in CLASSES:
        found = set()
        for variant in (name, name.capitalize()):
            toks = tok.encode(variant, add_special_tokens=False)
            if toks:
                found.add(toks[0])
        ids[name] = sorted(found)

    owner = {}
    for name, id_list in ids.items():
        for t in id_list:
            if t in owner and owner[t] != name:
                raise RuntimeError(
                    f"클래스 '{owner[t]}'와 '{name}'의 첫 토큰이 같아 확신도를 계산할 수 없습니다: {t}"
                )
            owner[t] = name

    _LABEL_TOKEN_IDS = ids
    return ids


def classify_image(image_path, model, processor, prompt="v6", refs=None, ref_max_pixels=None):
    """
    반환: {"label": 5클래스 중 하나 또는 "unknown",
           "confidence": 선택한 클래스의 확률 (0~1),
           "probs": {클래스: 확률},
           "class_mass": 5개 클래스에 모델이 준 확률 합 (낮으면 모델이 다른 답을 하려 했다는 뜻),
           "raw": 모델이 생성한 원문}

    확신도 계산: 모델이 답의 '첫 번째 토큰'을 고를 때의 확률분포(softmax)에서
    5개 클래스 이름의 첫 토큰 확률만 모아 합이 1이 되게 다시 정규화한 값.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"이미지 파일을 찾을 수 없습니다: {image_path}"
        )

    if refs:
        content = build_ref_content(refs, ref_max_pixels)
        content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": "Answer with only the class name."})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": PROMPTS[prompt],
                    },
                ],
            }
        ]

    # Qwen chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 이미지 처리
    image_inputs, video_inputs = process_vision_info(messages)

    # Model input 생성
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Qwen의 주 실행 장치 확인
    device = next(
        p.device
        for p in model.parameters()
        if p.device.type != "meta"
    )

    inputs = inputs.to(device)

    label_ids = _label_first_token_ids(processor)

    print("결함 분류 중...")

    # 추론 (확신도 계산을 위해 토큰별 점수도 함께 받음)
    with torch.no_grad():

        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # 생성된 텍스트 (입력 prompt 부분 제거)
    generated = out.sequences[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip().lower()

    label = validate_result(raw)

    # 첫 토큰의 확률분포에서 클래스별 확률 계산
    first_logits = out.scores[0][0].float()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    class_prob = {
        name: torch.logsumexp(log_probs[id_list], dim=0).exp().item()
        for name, id_list in label_ids.items()
    }
    mass = sum(class_prob.values())
    probs = {name: p / mass for name, p in class_prob.items()} if mass > 0 else class_prob

    confidence = probs.get(label, 0.0)

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "class_mass": mass,
        "raw": raw,
    }


# ============================================================
# v8 속도 개선: 참고 이미지 60장은 처음 한 번만 계산(KV cache)하고, 이후 검사 이미지마다 새 부분만 계산
#  - 처음 한 장은 "전부 다시 계산한 결과"와 비교해서 같을 때만 캐시를 사용 (다르거나 에러가 나면 기존 방식으로 자동 전환)
# ============================================================

def _result_from_logits(first_logits, processor):
    """첫 토큰 로짓 -> classify_image 와 같은 형식의 결과 (확신도 계산 방식도 동일)"""
    tok = processor.tokenizer
    label_ids = _label_first_token_ids(processor)
    log_probs = torch.log_softmax(first_logits.float(), dim=-1)
    top_id = int(torch.argmax(log_probs))
    raw = tok.decode([top_id]).strip().lower()
    label = validate_result(raw)
    class_prob = {
        name: torch.logsumexp(log_probs[id_list], dim=0).exp().item()
        for name, id_list in label_ids.items()
    }
    mass = sum(class_prob.values())
    probs = {name: p / mass for name, p in class_prob.items()} if mass > 0 else class_prob
    return {"label": label, "confidence": probs.get(label, 0.0), "probs": probs,
            "class_mass": mass, "raw": raw}


class CachedRefClassifier:

    def __init__(self, model, processor, refs, ref_max_pixels):
        self.model, self.processor = model, processor
        self.refs, self.ref_max_pixels = refs, ref_max_pixels
        self.n_ref = len(refs)
        self.device = next(p.device for p in model.parameters() if p.device.type != "meta")
        self.vision_start = model.config.vision_start_token_id
        self.state = "untested"          # untested -> on / off
        self.cache = self.prefix_ids = self.L = self.n_ref_patch = None
        self.n_cached = self.n_fallback = 0
        self.ref_images = self._load_ref_images()

    def _load_ref_images(self):
        from PIL import Image
        images = []
        for r in self.refs:
            try:
                from qwen_vl_utils.vision_process import fetch_image
                images.append(fetch_image({"image": r["path"], "max_pixels": self.ref_max_pixels}))
            except Exception:
                images.append(Image.open(r["path"]).convert("RGB"))
        return images

    def _build_inputs(self, image_path):
        content = build_ref_content(self.refs, self.ref_max_pixels, images=self.ref_images)
        content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": "Answer with only the class name."})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")
        return inputs.to(self.device)

    def _rope(self, ids, grid, mask):
        fn = getattr(self.model, "get_rope_index", None) or getattr(self.model.model, "get_rope_index")
        out = fn(input_ids=ids, image_grid_thw=grid, video_grid_thw=None, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out

    def _cached_logits(self, inputs):
        ids, grid, pv = inputs["input_ids"], inputs["image_grid_thw"], inputs["pixel_values"]
        T = ids.shape[1]
        mask = torch.ones_like(ids)
        pos = self._rope(ids, grid, mask)
        L = int((ids[0] == self.vision_start).nonzero()[-1])         # 마지막 이미지(검사 이미지) 시작 위치
        with torch.no_grad():
            if self.cache is None:                                    # 참고 이미지 부분: 처음 한 번만 계산
                self.n_ref_patch = int(grid[:self.n_ref].prod(-1).sum())
                out = self.model(input_ids=ids[:, :L], attention_mask=mask[:, :L],
                                 pixel_values=pv[:self.n_ref_patch], image_grid_thw=grid[:self.n_ref],
                                 position_ids=pos[:, :, :L], use_cache=True)
                self.cache, self.prefix_ids, self.L = out.past_key_values, ids[:, :L].clone(), L
            elif L != self.L or not torch.equal(ids[:, :L], self.prefix_ids):
                raise RuntimeError("참고 이미지 부분이 이전과 달라 캐시를 쓸 수 없습니다")
            out = self.model(input_ids=ids[:, L:], attention_mask=mask,
                             pixel_values=pv[self.n_ref_patch:], image_grid_thw=grid[self.n_ref:],
                             position_ids=pos[:, :, L:], past_key_values=self.cache,
                             cache_position=torch.arange(L, T, device=self.device), use_cache=True)
        self.cache.crop(self.L)                                       # 검사 이미지 부분은 지우고 참고 이미지 부분만 남김
        return out.logits[0, -1].float()

    def _full_logits(self, inputs):
        with torch.no_grad():
            out = self.model(**inputs, logits_to_keep=1)
        return out.logits[0, -1].float()

    def _fallback(self, image_path):
        self.n_fallback += 1
        return classify_image(image_path, self.model, self.processor, prompt="v8",
                              refs=self.refs, ref_max_pixels=self.ref_max_pixels)

    def classify(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        if self.state == "off":
            return self._fallback(image_path)
        try:
            inputs = self._build_inputs(image_path)
            cached = self._cached_logits(inputs)
            if self.state == "untested":                              # 처음 한 장: 전부 다시 계산한 결과와 비교
                full = self._full_logits(inputs)
                ids = sorted({t for v in _label_first_token_ids(self.processor).values() for t in v})
                lc, lf = torch.log_softmax(cached, -1), torch.log_softmax(full, -1)
                diff = float((lc[ids] - lf[ids]).abs().max())
                same_top = int(torch.argmax(cached)) == int(torch.argmax(full))
                if same_top and diff < 0.3:
                    self.state = "on"
                    print("[캐시 확인 통과] 전체 계산과 결과 일치 (클래스 로그확률 최대 차이 {:.3f}) -> 참고 이미지 캐시 사용".format(diff), flush=True)
                else:
                    self.state = "off"
                    print("[캐시 확인 실패] 전체 계산과 결과가 다릅니다 (top1 같음={}, 최대 차이 {:.3f}) -> 기존 방식으로 진행".format(same_top, diff), flush=True)
                    return self._fallback(image_path)
            self.n_cached += 1
            return _result_from_logits(cached, self.processor)
        except Exception as e:
            self.state = "off"
            print("[캐시 사용 불가] {}: {} -> 기존 방식(전부 다시 계산)으로 진행".format(type(e).__name__, str(e)[:200]), flush=True)
            return self._fallback(image_path)


# ============================================================
# Validate Qwen output
# ============================================================

def validate_result(result):

    # 정확히 class 이름이 나온 경우
    if result in CLASSES:
        return result

    # 문장 형태로 출력된 경우 class 이름 추출 (불량 타입 먼저, 마지막에 normal)
    for defect in DEFECT_CLASSES:
        if defect in result:
            return defect
    if "normal" in result:
        return "normal"

    return "unknown"


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="AeBAD defect classification using Qwen2-VL"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="검사할 AeBAD 이미지 경로",
    )

    args = parser.parse_args()

    # 모델 로딩
    model, processor = load_model()

    # 분류
    res = classify_image(
        args.image,
        model,
        processor,
    )

    # 결과 정리
    result = res["label"]

    print()
    print("====================================")
    print("       Classification Result")
    print("====================================")
    print("Image :", args.image)
    print("Result:", result)
    print("Confidence: {:.1%}".format(res["confidence"]))
    print("Probs :", {k: round(v, 3) for k, v in res["probs"].items()})
    print("====================================")


if __name__ == "__main__":
    main()
