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
CLASSES = DEFECT_CLASSES + ["normal"]


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
# Prompt: README의 프롬프트 원문 + 선택지만 Normal + 결함 4종(5지선다)으로 변경
#  - 참고 이미지 4장을 쓰지 않으므로 그 이미지를 가리키는 문장만 삭제
#  - 최종 출력만 "클래스 이름 한 단어" (확신도는 모델 확률로 계산)
# ============================================================

PROMPT_5CLASS = """당신은 항공엔진 터빈 블레이드 외관 검사를 수행하는 비전 분석 전문가입니다.

지금부터 제공되는 이미지는 AeBAD(Aero-engine Blade Anomaly Detection) 데이터셋의 터빈 블레이드 이미지입니다.

제공되는 이미지는 실제로 분류해야 하는 **검사 대상 이미지**입니다.

같은 결함이라도 결함의 크기, 모양, 위치, 방향, 깊이, 손상 정도, 촬영 각도에 따라 외관이 크게 달라질 수 있습니다.


### 정상 블레이드에 대한 기본 기준

정상적인 블레이드는 전체적인 외곽 형상이 자연스럽고 연속적으로 유지되며, 표면이나 가장자리에 명확한 재료 손실, 파손, 결손 또는 비정상적인 홈이 나타나지 않습니다.

터빈 블레이드는 금속성 표면을 가지므로 촬영 환경에 따라 밝은 반사, 어두운 영역, 하이라이트, 그림자, 색상 차이 등이 나타날 수 있습니다.

따라서 다음과 같은 현상을 결함으로 오인하지 마세요.

조명에 의한 밝기 변화, 금속 표면의 반사광, 그림자, 배경 변화, 카메라 촬영 각도 변화, 블레이드의 회전 방향이나 이미지 내 위치 변화, 원근에 따른 크기 변화.

단순히 다른 색이나 밝기를 보이는 것이 아니라 실제 블레이드의 **표면 구조나 형상 자체가 변화했는지**를 중요하게 판단하세요.


### 1. Ablation

Ablation은 블레이드가 고온 가스 등에 지속적으로 노출되면서 발생하는 열적·표면적 손상입니다.

단순히 한 지점이 깨지는 것보다는 표면이 침식되거나 마모되고 재료가 점차 손실되면서 원래 표면 상태가 변한 형태로 나타날 수 있습니다.

하지만 ablation은 항상 동일한 모양으로 나타나는 것은 아닙니다.

넓거나 좁은 영역에 나타날 수 있고, 표면의 거칠어짐, 침식, 불규칙한 재료 손실, 손상된 표면 질감 등 다양한 시각적 형태를 가질 수 있습니다.

단순한 밝기나 색상 변화만으로 Ablation이라고 판단하지 말고 실제 표면의 물리적인 변화가 보이는지 확인하세요.


### 2. Breakdown

Breakdown은 모래, 금속 조각, 새, 우박 등의 외부 물체 충돌과 같은 물리적 원인으로 발생할 수 있는 손상입니다.

블레이드 표면이나 가장자리가 찍히거나 깨지고, 국소적인 부분이 강하게 손상되는 등 비교적 뚜렷한 물리적 파괴 형태가 나타날 수 있습니다.

손상 크기는 매우 작을 수도 있고 비교적 클 수도 있으며, 가장자리나 표면 내부 등 다양한 위치에 발생할 수 있습니다.

불규칙한 깨짐, 찍힘, 충격 흔적, 국소적 재료 손실 등 **외부 충격에 의한 물리적 파손과 유사한 형태**가 있는지 확인하세요.

단, 단순한 작은 홈 하나만으로 Breakdown이라고 단정하지 말고 Groove와 비교하세요.


### 3. Fracture

Fracture는 손상이 심해져 블레이드의 일부가 실제로 파손되거나 결손된 상태를 의미합니다.

단순한 표면 흠집이나 얕은 홈보다 **블레이드의 전체적인 형상 또는 외곽선 자체가 크게 손상되었는지**를 중요하게 확인하세요.

블레이드의 일부 영역이 없어졌거나, 가장자리 또는 끝부분이 크게 떨어져 나간 형태처럼 보일 수 있습니다.

핵심은 표면에 작은 흔적이 있는 수준인지, 아니면 블레이드 자체의 일부가 손실되어 전체 형상이 변했는지를 구별하는 것입니다.


### 4. Groove

Groove는 블레이드에 생기는 작은 홈 형태의 결함으로, AeBAD에서는 특히 블레이드 가장자리의 작은 결함으로 나타날 수 있습니다.

얇고 좁은 패임, 작은 홈, 국소적으로 파인 부분처럼 보일 수 있으며 손상 크기가 매우 작을 수 있습니다.

Groove 역시 반드시 길고 일정한 선 형태일 필요는 없습니다.

짧거나 불규칙한 홈, 작은 가장자리 패임 등 다양한 형태일 수 있으므로 작은 결함도 놓치지 말고 확대해서 관찰하는 것처럼 세밀하게 분석하세요.

Breakdown과 혼동되는 경우에는 손상의 규모와 형태를 비교하세요. 작은 국소적인 홈이나 패임에 가까운지, 더 강한 파손이나 충격성 손상에 가까운지를 종합적으로 판단하세요.


### 검사 방법

검사 대상 이미지를 분석할 때 먼저 블레이드 전체 형상을 확인하세요.

그다음 블레이드의 가장자리와 표면을 각각 자세히 관찰하여 비정상적으로 보이는 영역을 찾으세요.

이상이 발견되면 그 영역의 위치, 크기, 모양, 방향, 경계, 표면 질감, 재료 손실 정도, 블레이드 외곽선의 변화 정도를 확인하세요.

그 이상이 실제 물리적 손상인지 아니면 조명·반사·그림자 등에 의해 그렇게 보이는 것인지 먼저 구분하세요.

그 후 Normal(정상), Ablation, Breakdown, Fracture, Groove 다섯 가지 가능성을 모두 비교하세요.

하나의 특징만 보고 바로 결론 내리지 마세요.

예를 들어 "선처럼 보인다 → Groove"와 같이 단순한 규칙으로 판단하지 말고, 결함의 전체적인 형태와 주변 구조를 종합적으로 분석하세요.

위의 결함 특성을 참고하되 **최종 판단은 검사 대상 이미지에서 실제로 관찰되는 시각적 증거를 기준으로 하세요.**

이미지에서 확인할 수 없는 특징이나 원인을 임의로 만들어내지 마세요.


### 최종 출력

반드시 다음 다섯 클래스 중 하나를 최종 선택하세요.

Normal / Ablation / Breakdown / Fracture / Groove

그리고 선택한 클래스 이름 한 단어만 출력하세요. 다른 글자나 설명은 쓰지 마세요.
"""


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


def classify_image(image_path, model, processor):
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
                    "text": PROMPT_5CLASS,
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
