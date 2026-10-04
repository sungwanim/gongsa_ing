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
# Defect classification
# ============================================================

def classify_image(image_path, model, processor):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"이미지 파일을 찾을 수 없습니다: {image_path}"
        )

    defect_list = "\n".join(
        [f"- {name}" for name in CLASSES]
    )

    prompt = f"""
You are inspecting an aircraft engine blade.

The blade in the image may be normal or may have one defect.

Classify the image into exactly ONE of the following five classes:

{defect_list}

Rules:
1. Select exactly one class.
2. Return only the class name.
3. Do not provide an explanation.
4. Do not return any class other than the five listed above.
"""

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
                    "text": prompt,
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

    print("결함 분류 중...")

    # 추론
    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    # 입력 prompt 부분 제거
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    # 텍스트 변환
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = result.strip().lower()

    return result


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
    raw_result = classify_image(
        args.image,
        model,
        processor,
    )

    # 결과 정리
    result = validate_result(raw_result)

    print()
    print("====================================")
    print("       Classification Result")
    print("====================================")
    print("Image :", args.image)
    print("Result:", result)
    print("====================================")


if __name__ == "__main__":
    main()
