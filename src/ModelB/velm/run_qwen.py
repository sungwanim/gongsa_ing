import argparse
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

DEFECT_CLASSES = [
    "ablation",
    "breakdown",
    "fracture",
    "groove",
]


def load_model():
    print("Qwen2-VL 모델 로딩 중...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("모델 로딩 완료")

    return model, processor


def classify_image(image_path, model, processor):
    defect_list = ", ".join(DEFECT_CLASSES)

    prompt = f"""
You are inspecting an aircraft engine blade image.

Classify the defect in the image into exactly one of the following classes:

{defect_list}

Return only the class name.
Do not provide any additional explanation.
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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
    )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="검사할 블레이드 이미지 경로",
    )

    args = parser.parse_args()

    model, processor = load_model()

    result = classify_image(
        args.image,
        model,
        processor,
    )

    print("\n===== Classification Result =====")
    print(result)


if __name__ == "__main__":
    main()
