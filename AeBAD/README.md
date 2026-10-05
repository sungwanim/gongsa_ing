# AeBAD-S (평가용 사본)

- 출처: https://github.com/zhangzilongc/MMR (논문: arXiv 2304.02216)
- 라이선스: `LICENSE-DATASET` 참고 (CC BY 4.0, 출처 표기 필요)
- 변경 사항: AeBAD_V 제외(AeBAD_S만 포함), `.DS_Store` 등 macOS 찌꺼기 제거. 이미지 파일 바이트는 그대로.
- 참고: 이미지 파일은 확장자가 `.png`이지만 실제 인코딩은 JPEG(3024×3024)이다. 마스크(ground_truth)는 실제 PNG.
- 구성: train/good 521장, test 1,639장 (ablation 169, breakdown 329, fracture 389, good 490, groove 262), ground_truth 1,149장
- 무결성: `AeBAD_S.sha256` (경로 기준 SHA-256 목록)
