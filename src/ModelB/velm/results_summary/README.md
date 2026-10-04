# velm 결과 요약 (서버 results/ 에서 가져온 파일)

라우터 + Qwen 최종 설정 (MMR 목표 0.98 / Qwen normal 재현율 목표 0.97), 보고용 829장 기준.

생성 명령:
python qwen_eval.py --mmr-out ~/gongsa_ing/src/ModelA/MMR_Test/log_MMR_AeBAD_S_54 --prompt v8_n12 --v8-rule --router --step eval --recall-target 0.98 --normal-recall 0.97

- comparison_v8_n12_report_router.txt / .csv : 성능 요약표 (전체 / 결함 타입별 / 혼동행렬)
- predictions_v8_n12_report_router.csv : 사진별 최종 판정
