# MMR_Test

AeBAD-S 데이터셋으로 **MMR(Masked Multi-scale Reconstruction)** 이상탐지 모델을 학습·평가한다.
프로젝트 구조상 **ModelA(빠른 1차 이상탐지)** 후보 모델이다.

- 원본 코드: [zhangzilongc/MMR](https://github.com/zhangzilongc/MMR) (commit `22d1b17`, Apache-2.0 → `LICENSE`, 데이터셋 라이선스 → `LICENSE-DATASET`)
- 논문: [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216) (Zhang et al., 2023)

---

## 1. MMR 개요

정상 이미지만으로 학습하는 비지도 이상탐지 방법이다.

1. **Teacher**: ImageNet 사전학습 WideResNet50(고정). `layer1~3`의 멀티스케일 특징을 뽑는다.
2. **Student**: MAE 사전학습 ViT-Base 인코더 + Simple FPN 디코더(디코더는 처음부터 학습).
   학습 시 입력 패치의 40%를 가린 상태에서 teacher의 3개 스케일 특징을 복원하도록 학습한다.
3. **추론**: 마스크 없이 복원 → 스케일별 teacher/student 특징 차이(cosine 거리)로 픽셀별 **이상 맵**을 만든다.
   - 픽셀 점수 = 이상 맵 (Gaussian 필터 σ=4)
   - 이미지 점수 = 이상 맵의 최댓값

가린 부분을 "정상 구조"로 복원하도록 배우기 때문에 조명·시점 변화(도메인 시프트)에 강하다.

## 2. 폴더 구조

```
MMR_Test/
├── main.py                       # 진입점: config 로드 → 학습(train) 또는 평가 전용(test)
├── AeBAD_S_run.sh                # 학습 + 평가 실행 스크립트
├── AeBAD_S_test.sh               # 저장된 가중치로 평가만 실행
├── requirements.txt
├── config/defaults.py            # 전체 기본 설정 (fvcore CfgNode)
├── method_config/AeBAD_S/MMR.yaml  # AeBAD-S 실험 설정 (기본값을 덮어씀)
├── datasets/
│   ├── mvtec.py                  # 기본 Dataset: 이미지/마스크 로드, transform
│   └── aebad_S.py                # AeBAD-S 폴더 구조 파싱
├── models/MMR/
│   ├── MMR.py                    # MAE ViT 인코더 + Simple FPN 디코더
│   ├── MMR_pipeline.py           # 학습 루프, 평가, 가중치 저장/로드
│   └── utils.py                  # 위치 임베딩, forward hook, loss, 이상 맵, LR 스케줄
├── tools/train.py                # train(): 학습→저장→평가 / test(): 로드→평가
└── utils/
    ├── common.py                 # 지표(AUROC, PRO, 분류 지표), 시각화 저장
    ├── load_dataset.py           # DataLoader 생성
    ├── backbones.py              # 사전학습 backbone 로드
    ├── parser_.py, logging_.py
```

## 3. 데이터 준비

`AeBAD/`는 git에 포함되지 않는다(`.gitignore`). **레포 루트**에 두고, 코드는 상대경로 `../../../AeBAD`로 접근한다.

```
gongsa_ing/AeBAD/AeBAD_S/
├── train/good/<background|illumination|view 등>/*.png
├── test/<good|결함종류>/<same|background|illumination|view>/*.png
└── ground_truth/<결함종류>/<same|background|illumination|view>/*.png
```

- 다운로드: [Google Drive](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) (`gdown 14wkZAFFeudlg0NMFLsiGwS0E593b-lNo`)
- 코드가 읽는 이미지 수: train **521**장, test **1639**장 (same 689 / background 305 / illumination 273 / view 372)
- `find`로 세면 train 554, test 1780장이 나오는데, 차이는 전부 macOS 압축 시 생긴 `._*.png` 메타데이터 파일이다. 코드(`glob("*.png")`)는 이를 읽지 않으므로 정상이다.

MAE 사전학습 가중치: [mae_visualize_vit_base.pth](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) → 서버 경로 `/home/team14/weights/` (yaml의 `TRAIN.MMR.model_chkpt`).
WideResNet50 ImageNet 가중치는 첫 실행 시 torchvision이 자동으로 내려받는다.

## 4. 환경

| 항목 | 버전 |
|---|---|
| Python | 3.10 |
| torch / torchvision | 2.1.2+cu121 / 0.16.2 (conda 환경 `mmr`에 설치됨) |
| GPU | H200 MIG 18GB |

```bash
conda activate mmr
pip install -r src/ModelA/MMR_Test/requirements.txt
```

`timm`은 반드시 **0.4.12**를 쓴다. 0.3.2는 torch 2.x에서 import 오류가 나고, 0.5 이상은 `add_weight_decay` 등 MMR이 쓰는 API가 바뀌었다.

## 5. 실행 방법 (GPU 서버)

> 서버 규칙: GPU 서버에서는 `git pull`과 실행만 한다. **push 금지.**

```bash
cd ~/gongsa_ing
git checkout main && git pull
conda activate mmr
```

### 5-1. 동작 확인 (smoke test, 수 분)

```bash
bash src/ModelA/MMR_Test/AeBAD_S_run.sh TRAIN_SETUPS.epochs 1 TRAIN_SETUPS.warmup_epochs 0 \
  TRAIN_SETUPS.num_workers 6 OUTPUT_DIR ./log_smoke
```

`warmup_epochs 0`을 꼭 넣는다. 없으면 첫 epoch 학습률이 0이라 가중치가 전혀 학습되지 않는다(점수는 무의미).

### 5-2. 학습 + 평가 (약 1시간)

tmux 세션 안에서 실행한다. SSH 연결이 끊겨도 세션이 살아 있어 학습이 계속된다.

```bash
tmux new -s mmr                                    # 세션 생성 (이미 있으면: tmux attach -t mmr)
conda activate mmr
bash src/ModelA/MMR_Test/AeBAD_S_run.sh TRAIN_SETUPS.num_workers 6
# 세션에서 빠져나오기: Ctrl+B 누른 뒤 d  (학습은 계속됨)
```

- 다시 접속해서 `tmux attach -t mmr`로 진행 상황을 본다. 세션 목록은 `tmux ls`.
- 로그는 터미널과 `log_MMR_AeBAD_S_54/*.log`에 함께 남는다. 스크롤은 `Ctrl+B` 다음 `[` (나갈 때 `q`).
- 실행 중인지 확인: `ps aux | grep "[m]ain.py"`
- 소요 시간: 학습 epoch당 약 15초 × 200 ≈ 50분, 평가 4개 도메인 약 5분
- 서버 CPU 권장 worker 수가 6이라 `num_workers 6`을 넘긴다(기본 8은 경고 발생).
- 끝났는데 `main.py: Main function complete!` 이후에도 프로세스가 안 끝나면(DataLoader worker 정리 지연) 결과는 이미 저장된 상태이므로 `Ctrl+C`로 종료해도 된다. 안 되면 다른 창에서 `pkill -f main.py`.

### 5-3. 저장된 가중치로 평가만 (학습 없이)

```bash
bash src/ModelA/MMR_Test/AeBAD_S_test.sh \
  log_MMR_AeBAD_S_54/checkpoints/MMR_aebad_S_AeBAD_S.pth TRAIN_SETUPS.num_workers 6
```

가중치 경로는 `MMR_Test/` 기준 상대경로 또는 절대경로. 결과는 `log_MMR_AeBAD_S_eval_54/`에 저장된다.

### 설정 덮어쓰기

스크립트 뒤에 `KEY VALUE` 쌍을 붙이면 yaml 값을 덮어쓴다 (`config/defaults.py` 참고).

| 예시 | 의미 |
|---|---|
| `TRAIN_SETUPS.epochs 1` | epoch 수 |
| `TRAIN_SETUPS.num_workers 6` | DataLoader worker 수 (학습·평가 공통) |
| `OUTPUT_DIR ./log_xxx` | 결과 폴더 (`_<RNG_SEED>`가 뒤에 붙음) |
| `TEST.image_threshold 1.23` | 이미지 양품/불량 판정 기준값 고정 |
| `RNG_SEED 0` | 랜덤 시드 |

## 6. 결과물

`MMR_Test/log_MMR_AeBAD_S_54/`

| 파일 | 내용 | git |
|---|---|---|
| `*.log` | 전체 로그 (도메인별 지표, 평균) | 포함 |
| `image_scores_aebad_S_AeBAD_S_<도메인>.csv` | 이미지별 `image_path, label(1=불량), score, prediction` — 기준값을 바꿔 다시 분석할 때 사용 | 포함 |
| `anomaly_maps_aebad_S_AeBAD_S_<도메인>.npz` | 이미지별 **원본 anomaly score**(`scores`)와 **원본 anomaly map**(`anomaly_maps`, N×224×224 float32, Gaussian σ=4 적용 후 정규화 전), `image_paths`, `labels`. 행 순서는 위 csv와 같음 | 제외 (용량) |
| `checkpoints/MMR_aebad_S_AeBAD_S.pth` | 학습된 MMR 가중치 (약 390MB). WideResNet50은 고정 ImageNet 가중치라 저장하지 않음 | 제외 |
| `image_save/`, `video_save/` | 도메인별 무작위 40장의 이상 맵 시각화 (정규화·양자화된 이미지) | 제외 |

git에서는 가중치(`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`), `*.npz`, `image_save/`, `video_save/`만 제외하고 나머지 결과(로그, csv)는 커밋한다 (`MMR_Test/.gitignore`).

`.npz`는 NumPy 배열 묶음 압축 파일이다.

```python
import numpy as np
d = np.load("anomaly_maps_aebad_S_AeBAD_S_same.npz")
d["anomaly_maps"][i], d["scores"][i], d["image_paths"][i]   # i번째 이미지
```

같은 `OUTPUT_DIR`로 다시 실행하면 `image_save/`와 가중치가 덮어써지므로, 실험마다 `OUTPUT_DIR`를 바꾼다.

## 7. 평가 지표

도메인 4개(`same`, `background`, `illumination`, `view`) 각각에 대해 계산하고, 마지막에 평균을 출력한다.

### 이미지 단위 (양품/불량 분류)

| 지표 | 계산 | 의미 |
|---|---|---|
| Image AUROC | 이미지 점수로 ROC AUC | 기준값과 무관한 분리 능력 |
| Recall (재현율) | TP / (TP+FN) | 실제 불량 중 불량으로 잡은 비율. **1차 검사에서 가장 중요** |
| FPR | FP / (FP+TN) | 양품을 불량으로 잘못 판정한 비율 → 2차 검사 부담 |
| Precision (정밀도) | TP / (TP+FP) | 불량 판정 중 실제 불량 비율 |
| F1 Score | 2·P·R / (P+R) | 정밀도·재현율 조화평균 |
| Accuracy (정확도) | (TP+TN) / 전체 | 전체 정답률 |
| 혼동행렬 | `[[TP, FN], [FP, TN]]` | 행: 정답 불량/양품, 열: 예측 불량/양품 |

**판정 기준값**: 점수 ≥ 기준값이면 불량.
- 기본(`TEST.image_threshold < 0`): **`same` 도메인에서 F1이 최대가 되는 기준값**을 구해 4개 도메인 모두에 같은 값을 적용한다. 실제 현장에서는 도메인 변화를 미리 알 수 없기 때문.
- AeBAD-S에는 별도 검증 세트가 없어 기준값을 테스트 세트(`same`)에서 고른다. `same`의 분류 지표는 다소 낙관적이다.
- 고정값을 쓰려면 `TEST.image_threshold <값>`. 사용된 기준값은 로그에 출력된다.

### 픽셀 단위 (결함 위치)

| 지표 | 의미 |
|---|---|
| Pixel AUROC | 모든 픽셀을 정상/결함으로 나눈 ROC AUC. 큰 결함과 넓은 정상 배경의 영향이 큼 |
| PRO (Per-Region Overlap) | 결함 영역(연결 성분)마다 검출 비율을 구해 **영역별 동일 가중치**로 평균, FPR 0~0.3 구간의 곡선 아래 면적. 작은 결함도 공정하게 반영 |

이 코드의 PRO는 이미지별로 곡선을 계산해 평균한다(MVTec 표준 AUPRO는 전체 이미지를 한 곡선으로 계산). 논문 수치와는 같은 방식이라 비교 가능하지만, 다른 모델과 비교할 때는 계산 방식을 맞춰야 한다.

## 8. 결과 (2026-10-03, seed 54, 200 epoch)

| 도메인 | Image AUROC | 논문 | PRO | 논문 | Pixel AUROC |
|---|---|---|---|---|---|
| same | 85.8 | 85.6 ±0.5 | 89.6 | 89.6 ±0.2 | 93.2 |
| background | 83.5 | 84.4 ±0.7 | 90.4 | 90.1 ±0.2 | 88.0 |
| illumination | 89.2 | 88.8 ±0.5 | 90.4 | 90.2 ±0.2 | 92.8 |
| view | 79.7 | 79.9 ±0.6 | 86.4 | 86.3 ±0.3 | 88.7 |
| **평균** | **84.55** | **84.7** | **89.2** | **89.1** | **90.7** |

논문(Table 2, 3; 5회 평균) 수치를 재현했다. 이미지 분류 지표(Recall/FPR/Precision/F1/Accuracy)는 이 실행 이후 추가되어, 재학습 시 함께 출력된다.

## 9. 원본 대비 변경점

| 파일 | 변경 | 이유 |
|---|---|---|
| `requirements.txt` | torch 2.1.2 기준 재작성 (`timm==0.4.12`, `numpy<2`, `scikit-image` 추가) | 원본은 torch 1.10 / timm 0.3.2 |
| `models/MMR/MMR.py` | `Block(...)`의 `qk_scale=None` 제거 | timm 0.4.12에 없는 인자, 기본값과 동일 동작 |
| `models/MMR/utils.py` | `np.float` → `np.float64` | numpy 1.24에서 삭제 |
| `utils/common.py` | `np.bool` → `bool`, `DataFrame.append` 대체 | numpy/pandas 2.0에서 삭제 |
| `main.py`, `utils/parser_.py` | `--device` 미지정 시 `CUDA_VISIBLE_DEVICES` 유지 | MIG 환경 대응 |
| `method_config/AeBAD_S/MMR.yaml` | 데이터/MAE 경로, `save_model: True` | 팀 경로 규칙, 가중치 저장 |
| `AeBAD_S_run.sh` | 스크립트 폴더로 `cd`, 추가 옵션 전달 | 어디서 실행해도 상대경로 유지 |
| `models/MMR/MMR_pipeline.py`, `tools/train.py` | 평가 결과에 원본 anomaly map 포함, 이미지별 score·map을 `.npz`로 저장 | 히트맵 생성 전 원본 값 확보 |
| `models/MMR/MMR_pipeline.py` | `save_model`/`load_model` 구현, `evaluation()`이 이미지별 점수 반환 | 원본은 가중치를 저장하지 않음 |
| `tools/train.py`, `main.py` | 평가 전용 `test()` 추가, 이미지 분류 지표·점수 CSV 저장 | 1차 검사용 판정 지표 필요 |
| `utils/common.py` | `best_f1_threshold`, `compute_image_classification_metrics` 추가 | 〃 |
| `AeBAD_S_test.sh` | 신규 | 저장된 가중치로 평가만 실행 |

제외한 원본 파일: `assets/`(데모 GIF 206MB), AeBAD-V·MVTec 실행 스크립트, 원본 README.
