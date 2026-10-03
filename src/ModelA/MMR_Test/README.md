# MMR_Test

AeBAD-S 데이터셋으로 MMR(Masked Multi-scale Reconstruction) 이상탐지를 학습/평가한다.

- 원본: [zhangzilongc/MMR](https://github.com/zhangzilongc/MMR) (Apache-2.0, `LICENSE`, 데이터셋 라이선스 `LICENSE-DATASET`)
- 논문: [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216)

## 원본 대비 변경점 (Python 3.10 / torch 2.1.2 대응)

| 파일 | 변경 |
|---|---|
| `requirements.txt` | torch 2.1.2 환경 기준으로 재작성 (`timm==0.4.12`, `numpy<2`, `scikit-image` 추가) |
| `models/MMR/MMR.py` | timm 0.4.12 `Block`에 없는 `qk_scale=None` 인자 제거 (기본값과 동일 동작) |
| `models/MMR/utils.py` | `np.float` → `np.float64` |
| `utils/common.py` | `np.bool` → `bool`, pandas 2.0에서 삭제된 `DataFrame.append` 대체 |
| `main.py`, `utils/parser_.py` | `--device`를 안 주면 `CUDA_VISIBLE_DEVICES`를 덮어쓰지 않음 (MIG 대응) |
| `method_config/AeBAD_S/MMR.yaml` | 데이터셋 `../../../AeBAD`(레포 루트), MAE 가중치 `/home/team14/weights/` |
| `AeBAD_S_run.sh` | 스크립트 위치로 `cd` 후 실행, 추가 `--opts` 전달 가능 |

## 데이터 구조

레포 루트에 `AeBAD/`를 둔다 (git 제외).

```
gongsa_ing/AeBAD/AeBAD_S/
├── train/good/<sub>/*.png
├── test/<good|결함종류>/<same|background|illumination|view>/*.png
└── ground_truth/<결함종류>/<same|background|illumination|view>/*.png
```

## 서버 실행

```bash
cd ~/gongsa_ing && git fetch && git checkout mmr-test && git pull
conda activate mmr
pip install -r src/ModelA/MMR_Test/requirements.txt

# 1 epoch 동작 확인
bash src/ModelA/MMR_Test/AeBAD_S_run.sh TRAIN_SETUPS.epochs 1 TRAIN_SETUPS.warmup_epochs 0

# 본 학습 (200 epoch)
nohup bash src/ModelA/MMR_Test/AeBAD_S_run.sh > src/ModelA/MMR_Test/train_AeBAD_S.out 2>&1 &
```

결과는 `src/ModelA/MMR_Test/log_MMR_AeBAD_S_54/`에 로그(도메인별 Image AUROC / Pixel AUROC / PRO)와 시각화 이미지로 저장된다.
