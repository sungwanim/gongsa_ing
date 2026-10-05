"""MMR 단일 이미지 추론 (mmr conda 환경, torch 2.1.2). 기존 MMR 코드는 수정하지 않고 그대로 import 해서 쓴다.

기존 평가(MMR_pipeline_.evaluation)와 같은 계산:
  전처리  Resize((256,256)) -> CenterCrop(224) -> ToTensor -> ImageNet 정규화
  이상 맵 teacher(WideResNet50)/student(MMR) 특징의 cosine 거리 합(amap_mode='a') -> Gaussian σ=4
  점수    이상 맵의 최댓값
저장된 점수/이상 맵 파일은 읽지 않는다 (체크포인트 .pth 만 사용).

메모리 방식 B: resident=False 이면 요청마다 모델을 올리고(load) 추론 후 GPU 메모리를 반납(unload)한다.
resident=True 이면 시작할 때 한 번 올려 두고 계속 유지한다.
"""
import gc
import io
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MMR_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "ModelA", "MMR_Test"))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MMRInferencer:
    def __init__(self, checkpoint, device="cuda:0", resident=False, cfg_file=None):
        self.checkpoint = checkpoint
        self.device_name = device
        self.resident = resident
        self.cfg_file = cfg_file or os.path.join(MMR_DIR, "method_config", "AeBAD_S", "MMR.yaml")
        self._lock = threading.Lock()      # GPU 는 한 번에 한 요청만
        self._pipe = None
        self._cfg = None
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError("MMR 체크포인트가 없습니다: {}".format(self.checkpoint))
        if resident:
            with self._lock:
                self._load()

    @property
    def loaded(self):
        return self._pipe is not None

    # ------------------------------------------------------------------ 로드/해제
    def _load(self):
        if MMR_DIR not in sys.path:
            sys.path.insert(0, MMR_DIR)
        import torch
        from config import get_cfg
        from models.MMR import MMR_pipeline_
        from tools.train import _load_models

        cfg = get_cfg()
        cfg.merge_from_file(self.cfg_file)       # load_config() 는 출력 폴더를 만들기 때문에 쓰지 않는다
        cur_model, mmr_base = _load_models(cfg, load_pretrain_model=False)   # 전체 가중치는 체크포인트에서 옴
        pipe = MMR_pipeline_(cur_model=cur_model, mmr_model=mmr_base, optimizer=None,
                             device=torch.device(self.device_name), cfg=cfg)
        pipe.load_model(self.checkpoint)
        pipe.cur_model.eval()
        pipe.mmr_model.eval()
        self._pipe, self._cfg = pipe, cfg

    def _unload(self):
        import torch
        self._pipe = None
        self._cfg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 추론
    def _run(self, img):
        import torch
        from scipy.ndimage import gaussian_filter
        from torchvision import transforms
        from models.MMR.utils import cal_anomaly_map

        pipe, cfg = self._pipe, self._cfg
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        x = tf(img.convert("RGB")).unsqueeze(0).to(pipe.device)
        layers = cfg.TRAIN.MMR.layers_to_extract_from
        with torch.no_grad():
            pipe.teacher_outputs_dict.clear()
            pipe.cur_model(x)
            feats = [pipe.teacher_outputs_dict[k] for k in layers]
            rev = pipe.mmr_model(x, mask_ratio=cfg.TRAIN.MMR.test_mask_ratio)
            amap, _ = cal_anomaly_map(feats, [rev[k] for k in layers], x.shape[-1], amap_mode="a")
        amap = gaussian_filter(amap[0], sigma=4).astype(np.float32)
        return amap

    def infer_image(self, img):
        """PIL 이미지 -> {"score", "map"(224x224 float32), "orig_size", "timings"}"""
        import torch
        t0 = time.time()
        with self._lock:
            t_lock = time.time()
            was_loaded = self.loaded
            if not was_loaded:
                self._load()
            t_load = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                amap = self._run(img)
                peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
            finally:
                if not self.resident:
                    self._unload()
            t_end = time.time()
        return {"score": float(amap.max()), "map": amap, "orig_size": list(img.size),
                "timings": {"wait_s": round(t_lock - t0, 3), "load_s": round(t_load - t_lock, 3) if not was_loaded else 0.0,
                            "infer_s": round(t_end - t_load, 3), "gpu_peak_mb": round(peak, 1), "resident": self.resident}}

    def infer_bytes(self, data):
        return self.infer_image(Image.open(io.BytesIO(data)))      # 기존 데이터 로더와 같게 EXIF 회전은 하지 않는다
