"""모의 기준값 폴더 생성 (개발/테스트 전용): params.json, separation.json, gallery.json + 합성 이미지.
실제 서버에서는 offline/build_params.py 가 만든 폴더를 쓴다."""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "offline"))
import separation as sep  # noqa: E402


def synth_image(seed, size=(320, 240)):
    rnd = np.random.RandomState(seed)
    base = np.linspace(40, 200, size[0], dtype=np.float32)[None, :, None] * np.ones((size[1], 1, 3), np.float32)
    img = base + rnd.randn(size[1], size[0], 3) * 6
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make(out, n_gallery=6):
    os.makedirs(os.path.join(out, "gallery_imgs"), exist_ok=True)
    items = []
    for i in range(n_gallery):
        rel = "gallery_imgs/g{}.png".format(i)
        synth_image(100 + i).save(os.path.join(out, rel))
        sha = sep.sha256_file(os.path.join(out, rel))
        items.append({"id": hashlib.sha1(sha.encode()).hexdigest()[:12], "file": rel, "sha256": sha})
    calib_rel = "gallery_imgs/_calib_like.png"           # 보정에 쓰인 것처럼 취급할 이미지 (갤러리에는 넣지 않음)
    synth_image(999).save(os.path.join(out, calib_rel))
    calib_sha = sep.sha256_file(os.path.join(out, calib_rel))
    json.dump({"items": items, "refs": []}, open(os.path.join(out, "gallery.json"), "w"))
    json.dump({"refs": ["r" * 64], "calib": [calib_sha, "c" * 64], "gallery": [g["sha256"] for g in items]},
              open(os.path.join(out, "separation.json"), "w"))
    f = 13
    json.dump({"tau_lo": -0.5, "tau_hi": 1.2, "t": 0.98,
               "router": {"mean": [0.45] + [0.0] * (f - 1), "scale": [0.1] + [1.0] * (f - 1), "coef": [1.0] + [0.0] * (f - 1), "intercept": 0.0}},
              open(os.path.join(out, "params.json"), "w"))
    return {"gallery": items, "calib_file": os.path.join(out, calib_rel)}


if __name__ == "__main__":
    print(make(sys.argv[1]))
