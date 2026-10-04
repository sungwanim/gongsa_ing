"""MMR이 찾은 이상 위치를 원본에서 잘라 내고(해상도 유지), 전체 사진에는 빨간 박스로 표시한다.

흐름
  1) MMR 이상 맵(anomaly_maps_*.npz, 이미지당 224x224)에서 점수가 가장 높은 덩어리의 네모(박스)를 찾는다.
  2) 맵 좌표를 원본 좌표로 바꾼다.
       MMR은 이미지를 256x256으로 줄인 뒤 가운데 224x224를 썼다 -> 원본 x = (맵 x + 16) * (원본 가로 / 256)
  3) 박스를 감싸는 정사각형 영역을 원본에서 그대로 자른다(줄이지 않는 것이 목적).
  4) 전체 사진(줄임) + 같은 영역을 빨간 박스로 표시한 그림, 그리고 자른 사진(원본 해상도) 두 장을 돌려준다.

참고 이미지(정답 마스크가 있는 이미지)는 마스크에서 박스를 만든다.
"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

MAP_SIZE = 224          # MMR 이상 맵 크기
RESIZE = 256            # MMR이 줄인 크기 (Resize((256,256)) 후 CenterCrop(224))
OFFSET = (RESIZE - MAP_SIZE) // 2   # = 16


class MapIndex:
    """out_dir 의 anomaly_maps_*.npz 를 읽어 (타입폴더/도메인/파일이름) -> 224x224 맵 으로 찾을 수 있게 한다."""

    def __init__(self, mmr_out):
        self.maps = {}
        files = sorted(glob.glob(os.path.join(mmr_out, "anomaly_maps_*.npz")))
        if not files:
            raise FileNotFoundError("이상 맵이 없습니다: {}/anomaly_maps_*.npz".format(mmr_out))
        for f in files:
            z = np.load(f, allow_pickle=True)
            paths = [str(p) for p in z["image_paths"]]
            amaps = z["anomaly_maps"]
            if len(paths) != len(amaps):
                raise ValueError("{}: image_paths({})와 anomaly_maps({}) 개수가 다릅니다".format(f, len(paths), len(amaps)))
            for p, m in zip(paths, amaps):
                self.maps[self._key(p)] = m

    @staticmethod
    def _key(path):
        parts = path.replace("\\", "/").split("/")
        return tuple(parts[-3:])        # 결함폴더 / 도메인 / 파일이름

    def get(self, path):
        return self.maps[self._key(path)]


def bbox_from_map(amap, rel=0.5):
    """맵의 가장 높은 점을 포함하는 덩어리의 네모. rel: (최소~최대 범위에서) 몇 %부터 덩어리로 볼지."""
    lo, hi = float(amap.min()), float(amap.max())
    mask = amap >= lo + rel * (hi - lo)
    lab, _ = ndimage.label(mask)
    py, px = np.unravel_index(int(np.argmax(amap)), amap.shape)
    comp = lab == lab[py, px]
    ys, xs = np.where(comp)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1      # 맵 좌표 (x0, y0, x1, y1)


def map_box_to_original(box, width, height):
    x0, y0, x1, y1 = box
    sx, sy = width / float(RESIZE), height / float(RESIZE)
    return ((x0 + OFFSET) * sx, (y0 + OFFSET) * sy, (x1 + OFFSET) * sx, (y1 + OFFSET) * sy)


def square_region(box, width, height, min_side=560, scale=1.3, max_side=900):
    """박스를 감싸는 정사각형 (원본 좌표). 너무 작으면 주변이 보이게 min_side까지 넓히고, 너무 크면 max_side로 제한."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * scale
    side = min(max(side, min_side), max_side, width, height)
    left = min(max(cx - side / 2.0, 0), width - side)
    top = min(max(cy - side / 2.0, 0), height - side)
    return int(left), int(top), int(left + side), int(top + side)


def box_from_mask(mask_path, width, height):
    """정답 마스크에서 결함 네모 (원본 좌표). 마스크 크기가 원본과 다르면 비율로 맞춘다."""
    m = np.array(Image.open(mask_path).convert("L")) > 0
    if not m.any():
        raise ValueError("마스크가 비어 있습니다: {}".format(mask_path))
    ys, xs = np.where(m)
    mh, mw = m.shape
    sx, sy = width / float(mw), height / float(mh)
    return (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)


def make_views(image_path, region_box=None, overview_side=504, crop_side=560, draw_box=True):
    """(전체 사진 + 빨간 박스, 박스 부분 확대 사진). region_box 는 원본 좌표의 정사각형 영역.
    확대 사진은 원본에서 그대로 자르고, 영역이 crop_side보다 클 때만 그 크기로 줄인다."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left, top, right, bottom = region_box
    crop = img.crop((left, top, right, bottom))
    if crop.size[0] > crop_side:
        crop = crop.resize((crop_side, crop_side), Image.LANCZOS)
    overview = img.resize((overview_side, overview_side), Image.LANCZOS)
    if draw_box:
        d = ImageDraw.Draw(overview)
        k = overview_side / float(w)
        d.rectangle([left * k, top * k, right * k, bottom * k], outline=(255, 0, 0), width=4)
    return overview, crop


def views_from_map(image_path, map_index, region_kw=None, **kw):
    """검사 이미지: MMR 이상 맵으로 영역을 정한다. region_kw: square_region 인자(주변 맥락 크기 등)."""
    w, h = Image.open(image_path).size
    box = map_box_to_original(bbox_from_map(map_index.get(image_path)), w, h)
    return make_views(image_path, square_region(box, w, h, **(region_kw or {})), **kw)


def views_from_mask(image_path, region_kw=None, **kw):
    """참고 이미지: 정답 마스크로 영역을 정한다. (test -> ground_truth 폴더, 같은 파일 이름)"""
    mask_path = image_path.replace("/test/", "/ground_truth/")
    w, h = Image.open(image_path).size
    box = box_from_mask(mask_path, w, h)
    return make_views(image_path, square_region(box, w, h, **(region_kw or {})), **kw)
