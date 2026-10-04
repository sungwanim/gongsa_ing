AeBAD-S test images: 12 original image files (.png filenames)

background: 4
illumination: 4
view: 4

Each domain contains 1 image per defect: ablation, breakdown, fracture, groove.
Normal (good) images are not included. Images are unmodified.
Selection: first drawn with random seed 42 (5 images per domain and defect = 60 images, sorted original filenames, macOS metadata files excluded), then reduced by hand to 1 image per domain and defect (12 images).
manifest.csv records original paths and SHA-256 checksums.

Source: https://github.com/zhangzilongc/MMR
Official archive: https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view
Dataset authors: Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen.
Paper: Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction (2023).
https://arxiv.org/abs/2304.02216
Dataset license: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Changes: subset selected (reduced from 60 to 12 images); folder organization changed; image bytes unchanged.

Encoding note: The selected files are JPEG-encoded in the official archive despite their .png filenames. Original filenames and bytes were retained; image decoders can detect the format.
