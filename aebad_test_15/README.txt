AeBAD-S test subset: 15 original images

background, illumination, view: 5 images each.
Each domain: ablation 1, breakdown 1, fracture 1, groove 1, good (normal) 1.
Labels are inherited from original archive folders, not inferred from appearance.
Defect images: first manifest entry of each domain/class from aebad_test_60.zip.
Normal images: first original filename in sorted test/good/domain folder, excluding macOS metadata.
No image resizing or re-encoding. Original .png filenames retained (original bytes may be JPEG-encoded).
Original paths and SHA-256 checksums are in manifest.csv.
Ground-truth masks are not included.

Authors: Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen.
Source: https://github.com/zhangzilongc/MMR
Official archive: https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view
Paper: Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction (2023).
https://arxiv.org/abs/2304.02216
License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Changes: subset selection and folder reorganization; image bytes unchanged.
