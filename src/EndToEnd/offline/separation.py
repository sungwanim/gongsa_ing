"""참고 이미지 / 보정(파라미터 계산) 이미지 / 데모 이미지의 엄격한 분리를 해시(SHA-256)로 검사한다.

- 오프라인(build_params.py)이 만든 separation.json 을 앱 시작 때 다시 검사한다. 겹치면 실행을 중단한다.
- 업로드된 이미지가 참고/보정 이미지와 같은 파일이면 어느 집합인지 알려 준다(데모에 쓰면 안 되는 이미지).
"""
import argparse
import hashlib
import json
import os

SETS = ("refs", "calib", "gallery")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_disjoint(sets):
    """sets: {이름: 해시 집합}. 서로 겹치는 쌍이 있으면 {(a, b): 겹치는 개수} 로 돌려준다(없으면 빈 dict)."""
    names = list(sets)
    bad = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(sets[names[i]]) & set(sets[names[j]])
            if inter:
                bad[(names[i], names[j])] = len(inter)
    return bad


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {k: set(d[k]) for k in SETS}, d


def classify_upload(sha, sep_sets):
    """업로드 이미지의 해시가 어느 집합에 속하는지. 없으면 None."""
    for k in SETS:
        if sha in sep_sets[k]:
            return k
    return None


def verify(path):
    """separation.json 의 집합이 서로 겹치지 않는지 확인. 겹치면 예외."""
    sets, d = load(path)
    bad = check_disjoint(sets)
    if bad:
        raise RuntimeError("이미지 분리 위반: {}".format({"{}-{}".format(*k): v for k, v in bad.items()}))
    return {k: len(v) for k, v in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args()
    print("분리 검사 통과:", verify(os.path.join(a.artifacts, "separation.json")))


if __name__ == "__main__":
    main()
