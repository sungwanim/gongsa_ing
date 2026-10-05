"""환경변수 설정 (agent 서비스, velm_qwen 환경). 경로·포트·보안은 모두 여기서 읽는다."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.normpath(os.path.join(HERE, ".."))
VELM = os.path.normpath(os.path.join(E2E, "..", "ModelB", "velm"))
for p in (VELM, os.path.join(E2E, "offline"), os.path.join(E2E, "common")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOOPBACK = ("127.0.0.1", "localhost", "::1")


class Settings:
    def __init__(self, env=None):
        e = os.environ if env is None else env
        self.artifacts = e.get("AGENT_ARTIFACTS", "")             # params.json / separation.json / gallery.json 이 있는 폴더
        self.data_root = e.get("AGENT_DATA_ROOT", "")             # AeBAD 폴더 (갤러리·참고 이미지 읽기 전용)
        self.mmr_url = e.get("AGENT_MMR_URL", "http://127.0.0.1:8101")
        self.host = e.get("AGENT_HOST", "127.0.0.1")
        self.port = int(e.get("AGENT_PORT", "8200"))
        self.token = e.get("AGENT_TOKEN", "")                     # 비어 있으면 인증 없음(로컬 전용일 때만 허용)
        self.cors_origin = e.get("AGENT_CORS_ORIGIN", "")         # 같은 출처로 쓰면(프록시) 비워 둔다
        self.data_dir = e.get("AGENT_DATA_DIR", os.path.join(E2E, "data"))
        self.refs_manifest = e.get("AGENT_REFS_MANIFEST", os.path.join(VELM, "holdout_manifest_12.csv"))
        self.ref_px = int(e.get("AGENT_REF_PX", "128"))
        self.max_upload = int(e.get("AGENT_MAX_UPLOAD_MB", "30")) * 1024 * 1024

    def validate(self):
        if not self.artifacts or not os.path.isdir(self.artifacts):
            raise SystemExit("AGENT_ARTIFACTS(기준값 폴더)를 지정하세요: {!r}".format(self.artifacts))
        if not (self.data_root and os.path.isdir(self.data_root)):
            raise SystemExit("AGENT_DATA_ROOT(AeBAD 폴더)를 지정하세요: {!r}".format(self.data_root))
        if self.host not in LOOPBACK and not self.token:
            raise SystemExit("로컬 주소({})가 아닌 곳에 열려면 AGENT_TOKEN(접근 토큰)이 필요합니다.".format(self.host))
