"""대시보드 DB 관리 (표준 라이브러리만 사용, 서비스가 켜져 있어도 안전).

  python3 dashboard_admin.py --data-dir ~/end2end/data list                       건수와 최근 기록(테스트/실제 구분)
  python3 dashboard_admin.py --data-dir ~/end2end/data clear --test-only [--yes]  점검용 테스트 기록만 삭제
  python3 dashboard_admin.py --data-dir ~/end2end/data clear --all [--yes]        전부 삭제
삭제 전에 DB 를 같은 폴더에 dashboard_backup_<시각>.sqlite3 로 항상 복사해 둔다. --yes 가 없으면 건수만 보여 주고 삭제하지 않는다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("cmd", choices=["list", "clear"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test-only", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.data_dir, "dashboard.sqlite3")):
        print("대시보드 DB 가 없습니다: {}".format(a.data_dir))
        return 0
    st = Store(a.data_dir)
    c = st.counts()
    if a.cmd == "list":
        print("저장된 기록 {}건 (실제 {}건, 점검용 테스트 {}건)".format(c["total"], c["real"], c["test"]))
        for it in st.list(limit=10, include_test=True):
            print("#{:<4} {} {:<8} {:<9} {:<8} {:>6.1f}s {}".format(it["id"], it["created_at"], it["source"], it["decision"], it["defect_type"] or "-",
                                                                it["timings"]["total_s"], "[테스트]" if it["is_test"] else ""))
        return 0
    if not (a.test_only or a.all):
        print("--test-only 또는 --all 을 지정하세요.")
        return 2
    n = c["test"] if a.test_only else c["total"]
    if not a.quiet or n:
        print("삭제 대상: {}건 ({})".format(n, "점검용 테스트 기록만" if a.test_only else "전부"))
    if not a.yes:
        print("(삭제하지 않았습니다. 실제로 지우려면 --yes 를 붙이세요. 지우기 전에 DB 는 자동으로 백업됩니다.)")
        return 0
    if n == 0:
        return 0
    bak = st.backup()
    deleted = st.purge(test_only=a.test_only)
    print("삭제 {}건. 백업: {}".format(deleted, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
