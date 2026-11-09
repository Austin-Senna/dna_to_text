"""Pin every BLAS/OpenMP pool to one thread before numpy loads; hide the GPUs.

Thread count changes lbfgs results (measurements_2026-09.md §4), and the probe
core asserts it, so the suite runs under the same pin as the recompute. Tests
never use a GPU: the local one is shared with other work, and the clean-room
machine has none, so a test that initialises CUDA must fail, not borrow it.
"""
import os
import sys
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
# Inserted in reverse so src/ ends up first (moved to the front even when the
# editable install already lists it last): scripts/tss_overlap.py must not
# shadow the src/tss_overlap package.
for _p in (ROOT, ROOT / "tests", ROOT / "scripts", ROOT / "src"):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import pytest  # noqa: E402


def _split_digests() -> dict[str, str]:
    import hashlib
    files = sorted((ROOT / "data").glob("splits*.json")) + sorted((ROOT / "data").glob("binary_*.json"))
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


_SPLITS_AT_START: dict[str, str] = {}


def pytest_sessionstart(session):
    # Before collection, so a write at import time is caught too (G15).
    _SPLITS_AT_START.update(_split_digests())


def pytest_sessionfinish(session, exitstatus):
    """G15: no test may leave a split file different from how it found it.

    This sees net change only; a swap restored in ``finally`` is the static
    scan's job (tests/test_split_writers.py).
    """
    after = _split_digests()
    changed = sorted(k for k in _SPLITS_AT_START.keys() | after.keys()
                     if _SPLITS_AT_START.get(k) != after.get(k))
    if changed:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(f"\nG15: split files changed during the test session: {changed}", file=sys.stderr)


@pytest.fixture(scope="session", autouse=True)
def _one_thread():
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield
