"""Pin every BLAS/OpenMP pool to one thread before numpy loads.

Thread count changes lbfgs results (measurements_2026-09.md §4), and the probe
core asserts it, so the suite runs under the same pin as the recompute.
"""
import os
import sys
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
# Inserted in reverse so src/ ends up first (moved to the front even when the
# editable install already lists it last): scripts/tss_overlap.py must not
# shadow the src/tss_overlap package.
for _p in (ROOT, ROOT / "tests", ROOT / "scripts", ROOT / "src"):
    while str(_p) in sys.path:
        sys.path.remove(str(_p))
    sys.path.insert(0, str(_p))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _one_thread():
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield
