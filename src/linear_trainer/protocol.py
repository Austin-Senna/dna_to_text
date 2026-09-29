"""The probe protocol: one frozen recipe shared by every fit, plus run provenance.

``V2`` is the camera-ready protocol (recompute_plan.md, Phase 0 and 1A). The
May 2026 protocol that produced the accepted paper's numbers can be expressed
as a ``Protocol`` too; tests/test_acceptance.py builds it to prove the refactor
reproduces the recorded values.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import threadpoolctl

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = ("src", "scripts", "tests", "pyproject.toml", "uv.lock")


class ThreadPinError(RuntimeError):
    """A BLAS/OpenMP pool runs at a thread count the protocol does not allow."""


def decades(lo: int, hi: int) -> tuple[float, ...]:
    """(1e{lo}, ..., 1e{hi}) as exact decimal literals."""
    return tuple(float(f"1e{e}") for e in range(lo, hi + 1))


@dataclass(frozen=True)
class Protocol:
    name: str
    scale: bool                    # StandardScaler fitted on the fit rows, inside every fit
    dtype: str                     # "float64" or "float32"
    base_grid: tuple[float, ...]   # ascending decades, shared by C and alpha
    limits: dict = field(default_factory=dict)  # {"C": (lo, hi), "alpha": (lo, hi)}
    plateau_eps: float = 1e-4      # an extension gaining no more than this stops, flagged
    tol: float | None = None       # lbfgs tolerance; None keeps the sklearn default
    max_iter: int = 5000
    threads: int | None = 1        # asserted before every cell; None skips the assert

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


V2 = Protocol(
    name="v2-2026-09",
    scale=True,
    dtype="float64",
    base_grid=decades(-4, 4),
    limits={"C": (1e-6, 1e6), "alpha": (1e-6, 1e9)},
    plateau_eps=1e-4,
    tol=1e-6,
    max_iter=5000,
    threads=1,
)


def assert_threads(protocol: Protocol) -> None:
    """Refuse to fit unless every thread pool runs at ``protocol.threads``.

    Thread count changes lbfgs results (measurements_2026-09.md §4). The pools
    read their size when the library loads, so the env vars must be set before
    Python starts.
    """
    if protocol.threads is None:
        return
    bad = [p for p in threadpoolctl.threadpool_info() if p.get("num_threads") != protocol.threads]
    if bad:
        pools = ", ".join(f"{p.get('internal_api')}={p.get('num_threads')}" for p in bad)
        n = protocol.threads
        raise ThreadPinError(
            f"thread pools not pinned to {n} thread(s): {pools}. Set "
            f"OPENBLAS_NUM_THREADS={n} OMP_NUM_THREADS={n} MKL_NUM_THREADS={n} before Python starts.")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


@lru_cache(maxsize=1)
def _static_stamp() -> dict:
    import numpy
    import scipy
    import sklearn

    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in
                   Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "git_sha": _git("rev-parse", "HEAD") or None,
        # Code only, untracked modules included; metrics files appended during a
        # run are not code and must not mark it dirty.
        "git_dirty": bool(_git("status", "--porcelain", "--", *CODE_PATHS)),
        "machine": {"arch": platform.machine(), "system": platform.system(), "cpu": cpu},
        "versions": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }


def stamp() -> dict:
    """Provenance for one record: code, machine, library versions and thread pools."""
    pools = threadpoolctl.threadpool_info()
    counts = sorted({p.get("num_threads") for p in pools})
    return {
        **_static_stamp(),
        "threads": counts[0] if len(counts) == 1 else counts,
        "blas": sorted({f"{p.get('internal_api')} {p.get('version')}" for p in pools}),
    }
