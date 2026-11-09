"""The selection-sensitive cells: every cell that differs from the canonical run
under either perturbation of the frozen protocol (Phase 4 gate, decided Oct 2:
accept and disclose, no post-hoc protocol change).

  threads  the main and null groups refitted at 6 threads (``data/v2/determinism_t6.json``)
  kernel   the Phase 5 clean room on another OpenBLAS kernel (``data/v2/determinism_kernel.json``)

Both inputs are ``scripts/diff_records.py`` reports against ``data/v2``. Each
listed cell keeps what moved under which perturbation (with every metric delta,
which the table builder replays to mark unstable digits); ``pick_changed`` marks a
C or alpha that moved under either, and ``max_abs_d_test_f1`` the larger test
macro-F1 move. The builder-level picks that moved are listed per perturbation, and
``inputs`` holds the sha256 of the canonical records files the deltas apply to.
Writes ``data/v2/selection_sensitive.json``.

Run: uv run scripts/build_selection_sensitive.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from linear_trainer import records as R


def _side(cell: dict | None) -> dict | None:
    if cell is None:
        return None
    deltas = cell.get("metric_deltas") or {}
    return {"what": cell["what"], "hp": [cell["hp_a"], cell["hp_b"]],
            "d_test_f1": deltas.get("test_macro_f1"), "metric_deltas": deltas}


def build(threads: dict, kernel: dict, inputs: dict[str, str]) -> dict:
    t = {c["key"]: c for c in threads["cells"]}
    k = {c["key"]: c for c in kernel["cells"]}
    cells = []
    for key in sorted(set(t) | set(k)):
        sides = {"threads": _side(t.get(key)), "kernel": _side(k.get(key))}
        f1 = [abs(s["d_test_f1"]) for s in sides.values() if s and s["d_test_f1"] is not None]
        ref = t.get(key) or k[key]
        cells.append({"key": key, "d": ref["d"], "n_train": ref["n_train"], **sides,
                      "pick_changed": any(s and "pick" in s["what"] for s in sides.values()),
                      "max_abs_d_test_f1": max(f1) if f1 else None})
    builder = {"threads": threads["picks"], "kernel": kernel["picks"]}
    return {
        "sources": {"threads": {s: threads["stamps"][s]["git_sha"] for s in "ab"},
                    "kernel": {s: kernel["stamps"][s]["git_sha"] for s in "ab"}},
        "inputs": inputs,
        "summary": {"cells": len(cells), "pick_changed": sum(c["pick_changed"] for c in cells),
                    "both_perturbations": sum(c["threads"] is not None and c["kernel"] is not None
                                              for c in cells),
                    "builder_picks": len(builder["threads"]) + len(builder["kernel"])},
        "builder_picks": builder,
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=Path, default=R.V2 / "determinism_t6.json")
    ap.add_argument("--kernel", type=Path, default=R.V2 / "determinism_kernel.json")
    ap.add_argument("--out", type=Path, default=R.V2 / "selection_sensitive.json")
    args = ap.parse_args()
    # Both diffs take data/v2 as side a: record its files, so a rewrite there invalidates the list.
    out = build(json.loads(args.threads.read_text()), json.loads(args.kernel.read_text()), R.input_digests())
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out.name}: {out['summary']}")


if __name__ == "__main__":
    main()
