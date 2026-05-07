"""Build the minimal COMS 4761 final-report submission zip.

The course prompt asks for report content in one compressed folder, with a
pointer to a GitHub repository containing the software files. This bundle keeps
the Courseworks upload small: final PDF plus a short Markdown README.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "coms4761_dna_to_text_report"
GITHUB_URL = "https://github.com/Austin-Senna/dna_to_text"


README = f"""# COMS 4761 Final Report: DNA-to-Text

This folder contains the Bioinformatics-style final report PDF for the project
"Linear probes reveal coding-sequence dependence in frozen DNA encoder representations."

## Files

- `main.pdf` - final report PDF.
- `README.md` - this file.

## Project Repository

All source code, scripts, tests, sample inputs/outputs, generated analysis
tables, generated figures, and reproduction notes are available in the GitHub
repository:

{GITHUB_URL}

Useful repository paths:

- `README.md` - setup, public data sources, and pipeline commands.
- `analysis/figures/` - generated main and supplementary figures.
- `analysis/tables/` - generated CSV and Markdown analysis tables.
- `demo/cross_modal.py` - small sample run script.
- `demo/output.md` - small sample generated output.

## Quick Run Commands

Install and run tests:

```bash
uv sync
uv run --with pytest pytest
```

Regenerate cached paper analysis tables and figures from tracked metrics:

```bash
uv run python scripts/build_analysis_artifacts.py --overwrite
```

The full encoder pipeline uses public model/data sources listed in the project
repository and may require GPU-class hardware for practical runtime.
"""


def build_bundle(out_root: Path, name: str) -> tuple[Path, Path]:
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(REPO_ROOT / "dna_to_text_paper" / "main.pdf", out_dir / "main.pdf")
    (out_dir / "README.md").write_text(README)

    archive_base = out_root / name
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=out_root, base_dir=name))
    return out_dir, archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(REPO_ROOT / "dist"))
    parser.add_argument("--name", default=DEFAULT_NAME)
    args = parser.parse_args()

    out_dir, archive = build_bundle(Path(args.out_root), args.name)
    print(f"wrote folder: {out_dir}")
    print(f"wrote archive: {archive}")


if __name__ == "__main__":
    main()
