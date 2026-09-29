"""Refuse a model whose checkpoint did not fill every weight (1C).

``load_state_dict(strict=False)`` and ``from_pretrained`` both report missing
and unexpected keys and then carry on: a missing key leaves a randomly
initialised layer inside a "pretrained" encoder, and nothing downstream can
tell. These checks turn that report into an error. Unexpected keys are allowed
only by explicit prefix (heads the frozen encoder does not use).
"""
from __future__ import annotations

from typing import Iterable


class LoadError(RuntimeError):
    """A checkpoint left weights missing or carried weights nobody expected."""


def _check(missing: Iterable[str], unexpected: Iterable[str], what: str,
           allowed_unexpected: tuple[str, ...], extra: Iterable[str] = ()) -> None:
    missing = sorted(missing)
    stray = sorted(k for k in unexpected if not k.startswith(allowed_unexpected))
    extra = [e for e in extra if e]
    if missing or stray or extra:
        raise LoadError(f"{what}: missing {missing[:8]} ({len(missing)}), "
                        f"unexpected {stray[:8]} ({len(stray)}) {extra[:3]}")


def check_state_dict_load(result, *, what: str, allowed_unexpected: tuple[str, ...] = ()) -> None:
    """Check the value returned by ``module.load_state_dict(..., strict=False)``."""
    _check(result.missing_keys, result.unexpected_keys, what, allowed_unexpected)


def check_loading_info(info: dict, *, what: str, allowed_unexpected: tuple[str, ...] = ()) -> None:
    """Check the ``loading_info`` dict from ``from_pretrained(..., output_loading_info=True)``."""
    _check(info.get("missing_keys", ()), info.get("unexpected_keys", ()), what,
           allowed_unexpected,
           extra=[*map(str, info.get("mismatched_keys", ())), *info.get("error_msgs", ())])
