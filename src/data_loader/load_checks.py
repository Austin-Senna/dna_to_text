"""Refuse a model whose checkpoint did not fill every weight (1C).

``load_state_dict(strict=False)`` and ``from_pretrained`` both report missing
and unexpected keys and then carry on: a missing key leaves a randomly
initialised layer inside a "pretrained" encoder, and nothing downstream can
tell. These checks turn that report into an error. Unexpected keys are allowed
only by explicit prefix (heads the frozen encoder does not use).

A clean report is not enough. Under transformers 5.5.4, ``from_pretrained``
reported every GENA-LM key as loaded while leaving all 197 parameters at their
random init, a new draw in every process (found by the Phase 3 pilot, Sept 29).
``check_weights_match_checkpoint`` compares each parameter with the tensor in
the checkpoint file itself.
"""
from __future__ import annotations

from pathlib import Path
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


def read_checkpoint(snapshot_dir: str | Path) -> dict:
    """Every tensor in a hub snapshot's weights file (safetensors preferred)."""
    import torch
    from safetensors.torch import load_file

    snapshot_dir = Path(snapshot_dir)
    files = sorted(snapshot_dir.glob("*.safetensors")) or sorted(snapshot_dir.glob("*.bin"))
    if not files:
        raise LoadError(f"{snapshot_dir}: no weights file")
    out: dict = {}
    for f in files:
        out.update(load_file(f) if f.suffix == ".safetensors"
                   else torch.load(f, map_location="cpu", weights_only=True))
    return out


def check_weights_match_checkpoint(model, checkpoint: dict, *, what: str) -> None:
    """Every parameter of ``model`` equals its tensor in ``checkpoint``.

    Names are matched up to a leading prefix (a backbone taken out of its
    masked-LM wrapper drops ``bert.``; a wrapper may add one); each parameter
    must match exactly one checkpoint tensor.
    """
    import torch

    unmatched, differ = [], []
    for name, p in model.named_parameters():
        hits = [k for k in checkpoint if k == name or k.endswith("." + name) or name.endswith("." + k)]
        if len(hits) != 1:
            unmatched.append(f"{name} ({len(hits)} candidates)")
            continue
        ref = checkpoint[hits[0]]
        if p.shape != ref.shape or not torch.equal(p.detach().cpu().to(ref.dtype), ref):
            differ.append(name)
    if unmatched or differ:
        raise LoadError(f"{what}: {len(differ)} parameters differ from the checkpoint {differ[:5]}; "
                        f"{len(unmatched)} have no checkpoint tensor {unmatched[:5]}")
