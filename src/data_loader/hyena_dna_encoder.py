"""HyenaDNA loader for frozen CDS embedding extraction.

HyenaDNA is fed DNA tokens only (``EncoderSpec.boundary_tokens`` is False): its
tokenizer declares [CLS]/[SEP] but the model never saw a CLS, and being causal,
a CLS at position 0 would reach every position (G3).
"""
from __future__ import annotations

import warnings

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader.load_checks import check_loading_info
from data_loader.model_registry import ENCODER_SPECS

MODEL_NAME = ENCODER_SPECS["hyena_dna"].model_name
MODEL_REVISION = ENCODER_SPECS["hyena_dna"].revision


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _HiddenStateWrapper(nn.Module):
    """Expose last hidden states from HyenaDNA's causal-LM output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="An output with one or more elements was resized.*")
            warnings.filterwarnings("ignore", message="`use_return_dict` is deprecated.*")
            try:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            except TypeError:
                out = self.model(input_ids=input_ids, output_hidden_states=True)
        return type("Out", (), {"last_hidden_state": out.hidden_states[-1]})


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                              trust_remote_code=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True, output_loading_info=True)
    check_loading_info(info, what=f"{MODEL_NAME}@{MODEL_REVISION[:8]}")
    model.to(device).eval()
    return _HiddenStateWrapper(model).eval(), tokenizer, device
