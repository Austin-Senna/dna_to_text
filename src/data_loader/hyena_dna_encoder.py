"""HyenaDNA loader for frozen CDS embedding extraction."""
from __future__ import annotations

import warnings

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "LongSafari/hyenadna-large-1m-seqlen-hf"


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device).eval()
    return _HiddenStateWrapper(model).eval(), tokenizer, device
