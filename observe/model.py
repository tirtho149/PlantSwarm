"""
observe/model.py
================
OBSERVE — distilled student over Qwen2.5-VL + LoRA.

After the Algorithm-1 routing removal, the model predicts only
uncertainty / confidence over a single-pass delta extraction. Five
heads remain on a shared 512-dim representation:

    epistemic       (scalar in [0,1])
    aleatoric       (scalar in [0,1])
    confidence      (scalar in [0,1]; calibrated κ)
    overconfidence  (binary; OC_t)

The routing and backtrack heads from the previous revision are gone:
the swarm no longer routes between agents (specialists run in
parallel), so the student has no routing decision to learn.

Forward returns RAW LOGITS (loss layer applies activations).
``get_uncertainty`` does the inference-side sigmoid + threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class EpistemicState:
    """Inference-time uncertainty estimate from OBSERVE."""
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    confidence: float
    overconfidence: bool


class OBSERVE(nn.Module):
    """Qwen2.5-VL + LoRA + 4 uncertainty heads."""

    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        oc_threshold: float = 0.55,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.oc_threshold = oc_threshold

        self.processor = AutoProcessor.from_pretrained(backbone)
        kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        self.model = AutoModelForVision2Seq.from_pretrained(backbone, **kwargs)

        hidden_dim = int(self.model.config.hidden_size)

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass

        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.epistemic_head  = nn.Linear(512, 1)
        self.aleatoric_head  = nn.Linear(512, 1)
        self.confidence_head = nn.Linear(512, 1)
        self.oc_head         = nn.Linear(512, 1)

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            **batch_inputs,
            output_hidden_states=True, return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = batch_inputs.get("attention_mask")
        if mask is not None:
            last_idx = mask.sum(dim=1).clamp(min=1) - 1
            arange = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[arange, last_idx]
        else:
            pooled = hidden.mean(dim=1)
        pooled = pooled.to(self.shared_head[0].weight.dtype)
        shared = self.shared_head(pooled)
        return {
            "epistemic_logit":  self.epistemic_head(shared).squeeze(-1),
            "aleatoric_logit":  self.aleatoric_head(shared).squeeze(-1),
            "confidence_logit": self.confidence_head(shared).squeeze(-1),
            "oc_logit":         self.oc_head(shared).squeeze(-1),
        }

    @torch.no_grad()
    def get_uncertainty(self, image, context_text: str) -> EpistemicState:
        self.eval()
        inputs = self.processor(
            images=image, text=context_text,
            return_tensors="pt", padding=True,
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        out = self.forward(inputs)
        return EpistemicState(
            epistemic_uncertainty=torch.sigmoid(out["epistemic_logit"]).item(),
            aleatoric_uncertainty=torch.sigmoid(out["aleatoric_logit"]).item(),
            confidence=torch.sigmoid(out["confidence_logit"]).item(),
            overconfidence=torch.sigmoid(out["oc_logit"]).item() > self.oc_threshold,
        )
