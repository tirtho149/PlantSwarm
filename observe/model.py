"""
observe/model.py
================
OBSERVE Vision-Language-Action model — Qwen2.5-VL + LoRA + multi-task heads.

Architecture
------------
- Backbone: Qwen/Qwen2.5-VL-7B-Instruct (frozen base; hidden_size 3584)
- LoRA: r=16, alpha=32 on {q,k,v,o}_proj — ~56M trainable on 7B frozen
- Six heads on a shared 512-dim representation (last non-pad hidden):
    * routing       (5-class softmax over agent names)
    * backtrack     (binary)
    * epistemic     (scalar in [0,1])
    * aleatoric     (scalar in [0,1])
    * confidence    (scalar in [0,1]; calibrated kappa)
    * overconfidence(binary; OC_t)

The previous revision wrapped the backbone forward in ``torch.no_grad()``
which silently blocked LoRA gradients — fixed here.

Forward returns RAW LOGITS (not softmax/sigmoid). The loss layer applies
the right activation per head; ``get_epistemic_action`` applies them at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor


AGENT_CLASSES_DEFAULT: List[str] = [
    "MorphologyAgent", "SymptomAgent", "PathogenAgent",
    "SeverityAgent", "DiagnosisAgent",
]


@dataclass
class EpistemicAction:
    """Structured action output from OBSERVE at inference."""
    next_agent: str
    backtrack: bool
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    confidence: float
    overconfidence: bool


class OBSERVE(nn.Module):
    """Qwen2.5-VL + LoRA + 6 task heads, trained on Phase 0R routed traces."""

    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        agent_classes: Optional[List[str]] = None,
        oc_threshold: float = 0.55,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.agent_classes = agent_classes or AGENT_CLASSES_DEFAULT
        self.oc_threshold = oc_threshold

        self.processor = AutoProcessor.from_pretrained(backbone)
        kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        self.model = AutoModelForVision2Seq.from_pretrained(backbone, **kwargs)

        hidden_dim = int(self.model.config.hidden_size)        # 3584 for 7B

        # LoRA — only these weights train; base remains frozen.
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass

        # Shared trunk + task heads.
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.routing_head    = nn.Linear(512, len(self.agent_classes))
        self.backtrack_head  = nn.Linear(512, 1)
        self.epistemic_head  = nn.Linear(512, 1)
        self.aleatoric_head  = nn.Linear(512, 1)
        self.confidence_head = nn.Linear(512, 1)
        self.oc_head         = nn.Linear(512, 1)

    # ------------------------------------------------------------------
    # Training forward — takes pre-processed batch from the collator
    # ------------------------------------------------------------------

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return RAW logits for every head.

        ``batch_inputs`` is the dict the OBSERVE collator emits:
        ``pixel_values``, ``input_ids``, ``attention_mask`` (and
        optionally ``image_grid_thw`` for Qwen2.5-VL). Gradients flow
        through the LoRA adapters.
        """
        outputs = self.model(
            **batch_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]                  # [B, T, H]
        mask = batch_inputs.get("attention_mask")
        if mask is not None:
            last_idx = mask.sum(dim=1).clamp(min=1) - 1
            arange = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[arange, last_idx]
        else:
            pooled = hidden.mean(dim=1)
        pooled = pooled.to(self.shared_head[0].weight.dtype)
        shared = self.shared_head(pooled)                   # [B, 512]

        return {
            "routing_logits":   self.routing_head(shared),                  # [B, 5]
            "backtrack_logit":  self.backtrack_head(shared).squeeze(-1),    # [B]
            "epistemic_logit":  self.epistemic_head(shared).squeeze(-1),    # [B]
            "aleatoric_logit":  self.aleatoric_head(shared).squeeze(-1),    # [B]
            "confidence_logit": self.confidence_head(shared).squeeze(-1),   # [B]
            "oc_logit":         self.oc_head(shared).squeeze(-1),           # [B]
        }

    # ------------------------------------------------------------------
    # Inference — one (image, context_text) → one EpistemicAction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_epistemic_action(
        self,
        image,
        context_text: str,
        backtrack_threshold: float = 0.5,
    ) -> EpistemicAction:
        """Wrap one sample through the processor + forward + activations."""
        self.eval()
        inputs = self.processor(
            images=image, text=context_text,
            return_tensors="pt", padding=True,
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.forward(inputs)
        routing_probs = torch.softmax(out["routing_logits"], dim=-1)
        cls_idx = int(routing_probs.argmax(dim=-1).item())
        return EpistemicAction(
            next_agent=self.agent_classes[cls_idx],
            backtrack=torch.sigmoid(out["backtrack_logit"]).item() > backtrack_threshold,
            epistemic_uncertainty=torch.sigmoid(out["epistemic_logit"]).item(),
            aleatoric_uncertainty=torch.sigmoid(out["aleatoric_logit"]).item(),
            confidence=torch.sigmoid(out["confidence_logit"]).item(),
            overconfidence=torch.sigmoid(out["oc_logit"]).item() > self.oc_threshold,
        )
