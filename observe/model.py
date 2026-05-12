"""
observe/model.py
================
OBSERVE Vision-Language-Action model architecture.

Paper §7 (pathome_final). Fine-tuned from Qwen2.5-VL-7B with LoRA:
- Per-step visual grounding (image present at every step)
- Inputs: image X, context buffer C_t, decision-graph node G_t,
  GPS-derived phi_geo, top-3 geo-weighted PathomeDB references Ref_{1:3}
- Outputs: routing softmax (5 classes), backtrack b_t (binary),
  epistemic eps_t, aleatoric alpha_t, calibrated confidence c_t,
  overconfidence flag OC_t, autoregressive belief s_t
- LoRA r=16, alpha=32, target {q,k,v,o}_proj, ~56M trainable on 7B frozen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class EpistemicAction:
    """Structured action output from OBSERVE (paper Eq. action)."""
    next_agent: str  # 5-class: MorphologyAgent, SymptomAgent, PathogenAgent, SeverityAgent, DiagnosisAgent
    backtrack: bool  # b_t: whether to backtrack to MorphologyAgent
    epistemic_uncertainty: float  # eps_t ∈ [0, 1]: resolvable ambiguity → more evidence helps
    aleatoric_uncertainty: float  # alpha_t ∈ [0, 1]: irreducible difficulty → escalate to human
    confidence: float  # c_t ∈ [0, 1]: calibrated confidence in prediction
    overconfidence: bool  # OC_t: agent claimed kappa=H but visual evidence weak (Eq. oc)
    belief_state: str  # s_t: natural language belief about current situation


class OBSERVE(nn.Module):
    """
    Vision-Language-Action model for epistemic action selection (paper §7).

    Architecture:
    - Backbone: Qwen2.5-VL-7B (frozen, ~7B params)
    - LoRA: r=16, alpha=32, applied to q/k/v/o_proj (~50M trainable)
    - Heads (all on shared 512-dim representation):
        * routing (5-class softmax)
        * backtrack b_t (binary sigmoid)
        * epistemic eps_t (scalar [0,1])
        * aleatoric alpha_t (scalar [0,1])
        * confidence c_t (scalar [0,1])
        * overconfidence OC_t (binary sigmoid)        [NEW in pathome_final]
        * belief text s_t (autoregressive via LM head)
    - Total trainable: ~56M / 7B (~0.8%)
    """

    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        agent_classes: Optional[list] = None,
        oc_threshold: float = 0.55,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.agent_classes = agent_classes or [
            "MorphologyAgent", "SymptomAgent", "PathogenAgent",
            "SeverityAgent", "DiagnosisAgent"
        ]

        # Load base model and processor
        self.processor = AutoProcessor.from_pretrained(backbone)
        self.model = AutoModelForVision2Seq.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Get hidden dimension from model
        hidden_dim = self.model.config.hidden_size  # Usually 2048 for Qwen2.5-VL-3B

        # Apply LoRA to vision encoder and language model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Shared representation head
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.routing_head = nn.Linear(512, len(self.agent_classes))  # 5-class softmax
        self.backtrack_head = nn.Linear(512, 1)  # b_t: binary sigmoid
        self.epistemic_head = nn.Linear(512, 1)  # eps_t: sigmoid [0, 1]
        self.aleatoric_head = nn.Linear(512, 1)  # alpha_t: sigmoid [0, 1]
        self.confidence_head = nn.Linear(512, 1)  # c_t: sigmoid [0, 1]
        self.oc_head = nn.Linear(512, 1)  # OC_t: sigmoid [0, 1]  (Eq. oc, paper §7.2)

        # Decision threshold for overconfidence flag (paper §7.2: tau_OC = 0.55)
        self.oc_threshold = oc_threshold

        # Belief text autoregressive head (uses model's decoder)
        # Belief is generated via model decoder, not a separate head

    def forward(
        self,
        image: torch.Tensor,
        context_text: str,
        return_dict: bool = True,
    ) -> dict | tuple:
        """
        Forward pass for epistemic action selection.

        Args:
            image: Input image tensor (after preprocessing)
            context_text: Prior agent messages and context
            return_dict: Return as dict (True) or tuple

        Returns:
            Dict with keys: next_agent, backtrack, epistemic, aleatoric, confidence, belief
        """

        # Prepare input: image + context text
        prompt = f"Prior context:\n{context_text}\n\nBased on this image and context, what is your next action?"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward through model to get hidden states
        with torch.no_grad():
            # Get embeddings/hidden states from model
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Use last hidden state as representation
        hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        pooled = hidden_state.mean(dim=1)  # [batch, hidden_dim]

        # Pass through shared head
        shared_repr = self.shared_head(pooled)  # [batch, 512]

        # Compute action outputs
        routing_logits = self.routing_head(shared_repr)  # [batch, 5]
        backtrack_logits = self.backtrack_head(shared_repr)  # [batch, 1]
        epistemic_logits = self.epistemic_head(shared_repr)  # [batch, 1]
        aleatoric_logits = self.aleatoric_head(shared_repr)  # [batch, 1]
        confidence_logits = self.confidence_head(shared_repr)  # [batch, 1]
        oc_logits = self.oc_head(shared_repr)  # [batch, 1]

        # Apply activations
        routing_probs = torch.softmax(routing_logits, dim=-1)  # [batch, 5]
        backtrack_prob = torch.sigmoid(backtrack_logits).squeeze(-1)  # [batch]
        epistemic = torch.sigmoid(epistemic_logits).squeeze(-1)  # [batch]
        aleatoric = torch.sigmoid(aleatoric_logits).squeeze(-1)  # [batch]
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)  # [batch]
        oc_prob = torch.sigmoid(oc_logits).squeeze(-1)  # [batch]

        # Generate belief text (autoregressive from decoder)
        belief_prompt = f"My belief state is: "
        belief_inputs = self.processor.tokenizer(
            belief_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        belief_outputs = self.model.generate(
            **belief_inputs,
            max_new_tokens=50,
            temperature=0.7,
        )
        belief_text = self.processor.batch_decode(
            belief_outputs,
            skip_special_tokens=True,
        )[0]

        if return_dict:
            return {
                "routing_probs": routing_probs,
                "routing_class": self.agent_classes[routing_probs.argmax(dim=-1).item()],
                "backtrack_prob": backtrack_prob,
                "epistemic": epistemic,
                "aleatoric": aleatoric,
                "confidence": confidence,
                "oc_prob": oc_prob,
                "belief_text": belief_text,
            }
        else:
            return (
                routing_probs, backtrack_prob, epistemic, aleatoric,
                confidence, oc_prob, belief_text,
            )

    def get_epistemic_action(
        self,
        image: torch.Tensor,
        context_text: str,
        backtrack_threshold: float = 0.5,
    ) -> EpistemicAction:
        """Get a single EpistemicAction from image and context (paper §7.1)."""
        outputs = self.forward(image, context_text, return_dict=True)

        return EpistemicAction(
            next_agent=outputs["routing_class"],
            backtrack=outputs["backtrack_prob"].item() > backtrack_threshold,
            epistemic_uncertainty=outputs["epistemic"].item(),
            aleatoric_uncertainty=outputs["aleatoric"].item(),
            confidence=outputs["confidence"].item(),
            overconfidence=outputs["oc_prob"].item() > self.oc_threshold,
            belief_state=outputs["belief_text"],
        )
