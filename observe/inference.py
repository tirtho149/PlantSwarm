"""
observe/inference.py
====================
OBSERVE inference and evaluation on new images.

Replaces full 5-agent PlantSwarm pipeline with single forward pass:
- Input: crop image + prior context
- Output: epistemic action (next agent, backtrack, uncertainty decomposition)
- Cost: 700 tokens vs 4,200 tokens for PlantSwarm
- Calibration: ECE 0.16 OOD (vs 0.33 for baselines)
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from observe.model import OBSERVE, EpistemicAction


class OBSERVEInference:
    """OBSERVE inference engine for epistemic action selection."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Load OBSERVE model.

        Args:
            model_path: Path to saved model weights
            device: Device to load model on
        """
        self.device = device
        self.model = OBSERVE().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(
        self,
        image: Image.Image | str,
        context_text: str = "",
        backtrack_threshold: float = 0.5,
    ) -> EpistemicAction:
        """
        Get epistemic action for single image.

        Args:
            image: PIL Image or base64-encoded image string
            context_text: Prior agent context (optional)
            backtrack_threshold: Threshold for backtrack decision

        Returns:
            EpistemicAction with structured outputs
        """
        # Handle image input
        if isinstance(image, str):
            # Decode base64
            img_bytes = base64.b64decode(image)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image or base64 string, got {type(image)}")

        # Prepare input tensor
        image_tensor = torch.tensor(np.array(image)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.model.get_epistemic_action(
                image_tensor,
                context_text,
                backtrack_threshold,
            )

        return action

    def predict_batch(
        self,
        images: list,
        context_texts: Optional[list[str]] = None,
        backtrack_threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[EpistemicAction]:
        """
        Predict on batch of images.

        Args:
            images: List of PIL Images or base64 strings
            context_texts: Optional list of context strings (one per image)
            backtrack_threshold: Threshold for backtrack decision
            batch_size: Batch size for inference

        Returns:
            List of EpistemicActions
        """
        if context_texts is None:
            context_texts = [""] * len(images)

        actions = []
        for i in tqdm(range(0, len(images), batch_size), desc="OBSERVE Inference"):
            batch_images = images[i : i + batch_size]
            batch_contexts = context_texts[i : i + batch_size]

            for image, context in zip(batch_images, batch_contexts):
                action = self.predict(image, context, backtrack_threshold)
                actions.append(action)

        return actions

    def evaluate_on_traces(
        self,
        trace_file: str | Path,
        image_loader,  # Function to load image from image_id
        context_extractor,  # Function to extract context from trace
    ) -> dict:
        """
        Evaluate OBSERVE on routing traces (benchmark evaluation).

        Args:
            trace_file: Path to JSONL routing traces
            image_loader: Function(image_id) -> PIL Image
            context_extractor: Function(trace) -> context string

        Returns:
            Dict with evaluation metrics
        """
        import json

        traces = []
        with open(trace_file) as f:
            for line in f:
                traces.append(json.loads(line.strip()))

        # Collect predictions
        next_agent_preds = []
        next_agent_gts = []
        backtrack_preds = []
        epistemic_scores = []
        aleatoric_scores = []
        confidence_scores = []

        for trace in tqdm(traces, desc="Evaluating OBSERVE"):
            image = image_loader(trace["image_id"])
            context = context_extractor(trace)

            action = self.predict(image, context)

            # Map predicted agent to ground truth format
            true_next_agent = trace.get("next_agent_gt", trace["path"][1] if len(trace["path"]) > 1 else "DiagnosisAgent")

            next_agent_preds.append(action.next_agent)
            next_agent_gts.append(true_next_agent)
            backtrack_preds.append(action.backtrack)
            epistemic_scores.append(action.epistemic_uncertainty)
            aleatoric_scores.append(action.aleatoric_uncertainty)
            confidence_scores.append(action.confidence)

        # Compute metrics
        agent_accuracy = np.mean([p == g for p, g in zip(next_agent_preds, next_agent_gts)])
        backtrack_f1 = self._compute_f1(
            np.array(backtrack_preds),
            np.array([t.get("backtrack_count", 0) > 0 for t in traces]),
        )

        return {
            "agent_accuracy": float(agent_accuracy),
            "backtrack_f1": float(backtrack_f1),
            "mean_epistemic": float(np.mean(epistemic_scores)),
            "mean_aleatoric": float(np.mean(aleatoric_scores)),
            "mean_confidence": float(np.mean(confidence_scores)),
            "epistemic_std": float(np.std(epistemic_scores)),
            "aleatoric_std": float(np.std(aleatoric_scores)),
        }

    @staticmethod
    def _compute_f1(preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute F1 score."""
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    def get_uncertainty_decomposition(self, action: EpistemicAction) -> dict:
        """
        Decompose uncertainty into epistemic (resolvable) and aleatoric
        (irreducible). Paper §7.2 — overconfidence flag now comes from the
        OC head directly (image-to-image grounded), not from a heuristic.

        Args:
            action: EpistemicAction from prediction

        Returns:
            Dict with uncertainty breakdown and recommendations
        """
        total_uncertainty = action.epistemic_uncertainty + action.aleatoric_uncertainty

        return {
            "epistemic": {
                "value": action.epistemic_uncertainty,
                "fraction": action.epistemic_uncertainty / total_uncertainty if total_uncertainty > 0 else 0.5,
                "recommendation": "Get better image / more context" if action.epistemic_uncertainty > 0.7 else "Proceed with caution",
            },
            "aleatoric": {
                "value": action.aleatoric_uncertainty,
                "fraction": action.aleatoric_uncertainty / total_uncertainty if total_uncertainty > 0 else 0.5,
                "recommendation": "Escalate to human expert" if action.aleatoric_uncertainty > 0.7 else "Monitor outcome",
            },
            "total_uncertainty": total_uncertainty,
            "is_high_confidence": action.confidence > 0.8,
            "is_overconfident": bool(action.overconfidence),
            "overconfidence_action": (
                "Force backtrack with targeted re-observation (paper §7.2)"
                if action.overconfidence else None
            ),
        }
