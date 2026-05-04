"""
utils/hf_client.py
==================
HuggingFace in-process client for Qwen2.5-VL / Qwen3-VL.

Drop-in replacement for VLLMClient on single-GPU nodes where a separate
vLLM server cannot run alongside the main process.

Same public interface:
    chat(messages, image_b64, system_prompt)        -> (str, int)
    chat_with_logprobs(messages, image_b64, ...)    -> ChatResult
    score_labels(prompt_prefix, label_list, image_b64) -> Dict[str, float]
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from utils.vllm_client import ChatResult

logger = logging.getLogger(__name__)

# Global model cache — one load per process
_hf_model = None
_hf_processor = None
_hf_model_name: Optional[str] = None


def _load_model(model_name: str):
    global _hf_model, _hf_processor, _hf_model_name
    if _hf_model is not None and _hf_model_name == model_name:
        return _hf_model, _hf_processor

    logger.info("Loading %s (this takes ~1-2 min on first call)...", model_name)
    import torch
    from transformers import AutoProcessor

    # Only use Qwen2VLForConditionalGeneration for actual Qwen2/2.5-VL checkpoints.
    # For anything else (Qwen3-VL, other architectures) fall through to AutoModelForCausalLM
    # so we never load a non-Qwen model into the wrong class.
    _is_qwen2vl = "Qwen2" in model_name or "Qwen2.5" in model_name
    if _is_qwen2vl:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"Could not load {model_name} as Qwen2VLForConditionalGeneration: {e}"
            ) from e
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _hf_model = model
    _hf_processor = processor
    _hf_model_name = model_name
    logger.info("Model loaded: %s", model_name)
    return _hf_model, _hf_processor


def _b64_to_pil(image_b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _build_qwen_messages(
    messages: List[Dict],
    image_b64: Optional[str],
    system_prompt: Optional[str],
) -> Tuple[List[Dict], List[Any]]:
    """
    Convert OpenAI-style messages + image_b64 into Qwen VL message dicts
    and a flat list of PIL images (passed separately to the processor).
    """
    from PIL import Image as PILImage

    out_msgs: List[Dict] = []
    pil_images: List[PILImage.Image] = []

    if system_prompt:
        out_msgs.append({"role": "system", "content": system_prompt})

    image_inserted = False
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and image_b64 is not None and not image_inserted:
            pil_img = _b64_to_pil(image_b64)
            pil_images.append(pil_img)
            image_inserted = True

            # Flatten content: strip existing image_url items, keep text
            text_parts: List[str] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        # skip image_url items — we insert a fresh PIL image instead
                    elif isinstance(item, str):
                        text_parts.append(item)
            combined_text = "\n".join(t for t in text_parts if t)

            new_content = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": combined_text},
            ]
            out_msgs.append({"role": role, "content": new_content})
        else:
            # Non-image message or assistant turn
            if isinstance(content, str):
                out_msgs.append({"role": role, "content": content})
            else:
                # Flatten to plain text for non-first-user turns
                parts: List[str] = []
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            parts.append(item)
                out_msgs.append({"role": role, "content": "\n".join(parts)})

    return out_msgs, pil_images


class HFClient:
    """
    In-process HuggingFace client for Qwen2.5-VL / Qwen3-VL.

    Loads the model once on first use and caches it globally.
    No HTTP server required — runs entirely on the local GPU.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.0,
        seed: int = 42,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        # Compatibility flags (mirrors VLLMClient attrs used by agents)
        self.chat_request_logprobs: bool = True
        self.guided_scoring_enabled: bool = True
        self.top_logprobs: int = 20
        # Pre-load
        _load_model(model)

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, int]:
        r = self.chat_with_logprobs(messages, image_b64, system_prompt)
        return r.text, r.completion_tokens

    def chat_with_logprobs(
        self,
        messages: List[Dict],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResult:
        import torch

        model, processor = _load_model(self.model)
        qwen_msgs, pil_images = _build_qwen_messages(messages, image_b64, system_prompt)

        text_prompt = processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}

        torch.manual_seed(self.seed)
        do_sample = self.temperature > 0.0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=self.chat_request_logprobs,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0][input_len:]
        response_text = processor.decode(gen_ids, skip_special_tokens=True)
        completion_tokens = int(gen_ids.shape[0])

        content_logprobs: Optional[List[Dict[str, Any]]] = None
        token_strings: List[str] = []

        if self.chat_request_logprobs and out.scores:
            content_logprobs = []
            for step_idx, step_scores in enumerate(out.scores):
                lp = torch.log_softmax(step_scores[0].float(), dim=-1)
                tok_id = int(gen_ids[step_idx])
                tok_str = processor.tokenizer.decode([tok_id])
                token_strings.append(tok_str)

                top_k = min(self.top_logprobs, lp.shape[-1])
                top_vals, top_ids = torch.topk(lp, top_k)
                top_lp = [
                    {"token": processor.tokenizer.decode([int(tid)]), "logprob": float(tv)}
                    for tid, tv in zip(top_ids.tolist(), top_vals.tolist())
                ]
                content_logprobs.append({
                    "token": tok_str,
                    "logprob": float(lp[tok_id]),
                    "top_logprobs": top_lp,
                })

        return ChatResult(
            text=response_text,
            completion_tokens=completion_tokens,
            content_logprobs=content_logprobs,
            token_strings=token_strings,
        )

    # ------------------------------------------------------------------
    # Constrained label scoring (Appendix B / Eq. 2)
    # ------------------------------------------------------------------

    def score_labels(
        self,
        prompt_prefix: str,
        label_list: List[str],
        image_b64: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute label probabilities by comparing each label's first-token
        logit at the next-token position after the prompt.
        """
        if not label_list:
            return {}
        if not self.guided_scoring_enabled:
            u = 1.0 / len(label_list)
            return {lbl: u for lbl in label_list}

        import torch

        model, processor = _load_model(self.model)

        msgs = [{"role": "user", "content": prompt_prefix}]
        qwen_msgs, pil_images = _build_qwen_messages(msgs, image_b64, None)
        text_prompt = processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}

        with torch.no_grad():
            out = model(**inputs)

        # Logits at the last prompt position → next-token distribution
        next_logits = out.logits[0, -1, :].float()

        # Score each label by its first token's logit
        log_scores: List[float] = []
        for lbl in label_list:
            # Try with a leading space (common tokenizer convention)
            toks = processor.tokenizer.encode(" " + str(lbl), add_special_tokens=False)
            if not toks:
                toks = processor.tokenizer.encode(str(lbl), add_special_tokens=False)
            first_tok = toks[0] if toks else 0
            log_scores.append(float(next_logits[first_tok]))

        max_v = max(log_scores)
        exps = [math.exp(v - max_v) for v in log_scores]
        total = sum(exps)
        if total <= 0:
            u = 1.0 / len(label_list)
            return {lbl: u for lbl in label_list}
        return {lbl: e / total for lbl, e in zip(label_list, exps)}

    # ------------------------------------------------------------------
    # Misc helpers (VLLMClient compat)
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(text.split())
