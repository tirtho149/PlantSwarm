"""
utils/vllm_inproc.py
====================
In-process Qwen2.5-VL backend for the Phase 0R swarm — **transformers**,
not vLLM.

Why transformers and not vLLM
-----------------------------
vLLM (server OR in-process EngineCore) repeatedly failed on Nova:
first ``HTTPError 400`` over the socket, then ``CUDA unknown error ...
Setting the available devices to be zero`` because vLLM v1 spawns an
EngineCore subprocess that could not initialise CUDA, retry-storming
forever. Phase 0R is a single-GPU, single-node job — it does not need
a serving engine. Plain HuggingFace ``transformers`` just loads the
model in THIS process and calls ``model.generate``: no server, no
port, no subprocess, no spawn, no CUDA-init race.

This mirrors the canonical transformers recipe:

    tok   = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(...)
    text  = tok.apply_chat_template(messages, add_generation_prompt=True, ...)
    out   = model.generate(**inputs, max_new_tokens=...)
    print(tok.decode(out[0][inputs["input_ids"].shape[-1]:]))

adapted to the VISION model: the swarm sends an image + text, so it
uses ``AutoProcessor`` (image + text) instead of a text-only tokenizer.

API surface (unchanged — duck-types the old vLLM client)
--------------------------------------------------------
  - ``chat(messages, system_prompt=, seed=, temperature=) -> (text, tokens)``
  - ``chat_with_logprobs(...) -> ChatResult``  (logprobs disabled)
  - ``warmup()``      build the model now, on the calling (main) thread
  - ``count_tokens(text)``
  - ``get_inproc_client()``  process-wide singleton from env

Concurrency
-----------
``model.generate`` is not safe to call concurrently on one model, so
every call is serialised through ``_engine_lock``. The swarm's
ThreadPoolExecutor still fans out, but the threads queue through the
lock. The model is built ONCE, on the main thread, via ``warmup()``
(``plantswarm.delta_pipeline`` calls it before any pool spawns).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Shape-compatible with the old utils.vllm_client.ChatResult."""

    text: str
    completion_tokens: int
    content_logprobs: Optional[List[Dict[str, Any]]] = None
    token_strings: List[str] = field(default_factory=list)


class InProcessVLLMClient:
    """Qwen2.5-VL run in-process via HuggingFace transformers.

    Name kept as ``InProcessVLLMClient`` so existing imports / type
    hints in ``plantswarm.delta_pipeline`` and ``agents`` do not change.
    """

    # Shared across all instances + threads in this process.
    _model = None
    _processor = None
    _engine_lock = threading.Lock()      # serialises model.generate
    _init_lock = threading.Lock()
    _engine_init_error: Optional[BaseException] = None

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.8,
        seed: int = 42,
        max_new_tokens: int = 512,
        max_model_len: int = 32768,          # kept for API compat
        min_image_pixels: int = 50176,
        max_image_pixels: int = 1003520,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,  # vLLM-only; ignored here
        **_ignored: Any,
    ):
        self.model_id = model
        self.temperature = temperature
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.dtype = dtype

        # No-op flags expected by old callers.
        self.chat_request_logprobs: bool = False
        self.prefer_structured_outputs: bool = False
        self.guided_scoring_enabled: bool = False
        self.top_logprobs: int = 0

    # ------------------------------------------------------------------
    # Model load (once, fail-fast)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Build the model NOW on the calling thread. The pipeline calls
        this on the MAIN thread before any ThreadPoolExecutor."""
        self._ensure_model()

    def _ensure_model(self):
        if InProcessVLLMClient._model is not None:
            return InProcessVLLMClient._model, InProcessVLLMClient._processor
        if InProcessVLLMClient._engine_init_error is not None:
            raise InProcessVLLMClient._engine_init_error
        with InProcessVLLMClient._init_lock:
            if InProcessVLLMClient._model is not None:
                return (InProcessVLLMClient._model,
                        InProcessVLLMClient._processor)
            if InProcessVLLMClient._engine_init_error is not None:
                raise InProcessVLLMClient._engine_init_error
            try:
                import torch  # noqa: F401
                from transformers import AutoProcessor

                # Qwen2.5-VL class name varies across transformers
                # versions; prefer the generic VL auto-class, fall back.
                try:
                    from transformers import AutoModelForImageTextToText \
                        as _VLModel
                except ImportError:  # older transformers
                    from transformers import (  # type: ignore
                        Qwen2_5_VLForConditionalGeneration as _VLModel)

                td = {
                    "auto": "auto", "bfloat16": torch.bfloat16,
                    "float16": torch.float16, "float32": torch.float32,
                }.get(str(self.dtype).lower(), "auto")

                logger.info(
                    "[hf_inproc] loading %s (dtype=%s, pixels %d..%d) "
                    "— transformers, in-process, no vLLM",
                    self.model_id, self.dtype,
                    self.min_image_pixels, self.max_image_pixels,
                )
                processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=self.min_image_pixels,
                    max_pixels=self.max_image_pixels,
                    trust_remote_code=True,
                )
                model = _VLModel.from_pretrained(
                    self.model_id,
                    torch_dtype=td,
                    device_map="cuda",
                    trust_remote_code=True,
                )
                model.eval()
            except BaseException as e:  # noqa: BLE001
                InProcessVLLMClient._engine_init_error = e
                logger.error("[hf_inproc] model load FAILED (cached, "
                             "fail-fast): %s: %s", type(e).__name__, e)
                raise
            InProcessVLLMClient._model = model
            InProcessVLLMClient._processor = processor
            logger.info("[hf_inproc] model ready on %s", model.device)
            return model, processor

    # ------------------------------------------------------------------
    # Message / image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _data_url_to_pil(url: str):
        from PIL import Image
        m = re.match(r"data:image/[^;]+;base64,(.+)$", url, re.DOTALL)
        if m:
            raw = base64.b64decode(m.group(1))
            return Image.open(io.BytesIO(raw)).convert("RGB")
        # bare base64 or local path fallback
        if os.path.exists(url):
            return Image.open(url).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unrecognised image url: {url[:48]}...") from e

    def _to_qwen(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        image_b64: Optional[str],
    ):
        """Return (qwen_messages, [PIL images in order]). Converts the
        swarm's OpenAI-style content blocks to Qwen2.5-VL format."""
        imgs: List[Any] = []
        out: List[Dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})

        msgs = [dict(m) for m in messages]
        if image_b64 is not None:
            for m in msgs:
                if m.get("role") == "user":
                    orig = m.get("content")
                    if isinstance(orig, str):
                        orig = [{"type": "text", "text": orig}]
                    m["content"] = [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"},
                    }] + list(orig or [])
                    break

        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            new_content: List[Dict[str, Any]] = []
            for block in content or []:
                btype = block.get("type")
                if btype == "image_url":
                    pil = self._data_url_to_pil(
                        (block.get("image_url") or {}).get("url", ""))
                    imgs.append(pil)
                    new_content.append({"type": "image", "image": pil})
                elif btype in ("image", "image_pil"):
                    pil = block.get("image") or block.get("image_pil")
                    imgs.append(pil)
                    new_content.append({"type": "image", "image": pil})
                else:
                    new_content.append(
                        {"type": "text", "text": block.get("text", "")})
            out.append({"role": role, "content": new_content})
        return out, imgs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        r = self.chat_with_logprobs(
            messages=messages, image_b64=image_b64,
            system_prompt=system_prompt, seed=seed,
            temperature=temperature, max_new_tokens=max_new_tokens)
        return r.text, r.completion_tokens

    def chat_with_logprobs(
        self,
        messages: List[Dict[str, Any]],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> ChatResult:
        import torch

        model, processor = self._ensure_model()
        qmsgs, imgs = self._to_qwen(messages, system_prompt, image_b64)

        text = processor.apply_chat_template(
            qmsgs, tokenize=False, add_generation_prompt=True)
        proc_kwargs: Dict[str, Any] = dict(
            text=[text], return_tensors="pt", padding=True)
        if imgs:
            proc_kwargs["images"] = imgs

        temp = self.temperature if temperature is None else float(temperature)
        sd = self.seed if seed is None else int(seed)

        # generate() is not concurrency-safe on one model — serialise.
        with InProcessVLLMClient._engine_lock:
            inputs = processor(**proc_kwargs).to(model.device)
            torch.manual_seed(sd)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sd)
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=(max_new_tokens
                                if max_new_tokens is not None
                                else self.max_new_tokens))
            if temp and temp > 0.0:
                gen_kwargs.update(do_sample=True, temperature=temp,
                                  top_p=0.9)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.inference_mode():
                out_ids = model.generate(**inputs, **gen_kwargs)
            in_len = inputs["input_ids"].shape[-1]
            trimmed = out_ids[0][in_len:]
            n_tok = int(trimmed.shape[-1])
            txt = processor.batch_decode(
                trimmed.unsqueeze(0), skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

        return ChatResult(text=(txt or "").strip(), completion_tokens=n_tok)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[InProcessVLLMClient] = None
_GLOBAL_LOCK = threading.Lock()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_inproc_client() -> InProcessVLLMClient:
    """Process-wide singleton, configured from env. Env knob names are
    kept (``VLLM_*``) for sbatch/doc continuity even though the backend
    is now transformers; ``VLLM_GPU_MEMORY_UTIL`` is ignored."""
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            return _GLOBAL
        _GLOBAL = InProcessVLLMClient(
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            temperature=_float_env("VLLM_TEMPERATURE", 0.8),
            max_new_tokens=_int_env("VLLM_MAX_NEW_TOKENS", 512),
            max_model_len=_int_env("VLLM_MAX_MODEL_LEN", 32768),
            min_image_pixels=_int_env("VLLM_MIN_PIXELS", 50176),
            max_image_pixels=_int_env("VLLM_MAX_PIXELS", 1003520),
            dtype=os.environ.get("VLLM_DTYPE", "auto"),
        )
        return _GLOBAL
