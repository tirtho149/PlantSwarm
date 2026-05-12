"""
utils/vllm_client.py
====================
vLLM OpenAI-compatible client wrapper.

Inference uses vLLM's OpenAI-compatible HTTP API. Chat completions request
``logprobs`` when ``VLLMClient.chat_request_logprobs`` is True (entropy routing
requires this).

Appendix B label scoring uses ``structured_outputs.choice`` or legacy
``guided_choice``: vision-conditioned via ``/v1/chat/completions`` when an image
is supplied, otherwise ``/v1/completions`` on text-only prefixes.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Result of a chat completion, optionally with per-token logprobs."""

    text: str
    completion_tokens: int
    content_logprobs: Optional[List[Dict[str, Any]]] = None
    token_strings: List[str] = field(default_factory=list)


class VLLMClient:
    """
    Thin wrapper around vLLM's OpenAI-compatible HTTP API.

    Supports:
    - /v1/chat/completions  (multi-turn vision chat)
    - /v1/completions       (constrained-decoding scoring pass — Appendix B)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.0,
        seed: int = 42,
        max_new_tokens: int = 512,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.top_logprobs = 20
        # When False, chat requests omit logprobs (saves bandwidth; entropy_routing forces True).
        self.chat_request_logprobs: bool = True
        # Prefer vLLM ≥0.12 ``structured_outputs`` over deprecated ``guided_choice`` on /completions.
        self.prefer_structured_outputs: bool = True
        # If True, run constrained label scoring; if False, return uniform distributions (debug only).
        self.guided_scoring_enabled: bool = True

    # ------------------------------------------------------------------
    # Chat completions (agent inference pass)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int]:
        """Send a chat request. Returns (response_text, tokens_used).

        ``seed`` and ``temperature`` override the client-level defaults
        for this single call — needed for the stochastic N-run swarm
        where each run must use a distinct seed to actually sample
        differently.
        """
        r = self.chat_with_logprobs(
            messages=messages,
            image_b64=image_b64,
            system_prompt=system_prompt,
            seed=seed,
            temperature=temperature,
        )
        return r.text, r.completion_tokens

    def chat_with_logprobs(
        self,
        messages: List[Dict],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResult:
        """
        Chat completion with per-token logprobs (``logprobs.content``) for entropy H_t, h_i.
        """
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = list(messages)

        if image_b64 is not None:
            for msg in full_messages:
                if msg["role"] == "user":
                    original = msg["content"]
                    if isinstance(original, str):
                        original = [{"type": "text", "text": original}]
                    msg["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        }
                    ] + original
                    break

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "seed": self.seed if seed is None else int(seed),
            "max_tokens": self.max_new_tokens,
        }
        if self.chat_request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)

        content_items: Optional[List[Dict[str, Any]]] = None
        token_strings: List[str] = []
        lp_block = choice.get("logprobs")
        if self.chat_request_logprobs and isinstance(lp_block, dict) and lp_block.get("content"):
            content_items = list(lp_block["content"])
            for item in content_items:
                if isinstance(item, dict) and "token" in item:
                    token_strings.append(str(item["token"]))

        return ChatResult(
            text=text or "",
            completion_tokens=int(tokens or 0),
            content_logprobs=content_items,
            token_strings=token_strings,
        )

    # ------------------------------------------------------------------
    # Constrained decoding scoring pass (Appendix B / Eq. 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _max_tokens_for_labels(label_list: List[str]) -> int:
        """Upper bound for generating one constrained label (may be multi-token)."""
        if not label_list:
            return 16
        rough = max(len(str(l).split()) * 6 for l in label_list) + 8
        return max(16, min(128, rough))

    @staticmethod
    def _merge_logprobs_from_choice(choice: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize vLLM/OpenAI logprob shapes from either /completions or /chat/completions
        into a flat token string -> logprob map.
        """
        raw: Dict[str, float] = {}
        lp = choice.get("logprobs")
        if not lp or not isinstance(lp, dict):
            return raw

        # Chat Completions API: logprobs.content[]
        content = lp.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                tok = item.get("token")
                self_lp = item.get("logprob")
                if tok is not None and self_lp is not None:
                    raw[str(tok).strip()] = float(self_lp)
                for alt in item.get("top_logprobs") or []:
                    if isinstance(alt, dict):
                        t = alt.get("token")
                        v = alt.get("logprob")
                        if t is not None and v is not None:
                            raw[str(t).strip()] = float(v)

        # Completions API: logprobs.top_logprobs — list of dicts token_str -> logprob
        tops = lp.get("top_logprobs")
        if isinstance(tops, list):
            for token_lps in tops:
                if not token_lps:
                    continue
                if isinstance(token_lps, dict):
                    for tok, v in token_lps.items():
                        raw[str(tok).strip()] = float(v)
                elif isinstance(token_lps, list):
                    for alt in token_lps:
                        if isinstance(alt, dict):
                            t = alt.get("token")
                            v = alt.get("logprob")
                            if t is not None and v is not None:
                                raw[str(t).strip()] = float(v)
        return raw

    def _softmax_over_labels(
        self, raw_logprobs: Dict[str, float], label_list: List[str]
    ) -> Dict[str, float]:
        if not raw_logprobs:
            uniform = 1.0 / len(label_list)
            return {lbl: uniform for lbl in label_list}
        log_vals = []
        for lbl in label_list:
            lv = raw_logprobs.get(lbl, raw_logprobs.get(lbl.strip(), -1e9))
            # Case-insensitive fallback for tokenizer spacing
            if lv <= -1e8:
                for k, v in raw_logprobs.items():
                    if k.lower() == str(lbl).lower():
                        lv = v
                        break
            log_vals.append(lv)
        max_lv = max(log_vals)
        exp_vals = [math.exp(v - max_lv) for v in log_vals]
        total = sum(exp_vals)
        if total <= 0:
            uniform = 1.0 / len(label_list)
            return {lbl: uniform for lbl in label_list}
        return {lbl: ev / total for lbl, ev in zip(label_list, exp_vals)}

    def _append_guided_params(
        self,
        payload: Dict[str, Any],
        label_list: List[str],
        *,
        chat: bool,
        prefer_structured_outputs: Optional[bool] = None,
    ) -> None:
        """Mutate payload with vLLM guided decoding.

        ``prefer_structured_outputs`` defaults to ``self.prefer_structured_outputs``.
        Passing it explicitly is the thread-safe way for the retry path to
        flip the flag without mutating shared state.
        """
        if not self.guided_scoring_enabled:
            return
        prefer = (
            self.prefer_structured_outputs
            if prefer_structured_outputs is None
            else bool(prefer_structured_outputs)
        )
        if prefer:
            payload["structured_outputs"] = {"choice": label_list}
            if not chat:
                payload["logprobs"] = min(20, max(len(label_list), 5))
        else:
            payload["guided_choice"] = label_list
            if not chat:
                payload["logprobs"] = len(label_list)

    def _score_labels_vision_chat(
        self,
        prompt_prefix: str,
        label_list: List[str],
        image_b64: str,
        prefer_structured_outputs: Optional[bool] = None,
    ) -> Optional[Dict[str, float]]:
        """Vision-conditioned scoring via /chat/completions + structured_outputs (or legacy guided_choice)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": prompt_prefix},
                ],
            }
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "seed": self.seed,
            "max_tokens": self._max_tokens_for_labels(label_list),
            "logprobs": True,
            "top_logprobs": min(max(len(label_list), 5), 20),
        }
        self._append_guided_params(
            payload, label_list, chat=True,
            prefer_structured_outputs=prefer_structured_outputs,
        )

        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            choice = data["choices"][0]
            raw = self._merge_logprobs_from_choice(choice)
            if raw:
                return self._softmax_over_labels(raw, label_list)
        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.debug("vision chat scoring failed: %s", e)
            return None
        return None

    def _score_labels_completions_text(
        self,
        prompt_prefix: str,
        label_list: List[str],
        prefer_structured_outputs: Optional[bool] = None,
    ) -> Optional[Dict[str, float]]:
        """Text-only /completions scoring (no image)."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_prefix,
            "temperature": 0.0,
            "seed": self.seed,
            "max_tokens": self._max_tokens_for_labels(label_list),
        }
        self._append_guided_params(
            payload, label_list, chat=False,
            prefer_structured_outputs=prefer_structured_outputs,
        )

        url = f"{self.base_url}/completions"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                return None
            data = resp.json()
            choice = data["choices"][0]
            raw = self._merge_logprobs_from_choice(choice)
            if raw:
                return self._softmax_over_labels(raw, label_list)
        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logger.debug("completions scoring failed: %s", e)
            return None
        return None

    def _retry_alternate_guided_api(
        self,
        prompt_prefix: str,
        label_list: List[str],
        image_b64: Optional[str],
    ) -> Optional[Dict[str, float]]:
        """Flip structured_outputs <-> guided_choice and retry once.

        Thread-safe: the flipped preference is passed as a parameter,
        ``self.prefer_structured_outputs`` is never mutated.
        """
        flipped = not self.prefer_structured_outputs
        if image_b64:
            return self._score_labels_vision_chat(
                prompt_prefix, label_list, image_b64,
                prefer_structured_outputs=flipped,
            )
        return self._score_labels_completions_text(
            prompt_prefix, label_list,
            prefer_structured_outputs=flipped,
        )

    def score_labels(
        self,
        prompt_prefix: str,
        label_list: List[str],
        image_b64: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Constrained-decoding scoring pass (Appendix B).

        Uses vLLM guided decoding (``structured_outputs.choice`` on modern servers, or
        legacy ``guided_choice`` on /completions) to obtain log-probabilities over the
        label vocabulary (Eq. 2), then softmax-normalises.

        When ``image_b64`` is set, uses ``/v1/chat/completions`` with the image so the
        scoring pass is **vision-conditioned** (same conditioning as the agent chat pass).

        For multi-token labels, token-level logprobs are merged from the response; exact
        chain-rule behaviour depends on vLLM's guided decoding for multi-token choices.
        """
        if not label_list:
            return {}
        if not self.guided_scoring_enabled:
            uniform = 1.0 / len(label_list)
            return {lbl: uniform for lbl in label_list}

        result: Optional[Dict[str, float]] = None

        if image_b64:
            result = self._score_labels_vision_chat(prompt_prefix, label_list, image_b64)
            if result is None:
                result = self._retry_alternate_guided_api(
                    prompt_prefix, label_list, image_b64
                )
            if result is None:
                warnings.warn(
                    "Vision-conditioned label scoring failed (check vLLM ≥0.4 with "
                    "structured_outputs or guided_choice on chat). Falling back to "
                    "text-only /completions scoring without the image — calibration "
                    "may not match multimodal agent outputs.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                result = self._score_labels_completions_text(prompt_prefix, label_list)
                if result is None:
                    result = self._retry_alternate_guided_api(
                        prompt_prefix, label_list, None
                    )
        else:
            result = self._score_labels_completions_text(prompt_prefix, label_list)
            if result is None:
                result = self._retry_alternate_guided_api(
                    prompt_prefix, label_list, None
                )

        if result is None:
            warnings.warn(
                "Label scoring HTTP request failed; using uniform distribution.",
                RuntimeWarning,
                stacklevel=2,
            )
            uniform = 1.0 / len(label_list)
            return {lbl: uniform for lbl in label_list}

        return result

    # ------------------------------------------------------------------
    # Token counting helper
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Approximate token count (whitespace split; replace with tiktoken if needed)."""
        return len(text.split())


def configure_vllm_client_from_yaml(
    client: VLLMClient,
    model_cfg: Optional[Dict[str, Any]] = None,
    *,
    orchestrator: str = "autogen_swarm",
) -> None:
    """
    Apply ``model.*`` keys from YAML so runs match the paper config.

    ``entropy_routing`` always enables chat logprobs (token entropy); other orchestrators
    respect ``model.logprobs``.
    """
    m = model_cfg or {}
    top_lp = m.get("top_logprobs")
    if top_lp is not None:
        client.top_logprobs = int(top_lp)
    client.chat_request_logprobs = True if orchestrator == "entropy_routing" else bool(
        m.get("logprobs", True)
    )
    client.guided_scoring_enabled = bool(m.get("guided_choice", True))
    client.prefer_structured_outputs = bool(m.get("prefer_structured_outputs", True))


def validate_model_server_matches_config(cfg: Dict[str, Any], *, timeout: float = 5.0) -> None:
    """
    If ``model.strict_server_model`` is true, require ``GET /v1/models`` to list
    ``model.backbone`` (fail-fast before a long benchmark).
    """
    m = cfg.get("model") or {}
    if not m.get("strict_server_model"):
        return
    backbone = m.get("backbone")
    bu = (m.get("vllm_base_url") or "").rstrip("/")
    if not bu or not backbone:
        raise ValueError("strict_server_model requires model.backbone and model.vllm_base_url")
    models_url = f"{bu}/models" if bu.endswith("/v1") else f"{bu}/v1/models"
    r = requests.get(models_url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    ids = [x.get("id") for x in data.get("data", []) if isinstance(x, dict)]
    if not ids:
        warnings.warn(
            "strict_server_model: server returned no models in /v1/models data[]",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if backbone not in ids:
        raise ValueError(
            f"strict_server_model: model.backbone {backbone!r} is not listed by the server. "
            f"Available ids (sample): {ids[:16]}"
        )
    print(f"  strict_server_model: backbone {backbone!r} matches served model id.")
