"""Cross-encoder rerankers (bge-reranker-v2-m3, Qwen3-Reranker) behind one interface."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from meno_rag.v2._batching import run_batched
from meno_rag.v2.embed import _dtype, _to_device

MAX_LENGTH = 512
QWEN3_INSTRUCTION = (
    "Given a question about Novosibirsk State University, retrieve passages that answer it"
)
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker(Protocol):
    def score(self, query: str, texts: list[str], batch_size: int = 32,
              max_length: int = MAX_LENGTH) -> list[float]: ...


class _BaseReranker:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_objects(cls, tokenizer: Any, model: Any):
        return cls(tokenizer, model)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _score_batch(self, query: str, texts: list[str], max_length: int) -> list[float]:
        raise NotImplementedError

    def score(self, query: str, texts: list[str], batch_size: int = 32,
              max_length: int = MAX_LENGTH) -> list[float]:
        if not texts:
            return []
        parts = run_batched(
            texts, lambda batch: self._score_batch(query, list(batch), max_length), batch_size
        )
        return [s for part in parts for s in part]


class BgeReranker(_BaseReranker):
    def __init__(self, tokenizer: Any = None, model: Any = None, *, path: str | None = None,
                 device: str = "cpu") -> None:
        if tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = _to_device(
                AutoModelForSequenceClassification.from_pretrained(path, dtype=_dtype(device)),
                device,
            )
        super().__init__(tokenizer, model)

    def _score_batch(self, query, texts, max_length):
        kw = dict(max_length=max_length, padding=True, return_tensors="pt")
        try:
            enc = self.tokenizer([query] * len(texts), texts, truncation="only_second", **kw)
        except Exception:  # bare Exception from tokenizers: query alone > max_length
            enc = self.tokenizer([query] * len(texts), texts, truncation=True, **kw)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits.reshape(len(texts), -1)[:, 0]
        return torch.sigmoid(logits.float()).tolist()


class Qwen3Reranker(_BaseReranker):
    def __init__(self, tokenizer: Any = None, model: Any = None, *, path: str | None = None,
                 device: str = "cpu") -> None:
        if tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
            model = _to_device(
                AutoModelForCausalLM.from_pretrained(path, dtype=_dtype(device)), device
            )
        super().__init__(tokenizer, model)
        self.prefix_ids = tokenizer(QWEN3_PREFIX, add_special_tokens=False)["input_ids"]
        self.suffix_ids = tokenizer(QWEN3_SUFFIX, add_special_tokens=False)["input_ids"]
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")

    def _score_batch(self, query, texts, max_length):
        budget = max(1, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        bodies = [
            f"<Instruct>: {QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {t}" for t in texts
        ]
        body_ids = self.tokenizer(
            bodies, add_special_tokens=False, truncation=True, max_length=budget
        )["input_ids"]
        ids = [self.prefix_ids + b + self.suffix_ids for b in body_ids]
        enc = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :].float()
        pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
        return torch.softmax(pair, dim=1)[:, 1].tolist()


def load_reranker(name: str, path: str, device: str) -> Reranker:
    if name == "bge":
        return BgeReranker(path=path, device=device)
    if name == "qwen3":
        return Qwen3Reranker(path=path, device=device)
    raise ValueError(f"unknown reranker {name!r}; expected 'bge' or 'qwen3'")
