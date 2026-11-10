"""Cross-encoder rerankers (bge-reranker-v2-m3, Qwen3-Reranker, mxbai-rerank-v2) behind one interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import torch

from meno_rag.v2._batching import run_batched
from meno_rag.v2.embed import _dtype, _to_device

MAX_LENGTH = 512
QWEN3_INSTRUCTION = "Given a question about Novosibirsk State University, retrieve passages that answer it"
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# mxbai-rerank-v2's own prompt (chat_template.jinja of the model repo)
MXBAI_TEMPLATE = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nquery: {query}\ndocument: {document}\nYou are a search relevance expert who evaluates how "
    "well documents match search queries. For each query-document pair, carefully analyze the semantic relationship "
    "between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\n"
    "Relevance:<|im_end|>\n<|im_start|>assistant\n"
)
# raw scores (logit '1' - logit '0') of mxbai-rerank-large-v2 on our KB sit around 4..13; 8.5 is the median best
# score of unanswerable benchmark questions (2026-11-03 pilot), so 0.5 rejects about half of them while keeping
# ~all answerable ones
MXBAI_CENTER = 8.5
MXBAI_SCALE = 1.0
MXBAI_MAX_DOC_TOKENS = 800


class Reranker(Protocol):
    def score(
        self, query: str, texts: list[str], batch_size: int = 32, max_length: int = MAX_LENGTH
    ) -> list[float]: ...


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

    def score(self, query: str, texts: list[str], batch_size: int = 32, max_length: int = MAX_LENGTH) -> list[float]:
        if not texts:
            return []
        parts = run_batched(texts, lambda batch: self._score_batch(query, list(batch), max_length), batch_size)
        return [s for part in parts for s in part]


class BgeReranker(_BaseReranker):
    def __init__(
        self, tokenizer: Any = None, model: Any = None, *, path: str | None = None, device: str = "cpu"
    ) -> None:
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
    def __init__(
        self, tokenizer: Any = None, model: Any = None, *, path: str | None = None, device: str = "cpu"
    ) -> None:
        if tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
            model = _to_device(AutoModelForCausalLM.from_pretrained(path, dtype=_dtype(device)), device)
        super().__init__(tokenizer, model)
        self.prefix_ids = tokenizer(QWEN3_PREFIX, add_special_tokens=False)["input_ids"]
        self.suffix_ids = tokenizer(QWEN3_SUFFIX, add_special_tokens=False)["input_ids"]
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")

    def _score_batch(self, query, texts, max_length):
        budget = max(1, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        bodies = [f"<Instruct>: {QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {t}" for t in texts]
        body_ids = self.tokenizer(bodies, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
        ids = [self.prefix_ids + b + self.suffix_ids for b in body_ids]
        enc = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :].float()
        pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
        return torch.softmax(pair, dim=1)[:, 1].tolist()


class MxbaiReranker(_BaseReranker):
    """mixedbread-ai/mxbai-rerank-v2 (a Qwen2 causal LM): logit('1') - logit('0') at the last position of the
    model's prompt, mapped to [0, 1] as sigmoid((raw - MXBAI_CENTER) / MXBAI_SCALE). max_length counts document
    tokens only (the prompt and the query come on top)."""

    def __init__(
        self,
        tokenizer: Any = None,
        model: Any = None,
        *,
        path: str | None = None,
        device: str = "cpu",
        true_id: int = 16,
        false_id: int = 15,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
            model = _to_device(AutoModelForCausalLM.from_pretrained(path, dtype=_dtype(device)), device)
            cfg = Path(path) / "1_LogitScore" / "config.json"
            if cfg.exists():
                ids = json.loads(cfg.read_text(encoding="utf-8"))
                true_id, false_id = int(ids["true_token_id"]), int(ids["false_token_id"])
        super().__init__(tokenizer, model)
        self.true_id, self.false_id = true_id, false_id

    def score(
        self, query: str, texts: list[str], batch_size: int = 8, max_length: int = MXBAI_MAX_DOC_TOKENS
    ) -> list[float]:
        return super().score(query, texts, batch_size, max_length)

    def _score_batch(self, query, texts, max_length):
        docs = self.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompts = [MXBAI_TEMPLATE.format(query=query, document=self.tokenizer.decode(d)) for d in docs]
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            last = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        raw = last[:, self.true_id] - last[:, self.false_id]
        return torch.sigmoid((raw - MXBAI_CENTER) / MXBAI_SCALE).tolist()


def load_reranker(name: str, path: str, device: str) -> Reranker:
    if name == "bge":
        return BgeReranker(path=path, device=device)
    if name == "qwen3":
        return Qwen3Reranker(path=path, device=device)
    if name == "mxbai":
        return MxbaiReranker(path=path, device=device)
    raise ValueError(f"unknown reranker {name!r}; expected 'bge', 'qwen3' or 'mxbai'")
