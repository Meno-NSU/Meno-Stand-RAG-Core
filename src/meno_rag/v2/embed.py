"""Embedders (FRIDA, Qwen3-Embedding) behind one interface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F

from meno_rag.v2._batching import run_batched

MAX_LENGTH = 512
QWEN3_INSTRUCT = (
    "Instruct: Given a question about Novosibirsk State University, retrieve passages that answer it\nQuery: "
)


class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: list[str]) -> torch.Tensor: ...

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> torch.Tensor: ...


class _BaseEmbedder:
    name = ""
    query_prefix = ""
    doc_prefix = ""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.dim = int(model.config.hidden_size)

    @classmethod
    def from_objects(cls, tokenizer: Any, model: Any):
        return cls(tokenizer, model)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _embed(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        hidden = out.last_hidden_state.to(torch.float32)
        emb = F.normalize(self._pool(hidden, enc["attention_mask"]), p=2, dim=1)
        return emb.to(torch.float16)

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float16, device=self.device)
        return self._embed([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        texts = [self.doc_prefix + t for t in texts]
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float16, device=self.device)
        parts = run_batched(texts, cast("Callable[[Sequence[Any]], Any]", self._embed), batch_size)
        return torch.cat(parts, dim=0)


class FridaEmbedder(_BaseEmbedder):
    name = "frida"
    query_prefix = "search_query: "
    doc_prefix = "search_document: "

    def __init__(
        self,
        tokenizer: Any = None,
        model: Any = None,
        *,
        path: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer, T5EncoderModel

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = _to_device(T5EncoderModel.from_pretrained(path, dtype=_resolve_dtype(device, dtype)), device)
        super().__init__(tokenizer, model)

    def _pool(self, hidden, mask):
        return hidden[:, 0]


class Qwen3Embedder(_BaseEmbedder):
    name = "qwen3"
    query_prefix = QWEN3_INSTRUCT
    doc_prefix = ""

    def __init__(
        self,
        tokenizer: Any = None,
        model: Any = None,
        *,
        path: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
            model = _to_device(
                AutoModel.from_pretrained(cast("str", path), dtype=_resolve_dtype(device, dtype)), device
            )
        super().__init__(tokenizer, model)

    def _pool(self, hidden, mask):
        # Last real token, correct for both left and right padding.
        n, length = mask.shape
        positions = torch.arange(length, device=mask.device).expand(n, length)
        last = (positions * mask).argmax(dim=1)
        return hidden[torch.arange(n, device=hidden.device), last.to(hidden.device)]


def _dtype(device: str):
    return torch.float16 if str(device).startswith("cuda") else torch.float32


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _resolve_dtype(device: str, dtype: str | None):
    """Weights dtype: bf16/fp16 on CUDA when asked, fp32 on CPU, `_dtype` when unset."""
    if dtype is not None and dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected 'bf16' or 'fp16'")
    if dtype is None or not str(device).startswith("cuda"):
        return _dtype(device)
    return _DTYPES[dtype]


def _to_device(model, device: str):
    return model.to(device).eval()


def load_embedder(name: str, path: str, device: str, dtype: str | None = None) -> Embedder:
    _resolve_dtype(device, dtype)  # fail fast on an unknown dtype
    if name == "frida":
        return FridaEmbedder(path=path, device=device, dtype=dtype)
    if name == "qwen3":
        return Qwen3Embedder(path=path, device=device, dtype=dtype)
    raise ValueError(f"unknown embedder {name!r}; expected 'frida' or 'qwen3'")
