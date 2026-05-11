"""Sampling parameters canonical to meno_stand.

SOURCE OF TRUTH:
- Rewrite + QA: /Users/sckwoky/Projects/meno_stand/code/chat.py:184-189
  (a single SamplingParams reused for both stages).
- Rerank: /Users/sckwoky/Projects/meno_stand/code/rerank_utils/rerank_utils.py:172-176
  (separate SamplingParams; no seed because temperature=0 -> greedy).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class QaSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class RerankSampling:
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: bool = True
    top_logprobs: int = 5
