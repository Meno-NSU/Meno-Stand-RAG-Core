import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from bm25s import BM25, tokenize
from nltk.stem.snowball import SnowballStemmer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, T5EncoderModel

from meno_rag.stand.tokenization import tokenize_and_normalize_text

MAX_EMBEDDER_TOKENS: int = 512


def frida_pool(hidden_state, mask, pooling_method: str = "cls"):
    if pooling_method not in {"mean", "cls"}:
        raise ValueError(f"The pooling method {pooling_method} is unknown!")
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        emb = s / d
    else:
        emb = hidden_state[:, 0]
    return emb


def vectorize_search_query(
    search_query: str,
    emb_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    emb_model: T5EncoderModel,
) -> np.ndarray:
    inputs = ["search_query: " + search_query]
    tokenized_inputs = emb_tokenizer(
        inputs,
        max_length=MAX_EMBEDDER_TOKENS,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = emb_model(**tokenized_inputs.to(emb_model.device))
    embeddings = frida_pool(
        outputs.last_hidden_state.to(torch.float32),
        tokenized_inputs["attention_mask"],
        pooling_method="cls",
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def find_relevant_chunks(
    search_query: str,
    retriever: Any,
    max_num_chunks: int,
    stemmer: Optional[SnowballStemmer] = None,
    embedder: Optional[tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, T5EncoderModel]] = None,
) -> list[tuple[int, float]]:
    relevant_chunks: list[tuple[int, float]] = []
    if isinstance(retriever, BM25):
        if stemmer is None:
            raise RuntimeError("The SnowballStemmer is not specified!")
        query_tokens = tokenize(tokenize_and_normalize_text(search_query, stemmer), stemmer=None, stopwords=[])
        results, scores = retriever.retrieve(query_tokens, k=max_num_chunks)
        sum_scores = sum(float(scores[0, i]) for i in range(results.shape[1]))
        if sum_scores <= 0.0:
            return []
        for i in range(results.shape[1]):
            relevant_chunks.append((int(results[0, i]), float(scores[0, i] / sum_scores)))
    else:
        if embedder is None:
            raise RuntimeError("The FRIDA-based embedder is not specified!")
        query_vector = vectorize_search_query(search_query, embedder[0], embedder[1])
        distances, indices = retriever.search(query_vector, max_num_chunks)
        scores = [
            (1.0 + math.exp(-1.0)) / (1.0 + math.exp(float(distances[0, i]) - 1.0)) for i in range(len(distances[0]))
        ]
        sum_scores = sum(scores)
        if sum_scores <= 0.0:
            return []
        for i in range(len(distances[0])):
            relevant_chunks.append((int(indices[0, i]), scores[i] / sum_scores))
    return relevant_chunks


def combine_relevant_chunks(
    chunk_list_1: Optional[list[tuple[int, float]]] = None,
    chunk_list_2: Optional[list[tuple[int, float]]] = None,
) -> list[tuple[int, float]]:
    united_chunks: dict[int, float] = {}
    if chunk_list_1 is not None:
        for idx, score in chunk_list_1:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    if chunk_list_2 is not None:
        for idx, score in chunk_list_2:
            if idx in united_chunks:
                if score > united_chunks[idx]:
                    united_chunks[idx] = score
            else:
                united_chunks[idx] = score
    return sorted([(idx, united_chunks[idx]) for idx in united_chunks], key=lambda it: (-it[1], it[0]))
