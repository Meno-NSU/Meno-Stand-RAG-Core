import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    T5Config,
    T5EncoderModel,
)

from meno_rag.v2.embed import FridaEmbedder, Qwen3Embedder, load_embedder

WORDS = ["search_query:", "search_document:", "Instruct:", "Query:", "hello", "world",
         "Given", "a", "question", "about", "Novosibirsk", "State", "University,",
         "retrieve", "passages", "that", "answer", "it", "foo", "bar", "baz"]


def make_tok(padding_side="right"):
    vocab = {"[PAD]": 0, "[UNK]": 1, "</s>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="</s>", padding_side=padding_side,
    )


class SpyTok:
    def __init__(self, tok):
        self.tok = tok
        self.calls = []
        self.padding_side = tok.padding_side

    def __call__(self, texts, **kw):
        self.calls.append(list(texts))
        return self.tok(texts, **kw)


def tiny_t5(vocab):
    torch.manual_seed(0)
    cfg = T5Config(d_model=32, num_layers=1, num_heads=2, d_kv=16, d_ff=64, vocab_size=vocab)
    return T5EncoderModel(cfg).eval()


def tiny_qwen(vocab):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=16)
    return AutoModel.from_config(cfg).eval()


def frida(spy=False):
    tok = make_tok()
    model = tiny_t5(len(tok))
    return FridaEmbedder.from_objects(SpyTok(tok) if spy else tok, model)


def qwen(spy=False, side="left"):
    tok = make_tok(side)
    model = tiny_qwen(len(tok))
    return Qwen3Embedder.from_objects(SpyTok(tok) if spy else tok, model)


@pytest.mark.parametrize("make", [frida, qwen])
def test_shape_dtype_norm(make):
    e = make()
    for out in (e.encode_queries(["hello world", "foo"]),
                e.encode_docs(["hello world", "foo bar baz", "a"], batch_size=2)):
        assert out.dtype == torch.float16
        assert out.shape[1] == e.dim == 32
        norms = out.float().norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
    assert e.encode_docs(["hello", "world", "foo"], batch_size=2).shape == (3, 32)


def test_frida_prefixes():
    e = frida(spy=True)
    e.encode_queries(["hello"])
    e.encode_docs(["world"])
    assert e.tokenizer.calls == [["search_query: hello"], ["search_document: world"]]


def test_qwen_prefixes():
    e = qwen(spy=True)
    e.encode_queries(["hello"])
    e.encode_docs(["world"])
    assert e.tokenizer.calls[0] == [
        "Instruct: Given a question about Novosibirsk State University, "
        "retrieve passages that answer it\nQuery: hello"
    ]
    assert e.tokenizer.calls[1] == ["world"]


def test_frida_is_cls_pooling():
    e = frida()
    tok, model = e.tokenizer, e.model
    enc = tok(["search_query: hello world"], return_tensors="pt", padding=True)
    with torch.no_grad():
        h = model(**enc).last_hidden_state[:, 0].float()
    ref = torch.nn.functional.normalize(h, dim=1)
    got = e.encode_queries(["hello world"]).float()
    assert torch.allclose(ref, got, atol=1e-2)


@pytest.mark.parametrize("side", ["left", "right"])
def test_qwen_last_token_padding_invariant(side):
    e = qwen(side=side)
    a = e.encode_docs(["hello world foo bar"], batch_size=1).float()
    batched = e.encode_docs(["hello world foo bar", "a"], batch_size=2).float()[0:1]
    assert torch.allclose(a, batched, atol=2e-2)


def test_oom_halving():
    seen = []

    class Out:
        def __init__(self, h):
            self.last_hidden_state = h

    class FakeModel:
        device = torch.device("cpu")
        config = type("C", (), {"hidden_size": 8})()

        def __call__(self, input_ids=None, attention_mask=None, **kw):
            n = input_ids.shape[0]
            seen.append(n)
            if n > 2:
                raise torch.cuda.OutOfMemoryError("boom")
            return Out(torch.randn(n, input_ids.shape[1], 8))

    e = FridaEmbedder.from_objects(make_tok(), FakeModel())
    out = e.encode_docs(["foo"] * 7, batch_size=8)
    assert out.shape == (7, 8)
    assert seen[:3] == [7, 4, 2]
    assert max(seen[3:]) <= 2


def test_oom_at_one_reraises():
    class FakeModel:
        device = torch.device("cpu")
        config = type("C", (), {"hidden_size": 8})()

        def __call__(self, **kw):
            raise torch.cuda.OutOfMemoryError("boom")

    e = FridaEmbedder.from_objects(make_tok(), FakeModel())
    with pytest.raises(torch.cuda.OutOfMemoryError):
        e.encode_docs(["foo", "bar"], batch_size=2)


def test_load_embedder_unknown():
    with pytest.raises(ValueError):
        load_embedder("nope", "/x", "cpu")
