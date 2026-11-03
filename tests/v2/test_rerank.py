import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BertConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
)

from meno_rag.v2.rerank import (
    MXBAI_CENTER,
    MXBAI_TEMPLATE,
    BgeReranker,
    MxbaiReranker,
    Qwen3Reranker,
    load_reranker,
)

WORDS = ["hello", "world", "foo", "bar", "baz", "yes", "no", "question", "answer",
         "<|im_start|>system", "Judge", "<Instruct>:", "<Query>:", "<Document>:"]


def make_tok(padding_side="right"):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for w in WORDS:
        vocab[w] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers.processors import TemplateProcessing

    t.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", padding_side=padding_side,
    )


def bge():
    tok = make_tok()
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, num_labels=1,
                     max_position_embeddings=1100)
    return BgeReranker.from_objects(tok, AutoModelForSequenceClassification.from_config(cfg).eval())


class SpyModel:
    def __init__(self, model):
        self.model = model
        self.inputs = []
        self.config = model.config

    @property
    def device(self):
        return self.model.device

    def __call__(self, **kw):
        self.inputs.append(kw["input_ids"].clone())
        return self.model(**kw)


def qwen(spy=False):
    tok = make_tok("left")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1100)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    if spy:
        model = SpyModel(model)
    return Qwen3Reranker.from_objects(tok, model)


def mxbai(spy=False):
    tok = make_tok("left")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=2048)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    if spy:
        model = SpyModel(model)
    return MxbaiReranker.from_objects(tok, model)


def test_mxbai_scores_are_calibrated_logit_differences():
    r = mxbai()
    r.true_id, r.false_id = 5, 6
    out = r.score("question", ["hello world", "foo bar"])
    assert len(out) == 2 and all(0.0 <= x <= 1.0 for x in out)
    enc = r.tokenizer([MXBAI_TEMPLATE.format(query="question", document="hello world")],
                      return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        last = r.model(**enc).logits[0, -1].float()
    raw = float(last[5] - last[6])
    assert out[0] == pytest.approx(float(torch.sigmoid(torch.tensor(raw - MXBAI_CENTER))), abs=1e-4)


def test_mxbai_truncates_the_document_only_and_ends_with_the_prompt():
    r = mxbai(spy=True)
    r.score("question", ["foo " * 900, "hello"], batch_size=8, max_length=10)
    ids = r.model.inputs[0]
    tail = r.tokenizer("Relevance:<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    for row in ids:  # left padding: every sequence ends with the model's prompt tail
        assert row[-len(tail):].tolist() == tail
    foo = r.tokenizer("foo", add_special_tokens=False)["input_ids"][0]
    assert int((ids[0] == foo).sum()) == 10  # the document cut to max_length tokens


@pytest.mark.parametrize("make", [bge, qwen, mxbai])
def test_scores_in_range_and_length(make):
    r = make()
    texts = ["hello world", "foo bar baz", "", "   ", "answer"]
    out = r.score("question", texts, batch_size=2)
    assert len(out) == len(texts)
    assert all(0.0 <= s <= 1.0 for s in out)


@pytest.mark.parametrize("make", [bge, qwen, mxbai])
def test_empty_texts(make):
    assert make().score("question", []) == []


@pytest.mark.parametrize("make", [bge, qwen, mxbai])
def test_long_text_truncated(make):
    r = make()
    out = r.score("question", ["foo " * 800, "hello"], max_length=32)
    assert len(out) == 2 and all(0.0 <= s <= 1.0 for s in out)


def test_bge_long_query_falls_back():
    r = bge()
    out = r.score("question " * 100, ["hello world"], max_length=16)
    assert len(out) == 1 and 0.0 <= out[0] <= 1.0


def test_qwen_input_ends_with_suffix_and_respects_max_length():
    r = qwen(spy=True)
    suffix = r.tokenizer("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                         add_special_tokens=False)["input_ids"]
    r.score("question", ["foo " * 800, "hello"], max_length=64, batch_size=8)
    ids = r.model.inputs[0]
    assert ids.shape[1] <= 64
    for row in ids:  # left padding: the sequence ends with the suffix
        assert row[-len(suffix):].tolist() == suffix


def test_qwen_yes_no_ids_used():
    r = qwen()
    assert r.yes_id != r.no_id


@pytest.mark.parametrize("make", [bge, qwen, mxbai])
def test_oom_halves_batch(make):
    r = make()
    inner = r.model
    seen = []

    class Wrap:
        config = inner.config
        device = inner.device

        def __call__(self, **kw):
            n = kw["input_ids"].shape[0]
            seen.append(n)
            if n > 2:
                raise torch.cuda.OutOfMemoryError("boom")
            return inner(**kw)

    r.model = Wrap()
    out = r.score("question", ["foo"] * 7, batch_size=8)
    assert len(out) == 7
    assert seen[:3] == [7, 4, 2]
    assert max(seen[3:]) <= 2


def test_load_reranker_unknown():
    with pytest.raises(ValueError):
        load_reranker("nope", "/x", "cpu")
