"""Device resolution and inference_mode guard for FRIDA."""

import pytest

torch = pytest.importorskip("torch")


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    from meno_rag.stand.resources import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    from meno_rag.stand.resources import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cpu():
    from meno_rag.stand.resources import _resolve_device

    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda_index():
    from meno_rag.stand.resources import _resolve_device

    assert _resolve_device("cuda:1") == "cuda:1"
