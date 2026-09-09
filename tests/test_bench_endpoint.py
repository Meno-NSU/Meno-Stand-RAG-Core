"""Поведение /v1/chat/completions в bench-режиме."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meno_rag.api.admission import AdmissionController

BENCH_TOKEN = "test-bench-token-0123456789"


@pytest.mark.asyncio
async def test_lifespan_installs_a_separate_bench_admission_controller():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.bench_admission, AdmissionController)
        assert app.state.bench_admission is not app.state.admission
        assert app.state.bench_admission.max_concurrent == 4
