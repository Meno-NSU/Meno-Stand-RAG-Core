from meno_rag.config import Settings


def test_trace_capture_defaults_off():
    s = Settings()
    assert s.capture_pipeline_trace is False
    assert s.pipeline_trace_sample_rate == 1.0
    assert s.trace_database_url == "sqlite+aiosqlite:///./var/meno_rag_trace.sqlite3"
    assert s.pipeline_trace_queue_max == 1000


def test_trace_capture_reads_env(monkeypatch):
    monkeypatch.setenv("CAPTURE_PIPELINE_TRACE", "true")
    monkeypatch.setenv("PIPELINE_TRACE_SAMPLE_RATE", "0.05")
    s = Settings()
    assert s.capture_pipeline_trace is True
    assert s.pipeline_trace_sample_rate == 0.05
