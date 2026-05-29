from __future__ import annotations

from datetime import datetime

from meno_rag.llm.status import ModelStatusState, ModelStatusStore
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


class ModelRateLimitedError(Exception):
    def __init__(self, model_id: str, until: datetime, retry_after_sec: int | None) -> None:
        super().__init__(f"{model_id} rate_limited until {until.isoformat()}")
        self.model_id = model_id
        self.until = until
        self.retry_after_sec = retry_after_sec


class ModelUnreachableError(Exception):
    def __init__(self, model_id: str, until: datetime) -> None:
        super().__init__(f"{model_id} unreachable until {until.isoformat()}")
        self.model_id = model_id
        self.until = until


class CoreModelUnavailableError(Exception):
    def __init__(self) -> None:
        super().__init__("no_vllm_model_available_for_rewrite_rerank")


async def resolve_pipeline_runtime(
    *,
    requested_model: str | None,
    vllm_registry,
    openrouter_registry,
    status_store: ModelStatusStore,
    rag_rewrite_rerank_model: str | None,
    openrouter_base_url: str,
    configured_default: str | None,
    vllm_endpoint_list: list[str],
) -> PipelineRuntime:
    or_models = await openrouter_registry.list_models() if openrouter_registry is not None else []
    or_ids = {m["id"] for m in or_models}

    normalized = requested_model.strip() if isinstance(requested_model, str) and requested_model.strip() else None

    if normalized and normalized in or_ids:
        status = await status_store.get(normalized)
        if status.state == ModelStatusState.RATE_LIMITED and status.until is not None:
            raise ModelRateLimitedError(normalized, status.until, retry_after_sec=None)
        if status.state == ModelStatusState.UNREACHABLE and status.until is not None:
            raise ModelUnreachableError(normalized, status.until)
        core = await _resolve_core_runtime(
            vllm_registry=vllm_registry,
            rag_rewrite_rerank_model=rag_rewrite_rerank_model,
            vllm_endpoint_list=vllm_endpoint_list,
        )
        gen = ModelRuntime(provider="openrouter", model_id=normalized, base_url=openrouter_base_url)
        return PipelineRuntime(core=core, generation=gen)

    # vllm path (default)
    model_id, base_url = await vllm_registry.resolve_model(normalized, configured_default)
    if base_url is None:
        if not vllm_endpoint_list:
            raise ValueError("No VLLM_ENDPOINTS configured.")
        base_url = f"{vllm_endpoint_list[0]}/v1"
    rt = ModelRuntime(provider="vllm", model_id=model_id, base_url=base_url)
    return PipelineRuntime.uniform(rt)


async def _resolve_core_runtime(
    *, vllm_registry, rag_rewrite_rerank_model: str | None, vllm_endpoint_list: list[str]
) -> ModelRuntime:
    vllm_models = await vllm_registry.list_models()
    if not vllm_models:
        raise CoreModelUnavailableError()

    # Order: by VLLM_ENDPOINTS declaration, then by 'created' ascending.
    endpoint_priority = {ep.rstrip("/"): idx for idx, ep in enumerate(vllm_endpoint_list)}
    vllm_models_sorted = sorted(
        vllm_models,
        key=lambda m: (endpoint_priority.get(str(m.get("endpoint", "")).rstrip("/"), 9999), m.get("created", 0)),
    )

    if rag_rewrite_rerank_model:
        for m in vllm_models_sorted:
            if m["id"] == rag_rewrite_rerank_model:
                # Resolve via the registry so a model served by several
                # endpoints round-robins instead of pinning to one.
                endpoint = vllm_registry.lookup_endpoint(m["id"])
                return ModelRuntime(
                    provider="vllm",
                    model_id=m["id"],
                    base_url=f"{endpoint}/v1" if endpoint else "",
                )

    first = vllm_models_sorted[0]
    endpoint = vllm_registry.lookup_endpoint(first["id"])
    return ModelRuntime(
        provider="vllm",
        model_id=first["id"],
        base_url=f"{endpoint}/v1" if endpoint else "",
    )


def resolve_core_model_id_sync(
    vllm_models: list[dict], rag_rewrite_rerank_model: str | None, vllm_endpoint_list: list[str]
) -> str | None:
    """Synchronous helper used by /v1/models to compute core_model_id from a
    snapshot of vllm models. Returns None when no vLLM is available."""
    if not vllm_models:
        return None
    endpoint_priority = {ep.rstrip("/"): idx for idx, ep in enumerate(vllm_endpoint_list)}
    sorted_models = sorted(
        vllm_models,
        key=lambda m: (endpoint_priority.get(str(m.get("endpoint", "")).rstrip("/"), 9999), m.get("created", 0)),
    )
    if rag_rewrite_rerank_model:
        for m in sorted_models:
            if m["id"] == rag_rewrite_rerank_model:
                return m["id"]
    return sorted_models[0]["id"]
