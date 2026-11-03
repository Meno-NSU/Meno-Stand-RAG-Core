"""Qdrant dense backend over the REST API (httpx): one collection per KB release, alias for switching."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import numpy as np

from meno_rag.v2.sparse import SPARSE_NAME, Bm25Encoder


class QdrantError(RuntimeError):
    pass


class QdrantStore:
    def __init__(self, url: str, timeout: float = 30, *, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _req(self, method: str, path: str, *, ok_404: bool = False, **kw) -> dict | None:
        r = self._http.request(method, path, **kw)
        if r.status_code == 404 and ok_404:
            return None
        if r.status_code >= 400:
            raise QdrantError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def get_collection(self, name: str) -> dict | None:
        body = self._req("GET", f"/collections/{name}", ok_404=True)
        return None if body is None else body["result"]

    def create_collection(self, name: str, dim: int, *, on_disk: bool = True, sparse: bool = False) -> None:
        """sparse=True adds the BM25 sparse vector (modifier idf) next to the unnamed dense one."""
        existing = self.get_collection(name)
        if existing is not None:
            params = existing["config"]["params"]
            have = params["vectors"]["size"]
            if have != dim:
                raise ValueError(f"collection {name!r} exists with dim {have}, requested {dim}")
            if sparse and SPARSE_NAME not in (params.get("sparse_vectors") or {}):
                raise ValueError(f"collection {name!r} exists without the {SPARSE_NAME!r} sparse vector")
            return
        body = {
            "vectors": {"size": dim, "distance": "Dot", "on_disk": on_disk},
            "hnsw_config": {"m": 16, "ef_construct": 128},
            "on_disk_payload": True,
            "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}},
        }
        if sparse:
            body["sparse_vectors"] = {SPARSE_NAME: {"modifier": "idf", "index": {"on_disk": False}}}
        self._req("PUT", f"/collections/{name}", json=body)

    def upsert_points(self, name: str, points: list[dict]) -> None:
        self._req("PUT", f"/collections/{name}/points", params={"wait": "true"}, json={"points": points})

    def upsert_matrix(
        self,
        name: str,
        matrix,
        batch: int = 2048,
        start_id: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Upsert rows as points (id = start_id + row). Returns the number of rows written."""
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().to("cpu").float().numpy()
        total = int(matrix.shape[0])
        step = max(1, total // 20)
        next_report = step
        for i in range(0, total, batch):
            rows = np.asarray(matrix[i : i + batch], dtype=np.float32)
            points = [
                {"id": start_id + i + j, "vector": row.tolist(), "payload": {}}
                for j, row in enumerate(rows)
            ]
            self._req("PUT", f"/collections/{name}/points", params={"wait": "true"}, json={"points": points})
            done = min(i + batch, total)
            if progress is not None and (done >= next_report or done == total):
                progress(done, total)
                next_report = done + step
        return total

    def count(self, name: str) -> int:
        body = self._req("POST", f"/collections/{name}/points/count", json={"exact": True})
        return int(body["result"]["count"])

    def set_alias(self, alias: str, collection: str) -> None:
        actions = [
            {"delete_alias": {"alias_name": alias}},
            {"create_alias": {"collection_name": collection, "alias_name": alias}},
        ]
        self._req("POST", "/collections/aliases", json={"actions": actions})

    def get_alias_target(self, alias: str) -> str | None:
        body = self._req("GET", "/aliases")
        for a in body["result"]["aliases"]:
            if a["alias_name"] == alias:
                return a["collection_name"]
        return None

    def search(
        self,
        name_or_alias: str,
        vector: list[float],
        k: int,
        *,
        ef: int = 128,
        exact: bool = False,
        rescore: bool = True,
    ) -> list[tuple[int, float]]:
        body = {
            "query": vector,
            "limit": k,
            "with_payload": False,
            "with_vector": False,
            "params": {"hnsw_ef": ef, "exact": exact, "quantization": {"rescore": rescore}},
        }
        res = self._req("POST", f"/collections/{name_or_alias}/points/query", json=body)
        pts = res["result"]["points"]
        out = [(int(p["id"]), float(p["score"])) for p in pts]
        out.sort(key=lambda t: -t[1])
        return out


    def search_sparse(self, name_or_alias: str, indices: list[int], values: list[float], k: int,
                      using: str = SPARSE_NAME) -> list[tuple[int, float]]:
        if not indices or k <= 0:
            return []
        body = {"query": {"indices": indices, "values": values}, "using": using, "limit": k,
                "with_payload": False, "with_vector": False}
        res = self._req("POST", f"/collections/{name_or_alias}/points/query", json=body)
        out = [(int(p["id"]), float(p["score"])) for p in res["result"]["points"] if p["score"] > 0]
        out.sort(key=lambda t: -t[1])
        return out


class QdrantBm25Retriever:
    """BM25 over the collection's sparse vector; same interface as Bm25Retriever."""

    def __init__(self, store: QdrantStore, collection: str, encoder: Bm25Encoder | None = None) -> None:
        self.store = store
        self.collection = collection
        self.encoder = encoder or Bm25Encoder()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        ids, vals = self.encoder.query(query)
        return self.store.search_sparse(self.collection, ids, vals, k)

    def idf(self, token: str) -> float:
        """Qdrant does not expose document frequencies: snippets fall back to plain term counts."""
        return 1.0


class QdrantRetriever:
    def __init__(self, store: QdrantStore, collection: str, embedder, *, ef: int = 128) -> None:
        self.store = store
        self.collection = collection
        self.embedder = embedder
        self.ef = ef

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not query or k <= 0:
            return []
        q = self.embedder.encode_queries([query])[0]
        vec = q.detach().float().cpu().tolist() if hasattr(q, "detach") else [float(x) for x in q]
        return self.store.search(self.collection, vec, k, ef=self.ef)
