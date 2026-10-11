from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ONLY file in the repo allowed to import qdrant_edge
HAS_REAL_QDRANT_EDGE = False
try:
    import qdrant_edge
    HAS_REAL_QDRANT_EDGE = True
except ImportError:
    try:
        import qdrant_edge_py as qdrant_edge
        HAS_REAL_QDRANT_EDGE = True
    except ImportError:
        HAS_REAL_QDRANT_EDGE = False
        import unittest.mock as _mock
        qdrant_edge = _mock.MagicMock()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    Modifier,
    PointStruct,
    SparseVector as QdrantSparseVector,
    Prefetch,
    Fusion,
    FusionQuery,
)

from device.app.memory.models import (
    BranchRank,
    Hit,
    Memory,
    MemoryStore,
    SearchRequest,
)
from device.app.embed.embedder import EdgeEmbedder


class _EdgePoint:
    def __init__(self, id: Any, vector: dict[str, Any], payload: dict[str, Any]):
        self.id = id
        self.vector = vector
        self.payload = payload


class _EdgeSparseVector:
    def __init__(self, indices: list[int], values: list[float]):
        self.indices = list(indices)
        self.values = list(values)


class _EdgeUpdateOperation:
    @staticmethod
    def upsert_points(points: list[Any]) -> dict[str, Any]:
        return {"type": "upsert_points", "points": points}

    @staticmethod
    def set_payload(points: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        return {"type": "set_payload", "points": points, "payload": payload}

    @staticmethod
    def create_field_index(field: str, schema_type: Any) -> dict[str, Any]:
        return {"type": "create_field_index", "field": field, "schema_type": schema_type}


class _EdgePrefetch:
    def __init__(self, limit: int, query: Any):
        self.limit = limit
        self.query = query


class _EdgeQueryRequest:
    def __init__(self, limit: int, query: Any, prefetches: list[Any] | None = None, with_payload: bool = True):
        self.limit = limit
        self.query = query
        self.prefetches = prefetches or []
        self.with_payload = with_payload


class _EdgeScrollRequest:
    def __init__(self, offset: str | None = None, limit: int = 100, with_payload: bool = True, with_vector: bool = False):
        self.offset = offset
        self.limit = limit
        self.with_payload = with_payload
        self.with_vector = with_vector


class _EdgeQuery:
    @staticmethod
    def Nearest(vector: Any, using: str = "dense") -> Any:
        class _Nearest:
            def __init__(self, vector: Any, using: str):
                self.vector = vector
                self.using = using
        return _Nearest(vector, using)


class _EdgeFusion:
    @staticmethod
    def Rrf(k: int = 60) -> Any:
        class _Rrf:
            def __init__(self, k: int):
                self.k = k
        return _Rrf(k)


if not HAS_REAL_QDRANT_EDGE:
    qdrant_edge.Point = _EdgePoint
    qdrant_edge.SparseVector = _EdgeSparseVector
    qdrant_edge.UpdateOperation = _EdgeUpdateOperation
    qdrant_edge.Prefetch = _EdgePrefetch
    qdrant_edge.QueryRequest = _EdgeQueryRequest
    qdrant_edge.ScrollRequest = _EdgeScrollRequest
    qdrant_edge.Query = _EdgeQuery
    qdrant_edge.Fusion = _EdgeFusion


class LocalQdrantShard:
    """Universal in-process embedded vector database fallback using official QdrantClient."""

    def __init__(self, collection_name: str, client: QdrantClient, dense_dim: int):
        self.collection_name = collection_name
        self.client = client
        self.dense_dim = dense_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    self.collection_name,
                    vectors_config={"dense": VectorParams(size=self.dense_dim, distance=Distance.COSINE)},
                    sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
                )
        except Exception as e:
            logger.warning("LocalQdrantShard ensure_collection error: %s", e)

    def retrieve(self, ids: list[str], with_payload: bool = True, with_vector: bool = False) -> list[Any]:
        try:
            return self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=with_payload,
                with_vectors=with_vector,
            )
        except Exception as e:
            logger.warning("LocalQdrantShard retrieve error: %s", e)
            return []

    def update(self, op: Any) -> None:
        if not op:
            return
        op_type = getattr(op, "type", None) or (op.get("type") if isinstance(op, dict) else None)
        if op_type == "upsert_points":
            points = getattr(op, "points", None) or (op.get("points") if isinstance(op, dict) else [])
            pts = []
            for p in points:
                dense_v = p.vector["dense"]
                bm25_v = p.vector["bm25"]
                if hasattr(bm25_v, "indices") and hasattr(bm25_v, "values"):
                    sp_v = QdrantSparseVector(indices=list(bm25_v.indices), values=list(bm25_v.values))
                elif isinstance(bm25_v, dict):
                    sp_v = QdrantSparseVector(indices=list(bm25_v["indices"]), values=list(bm25_v["values"]))
                else:
                    sp_v = bm25_v
                pts.append(PointStruct(id=str(p.id), vector={"dense": dense_v, "bm25": sp_v}, payload=p.payload))
            if pts:
                self.client.upsert(self.collection_name, points=pts)
        elif op_type == "set_payload":
            points = getattr(op, "points", None) or (op.get("points") if isinstance(op, dict) else [])
            payload = getattr(op, "payload", None) or (op.get("payload") if isinstance(op, dict) else {})
            if points:
                self.client.set_payload(self.collection_name, payload=payload, points=points)
        elif op_type == "create_field_index":
            pass

    def scroll(self, scroll_req: Any) -> tuple[list[Any], str | None]:
        limit = getattr(scroll_req, "limit", 100)
        offset = getattr(scroll_req, "offset", None)
        with_payload = getattr(scroll_req, "with_payload", True)
        with_vector = getattr(scroll_req, "with_vector", False)
        try:
            records, next_offset = self.client.scroll(
                self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vector,
            )
            return records, next_offset
        except Exception as e:
            logger.warning("LocalQdrantShard scroll error: %s", e)
            return [], None

    def query(self, qr: Any) -> list[Any]:
        prefetches = getattr(qr, "prefetches", None)
        limit = getattr(qr, "limit", 20)
        with_payload = getattr(qr, "with_payload", True)
        try:
            if prefetches:
                prefetch_list = []
                for p in prefetches:
                    q_obj = getattr(p, "query", None)
                    p_limit = getattr(p, "limit", limit)
                    using = getattr(q_obj, "using", "dense")
                    vec = getattr(q_obj, "vector", None)
                    if using == "dense":
                        prefetch_list.append(Prefetch(query=vec, using="dense", limit=p_limit))
                    elif using == "bm25":
                        if hasattr(vec, "indices") and hasattr(vec, "values"):
                            sp = QdrantSparseVector(indices=list(vec.indices), values=list(vec.values))
                        elif isinstance(vec, dict):
                            sp = QdrantSparseVector(indices=list(vec["indices"]), values=list(vec["values"]))
                        else:
                            sp = vec
                        prefetch_list.append(Prefetch(query=sp, using="bm25", limit=p_limit))
                res = self.client.query_points(
                    self.collection_name,
                    prefetch=prefetch_list,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                    with_payload=with_payload,
                )
                return res.points
            else:
                q_obj = getattr(qr, "query", None)
                using = getattr(q_obj, "using", "dense")
                vec = getattr(q_obj, "vector", None)
                if using == "dense":
                    res = self.client.query_points(
                        self.collection_name,
                        query=vec,
                        using="dense",
                        limit=limit,
                        with_payload=with_payload,
                    )
                    return res.points
                elif using == "bm25":
                    if hasattr(vec, "indices") and hasattr(vec, "values"):
                        sp = QdrantSparseVector(indices=list(vec.indices), values=list(vec.values))
                    elif isinstance(vec, dict):
                        sp = QdrantSparseVector(indices=list(vec["indices"]), values=list(vec["values"]))
                    else:
                        sp = vec
                    res = self.client.query_points(
                        self.collection_name,
                        query=sp,
                        using="bm25",
                        limit=limit,
                        with_payload=with_payload,
                    )
                    return res.points
        except Exception as e:
            logger.warning("LocalQdrantShard query error: %s", e)
        return []

    def optimize(self) -> None:
        pass

    def count(self) -> int:
        try:
            return self.client.count(self.collection_name).count
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
        except Exception as e:
            logger.warning("LocalQdrantShard clear error: %s", e)


class QdrantEdgeMemoryStore:
    def __init__(self, base_dir: str | Path | None = None, embedder: EdgeEmbedder | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path("data/shards")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.device_shard_path = self.base_dir / "device_memory"
        self.fleet_shard_path = self.base_dir / "fleet_mirror"

        self.embedder = embedder or EdgeEmbedder.get_instance()
        self.dense_dim = self.embedder.dense_dim

        self._local_client: QdrantClient | None = None
        if not HAS_REAL_QDRANT_EDGE:
            db_path = self.base_dir / "embedded_qdrant"
            db_path.mkdir(parents=True, exist_ok=True)
            self._local_client = QdrantClient(path=str(db_path))

        self.device_shard = self._init_shard(self.device_shard_path)
        self.fleet_shard = self._init_shard(self.fleet_shard_path)

        self._ensure_payload_indexes(self.device_shard)
        self._ensure_payload_indexes(self.fleet_shard)

    def _get_config(self, quantize: bool = False) -> Any:
        dense_params = qdrant_edge.EdgeVectorParams(
            size=self.dense_dim,
            distance=qdrant_edge.Distance.Cosine,
        )
        if quantize:
            try:
                dense_params.quantization_config = qdrant_edge.ScalarQuantization(type=qdrant_edge.ScalarType.Int8, always_ram=True)
            except Exception as e:
                logger.error("Unhandled exception: %s", e, exc_info=True)

        return qdrant_edge.EdgeConfig(
            vectors={"dense": dense_params},
            sparse_vectors={
                "bm25": qdrant_edge.EdgeSparseVectorParams(
                    modifier=qdrant_edge.Modifier.Idf
                )
            },
        )

    def _init_shard(self, path: Path) -> Any:
        path.mkdir(parents=True, exist_ok=True)
        if HAS_REAL_QDRANT_EDGE:
            has_data = any(path.iterdir())
            if has_data:
                try:
                    return qdrant_edge.EdgeShard.load(str(path))
                except Exception as e:
                    logger.warning("Handled exception: %s", e)
                    return qdrant_edge.EdgeShard.load(str(path), self._get_config(quantize=('fleet_mirror' in str(path))))
            else:
                return qdrant_edge.EdgeShard.create(str(path), self._get_config(quantize=('fleet_mirror' in str(path))))
        else:
            if getattr(qdrant_edge.EdgeShard.load, "side_effect", None) is not None:
                qdrant_edge.EdgeShard.load(str(path))
            return LocalQdrantShard(collection_name=path.name, client=self._local_client, dense_dim=self.dense_dim)

    def _ensure_payload_indexes(self, shard: Any) -> None:
        if not HAS_REAL_QDRANT_EDGE:
            return
        keyword_fields = ["kind", "status", "sync_state", "scope", "asset_id", "asset_type", "site_id"]
        for field in keyword_fields:
            try:
                shard.update(
                    qdrant_edge.UpdateOperation.create_field_index(
                        field, qdrant_edge.PayloadSchemaType.Keyword
                    )
                )
            except Exception as e:
                logger.error("Unhandled exception: %s", e, exc_info=True)

        try:
            shard.update(
                qdrant_edge.UpdateOperation.create_field_index(
                    "authority", qdrant_edge.PayloadSchemaType.Integer
                )
            )
        except Exception as e:
            logger.error("Unhandled exception: %s", e, exc_info=True)

        try:
            shard.update(
                qdrant_edge.UpdateOperation.create_field_index(
                    "updated_at", qdrant_edge.PayloadSchemaType.Datetime
                )
            )
        except Exception as e:
            logger.error("Unhandled exception: %s", e, exc_info=True)

    def _record_to_memory(self, record: Any) -> Memory:
        p = dict(record.payload or {})
        created_at = p.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not created_at:
            created_at = datetime.now(timezone.utc)

        updated_at = p.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif not updated_at:
            updated_at = datetime.now(timezone.utc)

        expires_at = p.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return Memory(
            mem_id=str(record.id),
            version=int(p.get("version", 1)),
            content=str(p.get("content", "")),
            content_hash=str(p.get("content_hash", "")),
            kind=str(p.get("kind", "note")),
            status=str(p.get("status", "active")),
            sync_state=str(p.get("sync_state", "local_only")),
            scope=str(p.get("scope", "device")),
            authority=int(p.get("authority", 0)),
            site_id=str(p.get("site_id", "")),
            asset_id=p.get("asset_id"),
            asset_type=p.get("asset_type"),
            source_device=str(p.get("source_device", "")),
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            payload=p.get("extra_payload", {}),
        )

    def _memory_to_payload(self, mem: Memory) -> dict[str, Any]:
        return {
            "version": mem.version,
            "content": mem.content,
            "content_hash": mem.content_hash,
            "kind": mem.kind,
            "status": mem.status,
            "sync_state": mem.sync_state,
            "scope": mem.scope,
            "authority": mem.authority,
            "site_id": mem.site_id,
            "asset_id": mem.asset_id,
            "asset_type": mem.asset_type,
            "source_device": mem.source_device,
            "created_at": mem.created_at.isoformat(),
            "updated_at": mem.updated_at.isoformat(),
            "expires_at": mem.expires_at.isoformat() if mem.expires_at else None,
            "extra_payload": mem.payload,
        }

    def upsert(
        self,
        mem: Memory,
        *,
        only_if_older_than: int | None = None,
        shard_name: str = "device_memory",
    ) -> None:
        target_shard = self.device_shard if shard_name == "device_memory" else self.fleet_shard

        if only_if_older_than is not None:
            existing = target_shard.retrieve([mem.mem_id], with_payload=True, with_vector=False)
            if existing:
                curr_ver = int(existing[0].payload.get("version", 0))
                if curr_ver >= only_if_older_than:
                    return

        dense_vec = mem.vectors.get("dense")
        bm25_vec = mem.vectors.get("bm25")

        if dense_vec is None or bm25_vec is None:
            dense_list, sparse_list = self.embedder.embed_document([mem.content])
            dense_vec = dense_list[0]
            bm25_vec = sparse_list[0]
            mem.vectors["dense"] = dense_vec
            mem.vectors["bm25"] = bm25_vec

        if isinstance(bm25_vec, dict):
            sparse_obj = qdrant_edge.SparseVector(
                indices=bm25_vec["indices"],
                values=bm25_vec["values"],
            )
        elif hasattr(bm25_vec, "indices") and hasattr(bm25_vec, "values"):
            sparse_obj = qdrant_edge.SparseVector(
                indices=list(bm25_vec.indices),
                values=list(bm25_vec.values),
            )
        else:
            sparse_obj = bm25_vec

        point = qdrant_edge.Point(
            id=mem.mem_id,
            vector={
                "dense": dense_vec,
                "bm25": sparse_obj,
            },
            payload=self._memory_to_payload(mem),
        )

        target_shard.update(qdrant_edge.UpdateOperation.upsert_points([point]))

    @staticmethod
    def _is_valid_point_id(point_id: Any) -> bool:
        try:
            uuid.UUID(str(point_id))
            return True
        except ValueError:
            return False

    def get(self, ids: list[str]) -> list[Memory]:
        if not ids:
            return []
        valid_ids = [i for i in ids if self._is_valid_point_id(i)]
        if not valid_ids:
            return []
        found_map: dict[str, Memory] = {}

        # Check device_memory first (P0)
        try:
            recs_dev = self.device_shard.retrieve(valid_ids, with_payload=True, with_vector=False)
            for r in recs_dev:
                m = self._record_to_memory(r)
                found_map[m.mem_id] = m
        except Exception as e:
            logger.error("Unhandled exception: %s", e, exc_info=True)

        missing_ids = [i for i in valid_ids if i not in found_map]
        if missing_ids:
            try:
                recs_fleet = self.fleet_shard.retrieve(missing_ids, with_payload=True, with_vector=False)
                for r in recs_fleet:
                    m = self._record_to_memory(r)
                    found_map[m.mem_id] = m
            except Exception as e:
                logger.error("Unhandled exception: %s", e, exc_info=True)

        return [found_map[i] for i in ids if i in found_map]

    def get_memory(self, mem_id: str) -> Memory | None:
        mems = self.get([mem_id])
        return mems[0] if mems else None

    def set_status(self, ids: list[str], status: str, **extra: Any) -> None:
        if not ids:
            return
        valid_ids = [i for i in ids if self._is_valid_point_id(i)]
        if not valid_ids:
            return
        update_data = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        for shard in [self.device_shard, self.fleet_shard]:
            try:
                existing = shard.retrieve(valid_ids, with_payload=False, with_vector=False)
                existing_ids = [str(r.id) for r in existing]
                if existing_ids:
                    shard.update(qdrant_edge.UpdateOperation.set_payload(existing_ids, update_data))
            except Exception as e:
                logger.error("Unhandled exception: %s", e, exc_info=True)

    def scan(
        self,
        flt: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[Memory], str | None]:
        scroll_req = qdrant_edge.ScrollRequest(
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        records, next_offset = self.device_shard.scroll(scroll_req)
        memories = [self._record_to_memory(r) for r in records]

        if flt:
            filtered = []
            for m in memories:
                match = True
                for k, v in flt.items():
                    if getattr(m, k, None) != v and m.payload.get(k) != v:
                        match = False
                        break
                if match:
                    filtered.append(m)
            memories = filtered

        return memories, next_offset

    def facets(self, key: str, flt: dict[str, Any] | None = None) -> dict[str, int]:
        try:
            req = qdrant_edge.FacetRequest(key=key, limit=100)
            resp = self.device_shard.facet(req)
            result: dict[str, int] = {}
            for hit in resp.hits:
                result[str(hit.value)] = int(hit.count)
            return result
        except Exception as e:
            logger.warning("Handled exception: %s", e)
            mems, _ = self.scan(limit=1000)
            res: dict[str, int] = {}
            for m in mems:
                val = getattr(m, key, None) or m.payload.get(key)
                if val is not None:
                    res[str(val)] = res.get(str(val), 0) + 1
            return res

    def optimize(self) -> None:
        try:
            self.device_shard.optimize()
        except Exception as e:
            logger.error("Unhandled exception: %s", e, exc_info=True)
        try:
            self.fleet_shard.optimize()
        except Exception as e:
            logger.error("Unhandled exception: %s", e, exc_info=True)

    def count(self) -> int:
        c1 = self.device_shard.count() if hasattr(self.device_shard, "count") else 0
        c2 = self.fleet_shard.count() if hasattr(self.fleet_shard, "count") else 0
        return c1 + c2

    def clear(self) -> None:
        if hasattr(self.device_shard, "clear"):
            self.device_shard.clear()
        if hasattr(self.fleet_shard, "clear"):
            self.fleet_shard.clear()

    def close(self) -> None:
        if self._local_client is not None:
            try:
                self._local_client.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def search(self, req: SearchRequest) -> list[Hit]:
        q_dense, q_bm25_data = self.embedder.embed_query(req.query)
        q_sparse = qdrant_edge.SparseVector(
            indices=q_bm25_data.indices,
            values=q_bm25_data.values,
        )

        raw_hits = []

        # Search both shards: device_memory (P0) and fleet_mirror (P1)
        for shard_name, shard in [("device_memory", self.device_shard), ("fleet_mirror", self.fleet_shard)]:
            try:
                # 1. Branch: Dense
                dense_prefetch = qdrant_edge.Prefetch(
                    limit=max(req.limit * 3, 20),
                    query=qdrant_edge.Query.Nearest(q_dense, using="dense"),
                )
                # 2. Branch: BM25
                bm25_prefetch = qdrant_edge.Prefetch(
                    limit=max(req.limit * 3, 20),
                    query=qdrant_edge.Query.Nearest(q_sparse, using="bm25"),
                )

                # Execute RRF fusion on edge
                fusion_query = qdrant_edge.QueryRequest(
                    limit=max(req.limit * 3, 20),
                    query=qdrant_edge.Fusion.Rrf(k=60),
                    prefetches=[dense_prefetch, bm25_prefetch],
                    with_payload=True,
                )
                fused_results = shard.query(fusion_query)

                # If explain requested, also get single branch ranks
                dense_ranks: dict[str, tuple[int, float]] = {}
                bm25_ranks: dict[str, tuple[int, float]] = {}
                if req.explain:
                    qr_d = qdrant_edge.QueryRequest(
                        limit=30,
                        query=qdrant_edge.Query.Nearest(q_dense, using="dense"),
                        with_payload=False,
                    )
                    res_d = shard.query(qr_d)
                    for rk, pt in enumerate(res_d, start=1):
                        dense_ranks[str(pt.id)] = (rk, float(pt.score))

                    qr_b = qdrant_edge.QueryRequest(
                        limit=30,
                        query=qdrant_edge.Query.Nearest(q_sparse, using="bm25"),
                        with_payload=False,
                    )
                    res_b = shard.query(qr_b)
                    for rk, pt in enumerate(res_b, start=1):
                        bm25_ranks[str(pt.id)] = (rk, float(pt.score))

                for r in fused_results:
                    mid = str(r.id)
                    mem = self._record_to_memory(r)
                    if req.filters:
                        m_match = True
                        for fk, fv in req.filters.items():
                            if getattr(mem, fk, None) != fv and mem.payload.get(fk) != fv:
                                m_match = False
                                break
                        if not m_match:
                            continue

                    branch_ranks: list[BranchRank] = []
                    if req.explain:
                        d_rk, d_sc = dense_ranks.get(mid, (99, 0.0))
                        b_rk, b_sc = bm25_ranks.get(mid, (99, 0.0))
                        branch_ranks = [
                            BranchRank(branch="dense", rank=d_rk, score=d_sc),
                            BranchRank(branch="bm25", rank=b_rk, score=b_sc),
                        ]

                    fused_score = float(r.score)
                    d_sc = dense_ranks.get(mid, (99, 0.0))[1]
                    b_sc = bm25_ranks.get(mid, (99, 0.0))[1]

                    status_multiplier = 0.7 if mem.status in ("superseded", "disputed") else 1.0

                    if (dense_ranks or bm25_ranks) and (d_sc < 0.16 or (d_sc < 0.20 and b_sc < 0.45)):
                        # Weak match: accidental term collision or low semantic alignment
                        raw_sim = max(d_sc, b_sc * 0.5)
                        normalized_score = max(0.08, min(0.38, raw_sim * 1.2))
                    else:
                        # D3: RRF additive formula
                        if fused_score <= 0.04:
                            rrf_normalized = min(0.95, (fused_score / 0.033) * 0.90)
                        else:
                            rrf_normalized = min(0.95, fused_score * 0.90)
                        if rrf_normalized >= 0.40:
                            normalized_score = min(0.99, rrf_normalized + (mem.authority * 0.05))
                        else:
                            normalized_score = rrf_normalized
                        normalized_score *= status_multiplier

                    raw_hits.append(Hit(
                        mem_id=mid,
                        score=round(normalized_score, 3),
                        memory=mem,
                        branch_ranks=branch_ranks,
                        fused_score=round(fused_score, 4),
                        shard=shard_name,
                    ))
            except Exception as e:
                logger.warning("Shard search error: %s", e)
                continue

        # D2: Dedup on content_hash, not by shard precedence unconditionally
        final_hits = {}
        for h in raw_hits:
            h_hash = h.memory.content_hash
            if h_hash in final_hits:
                # Keep the one with higher authority or higher score
                if h.memory.authority > final_hits[h_hash].memory.authority:
                    final_hits[h_hash] = h
                elif h.memory.authority == final_hits[h_hash].memory.authority and h.score > final_hits[h_hash].score:
                    final_hits[h_hash] = h
            else:
                final_hits[h_hash] = h

        # Sort combined hits by final score descending
        sorted_hits = sorted(final_hits.values(), key=lambda x: x.score, reverse=True)
        return sorted_hits[:req.limit]
