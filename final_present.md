12# EdgeMind — Project Technology Overview

## What it is

EdgeMind is an **offline-first industrial AI memory system** for factories, field devices, and privacy-sensitive operations. It lets operators search maintenance knowledge, capture field notes, resolve conflicts, and synchronize approved knowledge without depending on public-cloud AI services.

## Core idea

```
Operator/UI → Edge device → Local memory search → LAN peer (if needed) → Fleet hub (only when policy permits)
```

The preferred route is always local. This reduces latency, avoids routine internet egress, and keeps sensitive operational data on-device.

## Architecture

| Layer | Technology | Responsibility |
|---|---|---|
| Industrial console | Next.js, TypeScript, Tailwind CSS, Recharts, Lucide | Device status, search, note capture, conflicts, audit and benchmark views |
| Device API | Python, FastAPI, Uvicorn, Pydantic | REST APIs, health checks, authentication, routing, ingestion and search |
| Intelligence | Local SLM service + deterministic extractive answers | Generates answers from retrieved evidence and anchors citations to source records |
| Memory | Qdrant Edge / embedded Qdrant | Stores and retrieves local vector memories |
| Retrieval | FastEmbed BGE dense vectors + BM25 sparse vectors + RRF | Semantic and keyword search fused into one ranked result set |
| Durable state | SQLite with WAL | Memory lifecycle, audit events, sync outbox, conflicts and metadata |
| Device mesh | HTTPX over authenticated local HTTP | Queries a peer edge node when local confidence is insufficient |
| Fleet hub | FastAPI control plane | Receives permitted sync messages, heartbeats and fleet-level knowledge |

## End-to-end flow

1. A manual or field note is ingested on an edge device.
2. The policy layer detects PII/secrets and redacts or marks unsafe records as `local_only`.
3. Safe content is embedded into dense and sparse representations and stored locally.
4. A user query searches the local shard first using hybrid retrieval.
5. If confidence is low, the Smart Query Router can ask a configured LAN peer—never an arbitrary internet endpoint.
6. The answer service returns evidence-backed text with citations. If evidence is missing, it raises a knowledge gap instead of inventing an answer.
7. Approved changes are written to the SQLite ledger and selectively synchronized through the hub.

## Important technical features

### Hybrid RAG

- **Dense retrieval:** FastEmbed `BAAI/bge-small-en-v1.5` understands semantic similarity.
- **Sparse retrieval:** BM25 matches exact equipment IDs, error codes, names and technical terms.
- **Fusion:** Reciprocal Rank Fusion (RRF) combines both result lists so neither semantic nor exact-match recall is lost.
- **Fallback:** The memory store supports an in-process Qdrant client path when the edge binding is unavailable.

### Smart three-tier routing

| Tier | Route | Purpose |
|---|---|---|
| 1 | Local embedded shard | Fastest path; no network egress |
| 2 | Authenticated Wi-Fi/LAN peer | Finds knowledge held by another plant/device node |
| 3 | Fleet hub sync path | Controlled fleet-level coordination when policy allows |

### Security and governance

- API-key-protected device, peer and administrative routes.
- PII and credential-pattern detection with redaction.
- Policy decision can force records to remain `local_only` and block synchronization.
- SHA-256 content hashing supports provenance and tamper-aware audit records.
- SQLite WAL provides durable local writes even in intermittent-connectivity environments.
- Conflict arbitration prioritizes authority, version/vector state, Lamport time, then human sign-off.

### Reliability changes in the current implementation

- Resilient local Qdrant fallback and error logging for embedded-memory failures.
- Peer requests include device API authentication.
- Stronger LLM timeout/fallback behavior.
- Improved claim validation, policy decisions, answer/search behavior and evaluation tooling.
- Edge-case coverage for authentication, policy/PII handling, LLM availability and memory initialization paths.

## Services and ports

| Service | Default port | Role |
|---|---:|---|
| Next.js UI | 3000 | Operator console |
| Fleet Hub | 8000 | Central coordination |
| Device A | 8001 | Edge node / local memory |
| Device B | 8002 | Peer edge node / local memory |

## How to run

```bash
pip install -r requirements.txt
cd ui && npm install && cd ..
python scripts/run_project.py
```

Then open `http://localhost:3000`. API documentation is available at each FastAPI service's `/docs` route.

## How to validate

```bash
python -m pytest device/tests/
python eval/eval_harness.py
cd ui && npm run build
```

## One-line presentation pitch

**EdgeMind gives industrial teams fast, evidence-backed AI memory at the edge: data stays local by default, search works offline, peers share only through policy-controlled LAN/fleet routes, and every important decision is auditable.**
