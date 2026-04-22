"""FAISS-backed in-memory data store with synthetic CRM data.

All data lives in-memory with FAISS indices for vector similarity search
(orders, policies).  The embed() helper calls OpenAI for query-time embeddings.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import faiss
import numpy as np

# ── OpenAI embedding helper ──────────────────────────────────────────────


def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI."""
    import openai

    client = openai.OpenAI()
    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [d.embedding for d in resp.data]


# ── Synthetic data ───────────────────────────────────────────────────────

_EMBEDDING_DIM = 1536  # text-embedding-3-small


def _rand_vec(seed: int) -> list[float]:
    """Deterministic pseudo-random unit vector for a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(_EMBEDDING_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


_now = datetime.utcnow()

ORDERS: list[dict[str, Any]] = [
    {
        "_id": "ord_001",
        "user_id": "user_001",
        "product_name": "Bluetooth Earbuds Pro",
        "unit_price": 79.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=12)).isoformat(),
        "shipping_address": {"country": "US", "state": "CA", "city": "San Jose"},
        "embedding": _rand_vec(101),
    },
    {
        "_id": "ord_002",
        "user_id": "user_002",
        "product_name": "QuickDry Compact Dryer",
        "unit_price": 249.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=20)).isoformat(),
        "shipping_address": {"country": "US", "state": "NY", "city": "New York"},
        "embedding": _rand_vec(102),
    },
    {
        "_id": "ord_003",
        "user_id": "user_003",
        "product_name": "HyperClick Gaming Mouse",
        "unit_price": 59.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=5)).isoformat(),
        "shipping_address": {"country": "US", "state": "TX", "city": "Austin"},
        "embedding": _rand_vec(103),
    },
    {
        "_id": "ord_004",
        "user_id": "user_004",
        "product_name": "PureAir 300 Air Purifier",
        "unit_price": 189.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=8)).isoformat(),
        "shipping_address": {"country": "US", "state": "WA", "city": "Seattle"},
        "embedding": _rand_vec(104),
    },
    {
        "_id": "ord_005",
        "user_id": "user_005",
        "product_name": "BrewMaster 5000 Coffee Maker",
        "unit_price": 129.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=15)).isoformat(),
        "shipping_address": {"country": "US", "state": "IL", "city": "Chicago"},
        "embedding": _rand_vec(105),
    },
    {
        "_id": "ord_006",
        "user_id": "user_006",
        "product_name": "SoundStage 7.1 Speaker System",
        "unit_price": 349.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=10)).isoformat(),
        "shipping_address": {"country": "US", "state": "FL", "city": "Miami"},
        "embedding": _rand_vec(106),
    },
    {
        "_id": "ord_007",
        "user_id": "user_007",
        "product_name": "Halloween Deluxe Costume",
        "unit_price": 45.00,
        "quantity": 1,
        "status": "shipped",  # NOT delivered — hallucination scenario
        "order_date": (_now - timedelta(days=3)).isoformat(),
        "shipping_address": {"country": "US", "state": "OR", "city": "Portland"},
        "embedding": _rand_vec(107),
    },
    {
        "_id": "ord_008",
        "user_id": "user_008",
        "product_name": "Wireless Charging Pad",
        "unit_price": 34.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=7)).isoformat(),
        "shipping_address": {"country": "US", "state": "CA", "city": "Los Angeles"},
        "embedding": _rand_vec(108),
    },
    {
        "_id": "ord_009",
        "user_id": "user_009",
        "product_name": "USB-C Hub Adapter",
        "unit_price": 29.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=14)).isoformat(),
        "shipping_address": {"country": "US", "state": "MA", "city": "Boston"},
        "embedding": _rand_vec(109),
    },
    {
        "_id": "ord_010",
        "user_id": "user_010",
        "product_name": "Ergonomic Office Chair",
        "unit_price": 399.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=30)).isoformat(),
        "shipping_address": {"country": "US", "state": "CO", "city": "Denver"},
        "embedding": _rand_vec(110),
    },
    {
        "_id": "ord_011",
        "user_id": "user_011",
        "product_name": "Noise-Cancelling Headphones",
        "unit_price": 199.99,
        "quantity": 1,
        "status": "processing",
        "order_date": (_now - timedelta(days=2)).isoformat(),
        "shipping_address": {"country": "US", "state": "IL", "city": "Chicago"},
        "embedding": _rand_vec(111),
    },
    {
        "_id": "ord_012",
        "user_id": "user_012",
        "product_name": "Smart Home Hub",
        "unit_price": 149.99,
        "quantity": 1,
        "status": "delivered",
        "order_date": (_now - timedelta(days=25)).isoformat(),
        "shipping_address": {"country": "US", "state": "AZ", "city": "Phoenix"},
        "embedding": _rand_vec(112),
    },
]

REFUND_REQUESTS: list[dict[str, Any]] = [
    {
        "_id": "rr_001",
        "user_id": "user_001",
        "amount": 79.99,
        "currency": "USD",
        "status": "investigation",
        "description": "Customer unhappy with bluetooth earbuds",
        "created_at": (_now - timedelta(days=2)).isoformat(),
    },
    {
        "_id": "rr_002",
        "user_id": "user_002",
        "amount": 249.99,
        "currency": "USD",
        "status": "refund in progress",
        "description": "Dryer returned — repeated complaints",
        "created_at": (_now - timedelta(days=5)).isoformat(),
    },
]

TICKETS: list[dict[str, Any]] = [
    {
        "_id": "tkt_001",
        "user_id": "user_001",
        "title": "Bluetooth earbuds quality issue",
        "status": "open",
        "created_at": (_now - timedelta(days=3)).isoformat(),
    },
    {
        "_id": "tkt_002",
        "user_id": "user_002",
        "title": "Angry customer — dryer return",
        "status": "escalated",
        "created_at": (_now - timedelta(days=6)).isoformat(),
    },
    {
        "_id": "tkt_003",
        "user_id": "user_003",
        "title": "Gaming mouse defect report",
        "status": "open",
        "created_at": (_now - timedelta(days=1)).isoformat(),
    },
    {
        "_id": "tkt_004",
        "user_id": "user_010",
        "title": "Abusive language — flagged",
        "status": "open",
        "created_at": (_now - timedelta(hours=12)).isoformat(),
    },
]

# Policies: current + expired versions for drift testing
POLICIES: list[dict[str, Any]] = [
    {
        "_id": "pol_us_v3",
        "version": "v3",
        "region": "US",
        "refund_window_days": 30,
        "max_refund_amount": 500.00,
        "auto_approve_threshold": 100.00,
        "escalation_required_above": 300.00,
        "effective_from": (_now - timedelta(days=60)).isoformat(),
        "effective_until": None,  # current
        "description": "US refund policy — 30-day window, auto-approve under $100",
        "embedding": _rand_vec(201),
    },
    {
        "_id": "pol_us_v2",
        "version": "v2",
        "region": "US",
        "refund_window_days": 14,
        "max_refund_amount": 250.00,
        "auto_approve_threshold": 50.00,
        "escalation_required_above": 200.00,
        "effective_from": (_now - timedelta(days=180)).isoformat(),
        "effective_until": (_now - timedelta(days=60)).isoformat(),  # expired
        "description": "US refund policy v2 — stricter 14-day window (expired)",
        "embedding": _rand_vec(202),
    },
    {
        "_id": "pol_eu_v3",
        "version": "v3",
        "region": "EU",
        "refund_window_days": 45,
        "max_refund_amount": 600.00,
        "auto_approve_threshold": 150.00,
        "escalation_required_above": 400.00,
        "effective_from": (_now - timedelta(days=60)).isoformat(),
        "effective_until": None,
        "description": "EU refund policy — 45-day window per consumer protection directive",
        "embedding": _rand_vec(203),
    },
    {
        "_id": "pol_eu_v2",
        "version": "v2",
        "region": "EU",
        "refund_window_days": 30,
        "max_refund_amount": 400.00,
        "auto_approve_threshold": 100.00,
        "escalation_required_above": 300.00,
        "effective_from": (_now - timedelta(days=180)).isoformat(),
        "effective_until": (_now - timedelta(days=60)).isoformat(),
        "description": "EU refund policy v2 — 30-day window (expired)",
        "embedding": _rand_vec(204),
    },
]


# ── FAISS indices (built once at import time) ────────────────────────────


def _build_index(items: list[dict[str, Any]]) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from items with 'embedding' keys."""
    vecs = np.array([item["embedding"] for item in items], dtype=np.float32)
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(_EMBEDDING_DIM)
    index.add(vecs)
    return index


_order_index = _build_index(ORDERS)
_policy_index = _build_index(POLICIES)


# ── Public query API ─────────────────────────────────────────────────────


class _Collection:
    """Minimal collection-like interface over in-memory lists + FAISS."""

    def __init__(
        self, items: list[dict[str, Any]], index: faiss.IndexFlatIP | None = None
    ):
        self._items = items
        self._index = index

    # Simple filter-based find (no vector search)
    def find(
        self, filter_dict: dict | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        results = []
        for item in self._items:
            if filter_dict and not all(
                item.get(k) == v for k, v in filter_dict.items()
            ):
                continue
            results.append({k: v for k, v in item.items() if k != "embedding"})
            if len(results) >= limit:
                break
        return results

    # Vector similarity search
    def vector_search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filter_dict: dict | None = None,
    ) -> list[dict[str, Any]]:
        if self._index is None:
            return self.find(filter_dict, limit=limit)

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)
        # Search more candidates than limit to allow for filtering
        k = min(len(self._items), max(limit * 3, 20))
        scores, indices = self._index.search(q, k)

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            item = self._items[idx]
            if filter_dict and not all(
                item.get(k_) == v for k_, v in filter_dict.items()
            ):
                continue
            results.append({k_: v for k_, v in item.items() if k_ != "embedding"})
            if len(results) >= limit:
                break
        return results


class _Database:
    """Minimal database-like interface with named collections."""

    def __init__(self):
        self.orders = _Collection(ORDERS, _order_index)
        self.refund_requests = _Collection(REFUND_REQUESTS)
        self.tickets = _Collection(TICKETS)
        self.policies = _Collection(POLICIES, _policy_index)


_db = _Database()


def get_db() -> _Database:
    """Return the in-memory CRM database (FAISS-backed)."""
    return _db
