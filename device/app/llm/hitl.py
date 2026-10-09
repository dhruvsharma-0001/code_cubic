"""device/app/llm/hitl.py

Human-in-the-Loop (HITL) approval gate.
Gate A: Egress consent before frontier API calls.
Gate B: Knowledge promotion of cloud-generated answers.
All approvals/denials logged for audit trail.
"""
from __future__ import annotations
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

@dataclass
class HITLRequest:
    token: str
    query: str
    query_hash: str
    model_to_use: str
    estimated_cost_usd: float
    data_preview: str
    privacy_tier: str
    created_at: float
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    denied_reason: Optional[str] = None

_PENDING: Dict[str, HITLRequest] = {}

def create_hitl_request(
    query: str, 
    data_preview: str, 
    model: str = "gpt-4o-mini", 
    privacy_tier: str = "T2"
) -> HITLRequest:
    token = str(uuid.uuid4())
    query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
    
    est_cost = 0.001 if "gpt-4o" in model else 0.01
    
    req = HITLRequest(
        token=token,
        query=query,
        query_hash=query_hash,
        model_to_use=model,
        estimated_cost_usd=est_cost,
        data_preview=data_preview[:500] + ("..." if len(data_preview) > 500 else ""),
        privacy_tier=privacy_tier,
        created_at=time.time()
    )
    _PENDING[token] = req
    logger.info(f"Created HITL request {token} for query hash {query_hash}")
    return req

def approve_hitl(token: str, user_id: str = "operator") -> bool:
    req = _PENDING.get(token)
    if not req:
        logger.warning(f"HITL token {token} not found for approval")
        return False
    if req.approved is not None:
        logger.warning(f"HITL token {token} already processed")
        return False
        
    req.approved = True
    req.approved_by = user_id
    logger.info(f"HITL request {token} approved by {user_id}")
    return True

def deny_hitl(token: str, reason: str = "") -> bool:
    req = _PENDING.get(token)
    if not req:
        logger.warning(f"HITL token {token} not found for denial")
        return False
    if req.approved is not None:
        logger.warning(f"HITL token {token} already processed")
        return False
        
    req.approved = False
    req.denied_reason = reason
    logger.info(f"HITL request {token} denied. Reason: {reason}")
    return True

def get_hitl_status(token: str) -> Optional[HITLRequest]:
    return _PENDING.get(token)

def list_pending() -> List[HITLRequest]:
    return [req for req in _PENDING.values() if req.approved is None]

def cleanup_expired(max_age_seconds: int = 3600) -> int:
    now = time.time()
    expired_keys = [
        k for k, v in _PENDING.items() 
        if (now - v.created_at) > max_age_seconds
    ]
    for k in expired_keys:
        del _PENDING[k]
        
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired HITL requests")
    return len(expired_keys)
