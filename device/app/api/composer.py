from __future__ import annotations

import re
import logging
from typing import NamedTuple, Any
from pydantic import BaseModel, Field

from device.app.memory.models import Hit, Memory

from device.app.llm.router import LLMRouter, RouterContext, LLMBackend
from device.app.llm.local_slm import LocalSLMService
from device.app.llm.frontier import FrontierCaller
from device.app.llm.hitl import create_hitl_request
from device.app.llm.claim_validator import ClaimValidator

logger = logging.getLogger(__name__)

class ExtractiveCitation(BaseModel):
    mem_id: str
    display_id: str
    title: str
    authority: int
    source: str


class AnswerResult(BaseModel):
    answer: str
    confidence: float
    confidence_label: str
    citations: list[ExtractiveCitation]
    low_confidence: bool = False
    model_tag: str = "Extractive Deterministic (Zero Cloud Inference)"
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractiveComposer:
    def compose(
        self,
        query: str,
        hits: list[Hit],
        min_confidence: float = 0.40,
    ) -> AnswerResult:
        if not hits:
            return AnswerResult(
                answer="Not enough evidence found in local air-gapped memory shards for the specified query.",
                confidence=0.0,
                confidence_label="NONE",
                citations=[],
                low_confidence=True,
            )

        top_hit = hits[0]
        top_score = top_hit.score

        if top_score < min_confidence:
            return AnswerResult(
                answer="Not enough evidence in local memory to answer with sufficient confidence. Verify asset ID or query terms.",
                confidence=round(top_score, 2),
                confidence_label="LOW",
                citations=[],
                low_confidence=True,
            )

        # Build verified citations
        citations: list[ExtractiveCitation] = []
        key_sentences: list[str] = []
        seen_sentences = set()

        for h in hits[:3]:
            # Generate short display ID like M-1042 or first 6 chars of mem_id
            m_id = h.mem_id
            disp_id = f"M-{m_id[:4].upper()}" if not m_id.startswith("M-") else m_id
            title = h.memory.payload.get("title") or (h.memory.content[:40] + "...")
            cit = ExtractiveCitation(
                mem_id=h.mem_id,
                display_id=disp_id,
                title=title,
                authority=h.memory.authority,
                source=f"{h.memory.source_device} ({h.memory.site_id})",
            )
            citations.append(cit)

            # Extract best matching sentence from content
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", h.memory.content) if len(s.strip()) > 15]
            for s in sentences:
                if s not in seen_sentences:
                    seen_sentences.add(s)
                    key_sentences.append(f"{s} [{disp_id}]")
                    break

        if not citations:
            return AnswerResult(
                answer="Not enough evidence. No verified citations available.",
                confidence=0.0,
                confidence_label="NONE",
                citations=[],
                low_confidence=True,
            )

        # Extractive answer composition: assemble verified factual claims
        # Highlight key parameters if present
        composed_text = " ".join(key_sentences)

        conf_label = "HIGH" if top_score >= 0.80 else ("MEDIUM" if top_score >= 0.60 else "LOW")

        return AnswerResult(
            answer=composed_text,
            confidence=round(top_score, 2),
            confidence_label=conf_label,
            citations=citations,
            low_confidence=False,
            model_tag="Extractive Engine (Zero Cloud Inference)",
        )


class SmartAnswerEngine:
    def __init__(self):
        self.extractive_composer = ExtractiveComposer()
        self.router = LLMRouter()
        self.local_slm = LocalSLMService()
        self.frontier = FrontierCaller()
        self.validator = ClaimValidator()

    def compose(
        self,
        query: str,
        hits: list[Hit],
        ctx: RouterContext | None = None,
        memories: dict[str, Memory] | None = None,
        min_confidence: float = 0.40,
    ) -> AnswerResult:
        if not hits:
            return self.extractive_composer.compose(query, hits, min_confidence)
            
        if ctx is None:
            ctx = RouterContext(
                is_online=False,
                has_pii=False,
                query=query,
                top_qdrant_score=hits[0].score if hits else 0.0,
                score_margin=(hits[0].score - hits[1].score) if len(hits) > 1 else 0.0,
                max_authority=max([h.memory.authority for h in hits]) if hits else 1,
                memory_scope="global",
                frontier_approved=False,
                link_state="OFFLINE",
                device_tier="standard",
                jev_available=True,
                jev_complexity="SIMPLE",
                jev_confidence=0.0,
                jev_sensitivity="PUBLIC",
                safety_critical=False,
                disputed_present=False
            )
            
        memories = memories or {h.mem_id: h.memory for h in hits}

        decision = self.router.route(ctx)
        logger.info(f"Routing decision: {decision.backend.name} (Code: {decision.route_code}, Reason: {decision.reason})")
        
        metadata = {
            "route_code": decision.route_code,
            "backend": decision.backend.name,
            "reason": decision.reason,
            "requires_hitl": decision.requires_hitl,
            "warnings": [],
            "hitl_token": None
        }

        ans = None
        cite_ids = []
        conf = 0.0
        model_tag = ""

        if decision.backend == LLMBackend.LOCAL_SLM:
            if self.local_slm.is_available():
                try:
                    ans, cite_ids, conf, model_tag = self.local_slm.call(ctx.query, hits)
                    metadata["model_used"] = model_tag
                except Exception as e:
                    logger.error(f"Local SLM failed: {e}")
            else:
                logger.warning("Local SLM routed but not available. Falling back to extractive.")
                
        elif decision.backend == LLMBackend.FRONTIER:
            if decision.requires_hitl and not ctx.frontier_approved:
                preview = "\n".join([h.memory.content[:100] for h in hits[:2]])
                hitl_req = create_hitl_request(ctx.query, preview)
                metadata["hitl_token"] = hitl_req.token
                ans = "This query requires Cloud AI. Awaiting operator approval (HITL Gate A)."
                cite_ids = []
                conf = 0.0
                model_tag = "HITL Gate"
            else:
                try:
                    ans, cite_ids, conf, model_tag = self.frontier.call(ctx.query, hits)
                    metadata["model_used"] = model_tag
                except Exception as e:
                    logger.error(f"Frontier failed: {e}")
                    
        elif decision.backend == LLMBackend.HITL:
            preview = "\n".join([h.memory.content[:100] for h in hits[:2]])
            hitl_req = create_hitl_request(ctx.query, preview, model="human")
            metadata["hitl_token"] = hitl_req.token
            ans = "This query has been flagged for human review (HITL Gate A)."
            cite_ids = []
            conf = 0.0
            model_tag = "HITL"

        if ans is not None:
            warnings = self.validator.validate(ans, hits, memories)
            metadata["warnings"] = warnings
            citations = self._build_citations_from_ids(cite_ids, hits, memories)
            
            conf_label = "HIGH" if conf >= 0.80 else ("MEDIUM" if conf >= 0.60 else "LOW")
            
            return AnswerResult(
                answer=ans,
                confidence=conf,
                confidence_label=conf_label,
                citations=citations,
                low_confidence=(conf < min_confidence),
                model_tag=model_tag,
                metadata=metadata
            )

        # Fallback to extractive
        res = self.extractive_composer.compose(ctx.query, hits, min_confidence)
        res.metadata = metadata
        return res

    def _build_citations_from_ids(self, cite_ids: list[str], hits: list[Hit], memories: dict[str, Memory]) -> list[ExtractiveCitation]:
        citations = []
        for cid in cite_ids:
            matching_hit = next((h for h in hits if h.mem_id == cid), None)
            if matching_hit and matching_hit.mem_id in memories:
                h = matching_hit
                m_id = h.mem_id
                disp_id = f"M-{m_id[:4].upper()}" if not m_id.startswith("M-") else m_id
                title = h.memory.payload.get("title") or (h.memory.content[:40] + "...")
                cit = ExtractiveCitation(
                    mem_id=h.mem_id,
                    display_id=disp_id,
                    title=title,
                    authority=h.memory.authority,
                    source=f"{h.memory.source_device} ({h.memory.site_id})",
                )
                citations.append(cit)
        return citations
