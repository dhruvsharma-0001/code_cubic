"""device/app/memory/answer.py — Answer composers: Legacy Extractive and new SmartAnswerEngine."""

import re
import logging
from device.app.memory.models import Hit, Memory

logger = logging.getLogger(__name__)

class AnswerComposer:
    def compose(self, query: str, hits: list[Hit], memories: dict[str, Memory]) -> tuple[str, list[dict], float]:
        if not hits:
            return "No relevant memories found.", [], 0.0

        query_words = set(re.findall(r'\w+', query.lower()))
        best_sentences = []
        citations = []
        
        for hit in hits[:3]:
            mem = memories.get(hit.mem_id)
            if not mem:
                continue
                
            sentences = [s.strip() for s in re.split(r'[.!?]+', mem.content) if s.strip()]
            if not sentences:
                sentences = [mem.content]
                
            best_sentence = ""
            best_overlap = -1
            
            for sentence in sentences:
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                overlap = len(query_words.intersection(sentence_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence
                    
            if not best_sentence:
                best_sentence = mem.content[:200]
                
            best_sentences.append(best_sentence)
            citations.append({
                "mem_id": hit.mem_id,
                "snippet": best_sentence,
                "authority": getattr(mem, "authority", 0),
                "kind": getattr(mem, "kind", "note"),
            })
            
        answer_text = " | ".join(best_sentences)
        
        # Calculate confidence
        top_score = hits[0].score if hits else 0.0
        expected_max = 0.5  # arbitrary reference max for normalisation
        c_top = min(1.0, top_score / expected_max)
        
        c_gap = 0.0
        if len(hits) > 1 and top_score > 0:
            c_gap = (top_score - hits[1].score) / top_score
            
        c_count = min(1.0, len(hits) / 3.0)
        
        confidence = (0.5 * c_top) + (0.3 * c_gap) + (0.2 * c_count)
        
        return answer_text, citations, max(0.0, min(1.0, confidence))

from device.app.llm.router import LLMRouter, RouterContext, LLMBackend
from device.app.llm.local_slm import LocalSLMService
from device.app.llm.frontier import FrontierCaller
from device.app.llm.hitl import create_hitl_request
from device.app.llm.claim_validator import ClaimValidator

class SmartAnswerEngine:
    def __init__(self):
        self.extractive_composer = AnswerComposer()
        self.router = LLMRouter()
        self.local_slm = LocalSLMService()
        self.frontier = FrontierCaller()
        self.validator = ClaimValidator()

    def generate(self, ctx: RouterContext, hits: list[Hit], memories: dict[str, Memory]) -> tuple[str, list[dict], float, dict]:
        """
        Generate answer based on routing decision.
        Returns: answer_text, citations, confidence, metadata (including warnings and hitl token)
        """
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

        if decision.backend == LLMBackend.LOCAL_SLM:
            if self.local_slm.is_available():
                try:
                    ans, cite_ids, conf, model_tag = self.local_slm.call(ctx.query, hits)
                    metadata["model_used"] = model_tag
                    warnings = self.validator.validate(ans, hits, memories)
                    metadata["warnings"] = warnings
                    citations = self._build_citations_from_ids(cite_ids, hits, memories)
                    return ans, citations, conf, metadata
                except Exception as e:
                    logger.error(f"Local SLM failed: {e}")
            else:
                logger.warning("Local SLM routed but not available. Falling back to extractive.")
                
        elif decision.backend == LLMBackend.FRONTIER:
            if decision.requires_hitl and not ctx.frontier_approved:
                # Issue HITL request
                preview = "\n".join([h.memory.content[:100] for h in hits[:2]])
                hitl_req = create_hitl_request(ctx.query, preview)
                metadata["hitl_token"] = hitl_req.token
                return "This query requires Cloud AI. Awaiting operator approval (HITL Gate A).", [], 0.0, metadata
            else:
                try:
                    ans, cite_ids, conf, model_tag = self.frontier.call(ctx.query, hits)
                    metadata["model_used"] = model_tag
                    warnings = self.validator.validate(ans, hits, memories)
                    metadata["warnings"] = warnings
                    citations = self._build_citations_from_ids(cite_ids, hits, memories)
                    return ans, citations, conf, metadata
                except Exception as e:
                    logger.error(f"Frontier failed: {e}")
                    
        elif decision.backend == LLMBackend.HITL:
            preview = "\n".join([h.memory.content[:100] for h in hits[:2]])
            hitl_req = create_hitl_request(ctx.query, preview, model="human")
            metadata["hitl_token"] = hitl_req.token
            return "This query has been flagged for human review (HITL Gate A).", [], 0.0, metadata

        # Fallback to extractive
        ans, citations, conf = self.extractive_composer.compose(ctx.query, hits, memories)
        metadata["backend"] = "EXTRACTIVE"
        return ans, citations, conf, metadata

    def _build_citations_from_ids(self, cite_ids: list[str], hits: list[Hit], memories: dict[str, Memory]) -> list[dict]:
        citations = []
        for cid in cite_ids:
            # find matching hit
            matching_hit = next((h for h in hits if h.mem_id == cid), None)
            if matching_hit and matching_hit.mem_id in memories:
                mem = memories[matching_hit.mem_id]
                citations.append({
                    "mem_id": mem.mem_id,
                    "snippet": mem.content[:200], # just a preview
                    "authority": mem.authority,
                    "kind": mem.kind
                })
        return citations
