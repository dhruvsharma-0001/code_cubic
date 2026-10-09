import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

logger = logging.getLogger(__name__)

class LLMBackend(Enum):
    LOCAL_SLM = auto()
    FRONTIER = auto()
    HITL = auto()
    EXTRACTIVE = auto()
    SEMANTIC_CACHE = auto()

@dataclass
class RouterContext:
    is_online: bool
    has_pii: bool
    query: str
    top_qdrant_score: float
    score_margin: float
    max_authority: int
    memory_scope: str
    frontier_approved: bool
    link_state: str
    device_tier: str
    jev_available: bool = False
    jev_complexity: str = ""
    jev_confidence: float = 0.0
    jev_sensitivity: str = ""
    safety_critical: bool = False
    disputed_present: bool = False

@dataclass
class RoutingDecision:
    backend: LLMBackend
    route_code: str
    reason: str
    selector_used: str
    requires_hitl: bool
    confidence: float
    metadata: dict

class JeVClassifier:
    ADVANCED_KEYWORDS = {
        "compare", "analyse", "analyze", "explain", "summarize", "why", 
        "how does", "difference between", "trade-off", "plan", "recommend", 
        "design", "architecture", "diagnose", "troubleshoot"
    }
    
    SAFETY_KEYWORDS = {
        "torque", "pressure", "psi", "bar", "nm", "clearance", "lockout", 
        "voltage", "amp", "temperature", "limit", "maximum", "minimum", "calibrat"
    }
    
    PRIVATE_PATTERNS = [
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        re.compile(r'(?i)(password|secret|key|token|credential)\s*[:=]\s*\S+')
    ]
    
    def is_available(self) -> bool:
        return True
        
    def classify(self, query: str) -> Tuple[str, float, str, bool]:
        """Returns (complexity, confidence, sensitivity, safety_critical)"""
        query_lower = query.lower()
        
        for pattern in self.PRIVATE_PATTERNS:
            if pattern.search(query):
                return "SIMPLE", 0.95, "PRIVATE", False
                
        if any(keyword in query_lower for keyword in self.SAFETY_KEYWORDS):
            return "SIMPLE", 0.90, "PUBLIC", True
            
        if any(keyword in query_lower for keyword in self.ADVANCED_KEYWORDS):
            return "ADVANCED", 0.80, "PUBLIC", False
            
        words = query.split()
        if len(words) < 6:
            return "SIMPLE", 0.85, "PUBLIC", False
            
        return "SIMPLE", 0.70, "PUBLIC", False

class DecisionAPI:
    def decide(self, ctx: RouterContext) -> Tuple[LLMBackend, str, str]:
        if ctx.has_pii or not ctx.is_online:
            return LLMBackend.LOCAL_SLM, "PII or Offline", "R1"
            
        if ctx.top_qdrant_score > 0.75 and ctx.safety_critical:
            return LLMBackend.EXTRACTIVE, "High confidence safety critical", "R1"
            
        if ctx.top_qdrant_score > 0.75:
            return LLMBackend.LOCAL_SLM, "High qdrant score", "R2"
            
        if ctx.top_qdrant_score > 0.40:
            return LLMBackend.LOCAL_SLM, "Medium qdrant score", "R2"
            
        if ctx.frontier_approved and ctx.is_online:
            return LLMBackend.FRONTIER, "Frontier approved and online", "R3"
            
        return LLMBackend.EXTRACTIVE, "Default fallback", "R1"

class LLMRouter:
    JEV_CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self):
        self.jev = JeVClassifier()
        self.api = DecisionAPI()
        
    def route(self, ctx: RouterContext) -> RoutingDecision:
        logger.info("Routing query: %s", ctx.query)
        
        if ctx.has_pii:
            return RoutingDecision(LLMBackend.LOCAL_SLM, "R1", "Hard override: PII", "HardOverride", False, 1.0, {"forced_local": True})
            
        if not ctx.is_online:
            return RoutingDecision(LLMBackend.LOCAL_SLM, "R2", "Hard override: Offline", "HardOverride", False, 1.0, {"forced_local": True})
            
        if ctx.disputed_present:
            return RoutingDecision(LLMBackend.HITL, "R4", "Hard override: Disputed facts present", "HardOverride", True, 1.0, {})
            
        if self.jev.is_available():
            complexity, confidence, sensitivity, safety_critical = self.jev.classify(ctx.query)
            ctx.jev_available = True
            ctx.jev_complexity = complexity
            ctx.jev_confidence = confidence
            ctx.jev_sensitivity = sensitivity
            ctx.safety_critical = safety_critical
            
            if confidence >= self.JEV_CONFIDENCE_THRESHOLD:
                if sensitivity == "PRIVATE":
                    return RoutingDecision(LLMBackend.LOCAL_SLM, "R1", "JeV Private", "JeV", False, confidence, {})
                if safety_critical and ctx.top_qdrant_score > 0.75:
                    return RoutingDecision(LLMBackend.EXTRACTIVE, "R1", "JeV Safety Critical strong evidence", "JeV", False, confidence, {})
                if complexity == "ADVANCED" and ctx.frontier_approved:
                    return RoutingDecision(LLMBackend.FRONTIER, "R3", "JeV Advanced", "JeV", True, confidence, {})
                return RoutingDecision(LLMBackend.LOCAL_SLM, "R2", "JeV Default", "JeV", False, confidence, {})
                
        backend, reason, route_code = self.api.decide(ctx)
        
        requires_hitl = (backend == LLMBackend.HITL) or (route_code == "R3" and not ctx.is_online)
        if requires_hitl and not ctx.is_online:
            return RoutingDecision(LLMBackend.HITL, "R4", "Offline fallback to HITL", "Fallback", True, 0.5, {})
            
        return RoutingDecision(backend, route_code, reason, "DecisionAPI", requires_hitl, 0.7, {})
