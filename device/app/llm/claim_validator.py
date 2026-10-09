import logging
import re
from typing import List, Dict, Tuple, Any
from device.app.memory.models import Hit, Memory

logger = logging.getLogger(__name__)

class ClaimValidator:
    CLAIM_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(Nm|PSI|bar|°C|°F|RPM|mm|kg|kW|Amp|V|ft-lb|thou)\b', re.IGNORECASE)
    IDENTIFIER_PATTERN = re.compile(r'\b([A-Z]{1,4}-\d{2,4}|[A-Z]{3,4}-xxxx)\b')

    def _extract_claims(self, text: str) -> List[Tuple[float, str]]:
        claims = []
        for match in self.CLAIM_PATTERN.finditer(text):
            try:
                val = float(match.group(1))
                unit = match.group(2).lower()
                claims.append((val, unit))
            except ValueError:
                continue
        return claims

    def _extract_identifiers(self, text: str) -> List[str]:
        return [match.group(1) for match in self.IDENTIFIER_PATTERN.finditer(text)]

    def validate(self, answer_text: str, hits: List[Hit], memories: Dict[str, Memory]) -> List[Dict[str, Any]]:
        warnings = []
        
        claims = self._extract_claims(answer_text)
        identifiers = self._extract_identifiers(answer_text)
        
        source_texts = []
        source_claims = []
        max_authority = 0
        
        for hit in hits:
            # Safely handle Hit payload which might be dict or object based on actual models
            payload = getattr(hit, 'payload', {}) or {}
            
            if "text" in payload:
                source_texts.append(payload["text"])
                source_claims.extend(self._extract_claims(payload["text"]))
                
            if "claims" in payload and isinstance(payload["claims"], list):
                for c in payload["claims"]:
                    if isinstance(c, dict) and 'value' in c and 'unit' in c:
                        try:
                            source_claims.append((float(c['value']), str(c['unit']).lower()))
                        except ValueError:
                            pass
                            
            hit_id = getattr(hit, 'id', None) or getattr(hit, 'memory_id', None)
            if hit_id and hit_id in memories:
                mem = memories[hit_id]
                auth = getattr(mem, 'authority', 0)
                if auth > max_authority:
                    max_authority = auth
                    
        full_source_text = " ".join(source_texts)
        source_identifiers = self._extract_identifiers(full_source_text)
        
        severity = "HIGH" if max_authority >= 2 else "MEDIUM"
        
        for val, unit in claims:
            matched = False
            for s_val, s_unit in source_claims:
                if unit == s_unit:
                    if s_val == 0:
                        diff = abs(val - s_val)
                        if diff <= 0.05:
                            matched = True
                            break
                    elif abs(val - s_val) / abs(s_val) <= 0.05:
                        matched = True
                        break
                        
            if not matched:
                warnings.append({
                    "type": "value_mismatch" if any(u == unit for v, u in source_claims) else "unsupported_claim",
                    "detail": f"Claim {val} {unit} not supported by sources.",
                    "answer_says": f"{val} {unit}",
                    "source_says": str([f"{v} {u}" for v, u in source_claims if u == unit]) if any(u == unit for v, u in source_claims) else None,
                    "mem_id": getattr(hits[0], 'id', 'unknown') if hits else "unknown",
                    "severity": severity
                })
                
        for identifier in identifiers:
            if identifier not in source_identifiers and identifier not in full_source_text:
                warnings.append({
                    "type": "unsupported_identifier",
                    "detail": f"Identifier {identifier} not found in sources.",
                    "answer_says": identifier,
                    "source_says": None,
                    "mem_id": getattr(hits[0], 'id', 'unknown') if hits else "unknown",
                    "severity": severity
                })
                
        return warnings

    def has_hard_failures(self, warnings: List[Dict[str, Any]]) -> bool:
        for w in warnings:
            if w["type"] in ("value_mismatch", "unsupported_identifier"):
                return True
        return False
