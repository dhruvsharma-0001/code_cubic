"""device/app/llm/frontier.py

Frontier API caller (Gemini / OpenAI-compatible) with strict structured JSON output.
ONLY called after HITL Gate A approval. Cloud answers are NEVER authoritative (I9).
"""
from __future__ import annotations
import json
import logging
import os
import httpx
from typing import Any, Tuple, List

from device.app.memory.models import Hit, Memory

logger = logging.getLogger(__name__)

FRONTIER_URL = os.getenv("FRONTIER_API_URL", "https://api.openai.com/v1")
FRONTIER_KEY = os.getenv("FRONTIER_API_KEY", "")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "gpt-4o-mini")

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "citations", "confidence"],
    "additionalProperties": False
}

class FrontierCaller:
    def call(self, query: str, hits: list[Hit]) -> tuple[str, list[str], float, str]:
        if not FRONTIER_KEY:
            raise RuntimeError("FRONTIER_API_KEY not set")
            
        context = ""
        for i, hit in enumerate(hits[:5]):
            context += f"[{i+1}] ID: {hit.mem_id} | {hit.memory.content}\n"
            
        prompt = f"Context:\n{context}\n\nQuery: {query}\nProvide a helpful answer based strictly on the context if possible, or general knowledge if safe. Use JSON format matching the schema."
        
        payload = {
            "model": FRONTIER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful cloud AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {FRONTIER_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{FRONTIER_URL}/chat/completions", json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            return (
                result.get("answer", content),
                result.get("citations", []),
                float(result.get("confidence", 0.9)),
                FRONTIER_MODEL
            )
        except Exception as e:
            logger.error(f"Frontier call failed: {e}")
            raise RuntimeError(f"Frontier API error: {e}") from e
