"""device/app/llm/local_slm.py

Dual-engine local SLM: MLX for Apple Silicon Mac, GGUF for Windows/Linux/Intel.
Falls back to extractive AnswerComposer if no model is available.
All inference is 100% on-device with zero cloud egress.
"""
from __future__ import annotations
import json
import logging
import os
import platform
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Tuple, List

from device.app.memory.models import Hit, Memory

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an industrial grounding assistant.
Strict Instructions:
1. ONLY answer from the provided context.
2. Cite your sources using the provided citation IDs.
3. If the answer is not in the context, set "abstain" to true.
4. NEVER invent specifications or facts.
5. Return your response EXACTLY matching the expected JSON schema.
"""

SLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "citations", "abstain", "confidence"]
}

def _build_rag_prompt(query: str, hits: list[Hit]) -> str:
    prompt = f"Query: {query}\n\nContext:\n"
    for i, hit in enumerate(hits[:5]):
        prompt += f"[{i+1}] ID: {hit.mem_id} | Authority: {hit.memory.authority} | Kind: {hit.memory.kind}\n"
        prompt += f"{hit.memory.content}\n\n"
    prompt += "\nPlease answer the query based ONLY on the context above. Output strictly as JSON."
    return prompt

def select_backend() -> str:
    env_backend = os.getenv("LLM_BACKEND", "auto").lower()
    if env_backend in ["mlx", "gguf", "ollama"]:
        return env_backend
    
    if env_backend == "auto":
        # Check MLX for macOS arm64
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx_lm
                return "mlx"
            except ImportError:
                pass
        
        # Check GGUF
        try:
            import llama_cpp
            return "gguf"
        except ImportError:
            pass
            
        # Check Ollama (only if configured or explicitly requested to avoid airgap socket violation)
        if os.getenv("OLLAMA_URL"):
            try:
                import httpx
                ollama_url = os.getenv("OLLAMA_URL")
                res = httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
                if res.status_code == 200:
                    return "ollama"
            except Exception:
                pass
            
    return "none"

class MLXCaller:
    def __init__(self):
        self._available = False
        self.model = None
        self.tokenizer = None
        try:
            import mlx_lm
            self.mlx_lm = mlx_lm
            model_path = os.getenv("MLX_MODEL_PATH", "mlx-community/Qwen2.5-3B-Instruct-4bit")
            self.model, self.tokenizer = self.mlx_lm.load(model_path)
            self._available = True
        except ImportError:
            logger.warning("mlx_lm not installed")
        except Exception as e:
            logger.warning(f"MLX load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str, system: str, max_tokens: int = 256) -> str:
        if not self._available:
            raise RuntimeError("MLX backend unavailable")
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system}\n\n{prompt}"
            
        response = self.mlx_lm.generate(self.model, self.tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
        return response

class GGUFCaller:
    def __init__(self):
        self._available = False
        self.llm = None
        try:
            from llama_cpp import Llama
            model_path = os.getenv("GGUF_MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
            if os.path.exists(model_path):
                n_threads = int(os.getenv("LLAMA_N_THREADS", max(1, os.cpu_count() or 1)))
                n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                self._available = True
            else:
                logger.warning(f"GGUF model not found at {model_path}")
        except ImportError:
            logger.warning("llama_cpp not installed")
        except Exception as e:
            logger.warning(f"GGUF load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str, system: str, max_tokens: int = 256) -> str:
        if not self._available:
            raise RuntimeError("GGUF backend unavailable")
            
        text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        output = self.llm(
            text,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
            echo=False
        )
        return output["choices"][0]["text"].strip()

class OllamaCaller:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    def is_available(self) -> bool:
        try:
            import httpx
            res = httpx.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            return res.status_code == 200
        except Exception as e:
            import logging; logging.getLogger(__name__).warning("Handled exception: %s", e)
            return False

    def generate(self, prompt: str, system: str, max_tokens: int = 256) -> str:
        import httpx
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            },
            "format": "json"
        }
        try:
            res = httpx.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30.0)
            res.raise_for_status()
            return res.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

class LocalSLMService:
    def __init__(self):
        self.backend_name = select_backend()
        self._caller = None
        
        if self.backend_name == "mlx":
            self._caller = MLXCaller()
        elif self.backend_name == "gguf":
            self._caller = GGUFCaller()
        elif self.backend_name == "ollama":
            self._caller = OllamaCaller()

    def is_available(self) -> bool:
        return self._caller is not None and self._caller.is_available()

    def ping(self) -> bool:
        try:
            import httpx
            endpoint = os.getenv("LLM_ENDPOINT") or (
                getattr(self._caller, "ollama_url", None) if self.backend_name == "ollama" else "http://localhost:11434"
            )
            res = httpx.get(f"{endpoint}/api/tags", timeout=1.0)
            return res.status_code == 200
        except Exception:
            return False

    def call(self, query: str, hits: list[Hit]) -> tuple[str, list[str], float, str]:
        if not self.is_available():
            raise RuntimeError("local_slm_unavailable")
            
        prompt = _build_rag_prompt(query, hits)
        
        try:
            response_text = self._caller.generate(prompt, RAG_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"Engine generation failed: {e}")
            raise RuntimeError("local_slm_unavailable") from e
            
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
                
            data = json.loads(json_str)
            return (
                data.get("answer", response_text),
                data.get("citations", []),
                float(data.get("confidence", 0.8)),
                self.backend_name
            )
        except json.JSONDecodeError:
            logger.warning("SLM output was not valid JSON. Falling back to raw text.")
            return (response_text, [], 0.5, self.backend_name)

    def get_engine_info(self) -> dict:
        return {
            "backend": self.backend_name,
            "available": self.is_available(),
            "platform": platform.platform(),
            "model": os.getenv("MLX_MODEL_PATH") if self.backend_name == "mlx" else (os.getenv("GGUF_MODEL_PATH") if self.backend_name == "gguf" else os.getenv("OLLAMA_MODEL"))
        }
