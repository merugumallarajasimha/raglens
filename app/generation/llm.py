"""LLM abstraction layer — provides a unified interface for text generation.

Supports multiple providers (Ollama local, future API-based) with a
consistent interface. Designed for CPU-compatible, low-resource environments.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import Settings
from app.core.exceptions import LLMError

logger = logging.getLogger("raglens.generation.llm")


@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[dict] = None


@dataclass
class LLMMetrics:
    """Metrics for LLM generation performance."""

    latency_ms: float
    tokens_used: Optional[int] = None
    model: str = ""
    success: bool = True
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> LLMResponse:
        """Generate text from a prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is reachable."""


class OllamaProvider(LLMProvider):
    """LLM provider using a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> LLMResponse:
        import httpx

        start = time.time()

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                text = data.get("response", "")
                eval_count = data.get("eval_count")

                latency = (time.time() - start) * 1000
                logger.info(
                    "LLM generation complete",
                    extra={"extra_data": {
                        "model": self._model,
                        "latency_ms": round(latency, 2),
                        "tokens": eval_count,
                    }},
                )

                return LLMResponse(
                    text=text,
                    model=self._model,
                    tokens_used=eval_count,
                    latency_ms=latency,
                    raw_response=data,
                )

        except httpx.ConnectError:
            raise LLMError(f"Cannot connect to Ollama at {self._base_url}")
        except httpx.TimeoutException:
            raise LLMError(f"Ollama request timed out after {timeout}s")
        except Exception as e:
            raise LLMError(f"LLM generation failed: {e}") from e

    def is_available(self) -> bool:
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for testing.

    Generates placeholder responses that simulate LLM behavior
    without requiring a running model.
    """

    def __init__(self, model: str = "mock-llm") -> None:
        self._model = model
        self._responses: dict[str, str] = {}

    def set_response(self, key: str, response: str) -> None:
        """Set a predefined response for a query key."""
        self._responses[key.lower()] = response

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> LLMResponse:
        start = time.time()

        # Check for predefined response
        prompt_lower = prompt.lower()
        for key, response in self._responses.items():
            if key in prompt_lower:
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    text=response,
                    model=self._model,
                    tokens_used=len(response.split()),
                    latency_ms=latency,
                )

        # Default mock response
        text = "Based on the provided evidence, the key findings are summarized below."
        latency = (time.time() - start) * 1000
        return LLMResponse(
            text=text,
            model=self._model,
            tokens_used=10,
            latency_ms=latency,
        )

    def is_available(self) -> bool:
        return True


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Create an LLM provider based on settings."""
    if settings is None:
        settings = Settings()

    if settings.llm_provider == "ollama":
        provider = OllamaProvider(
            base_url=settings.ollama_url,
            model=settings.llm_model,
        )
        if not provider.is_available():
            logger.warning(
                f"Ollama not available at {settings.ollama_url}, "
                "falling back to mock provider"
            )
            return MockLLMProvider(model=settings.llm_model)
        return provider
    elif settings.llm_provider == "mock":
        return MockLLMProvider(model=settings.llm_model)
    else:
        return MockLLMProvider(model=settings.llm_model)
