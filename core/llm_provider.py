"""
LLM Provider Abstraction Layer

Supports multiple LLM providers with a unified interface:
- Groq (Llama 3.3 70B, Mixtral) - FREE (Default)
- Google Gemini - FREE tier available
- Ollama (Local models) - FREE, runs locally
- OpenAI (GPT-4, GPT-4o) - Paid
- OpenRouter (Access to all models)

Features:
- Automatic retry with exponential backoff
- Unified interface across providers
- Easy provider switching via .env

Usage:
    from core.llm_provider import get_llm_client, LLMConfig

    # Use default provider from .env (Groq)
    client = get_llm_client()
    response = client.chat("What is Python?")

    # Or specify provider
    client = get_llm_client(provider="gemini")
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from dotenv import load_dotenv

from core.cost_tracker import _current_role, record_usage
from core.interaction_log import record as record_interaction

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def _log_interaction(
    provider: str,
    model: str,
    messages: list[dict],
    response: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Record an interaction for fine-tuning, attributed to the current role.

    No-op if INTERACTION_LOG_PATH isn't set (interaction logging is opt-in).
    Pulls role from the cost_tracker contextvar set at agent invocation time.
    """
    role = _current_role.get() or "unknown"
    system_message = ""
    user_message = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg.get("content", "")
        elif msg.get("role") == "user":
            user_message = msg.get("content", "")
    try:
        record_interaction(
            role=role,
            provider=provider,
            model=model,
            system_message=system_message,
            user_message=user_message,
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception:
        # Logging must never break the pipeline
        pass


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: str = "groq"  # Default to free Groq
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024
    max_retries: int = 3  # Retry configuration

    # Default models per provider
    DEFAULT_MODELS: dict[str, str] = field(
        default_factory=lambda: {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.0-flash-exp",
            "ollama": "llama3.2",
            "openai": "gpt-4o",
            "openrouter": "meta-llama/llama-3.3-70b-instruct",
        }
    )

    def __post_init__(self):
        if self.model is None:
            self.model = self.DEFAULT_MODELS.get(self.provider, "llama-3.3-70b-versatile")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _resolve_params(
        self, temperature: float | None, max_tokens: int | None
    ) -> tuple[float, int]:
        """Resolve per-call overrides against config defaults.

        Explicit ``is not None`` checks: ``temperature=0.0`` is a legitimate
        request for deterministic output and must not be swallowed by
        ``temperature or default``.
        """
        return (
            temperature if temperature is not None else self.config.temperature,
            max_tokens if max_tokens is not None else self.config.max_tokens,
        )

    @abstractmethod
    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request"""
        pass

    @abstractmethod
    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion with full message history"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4, GPT-4o, etc.)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat_with_messages(messages, temperature, max_tokens)

    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature, max_tokens = self._resolve_params(temperature, max_tokens)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        content = response.choices[0].message.content.strip()
        pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
        # Record even usage-less responses so the call count stays honest
        record_usage(
            provider="openai",
            model=self.config.model,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        _log_interaction("openai", self.config.model, messages, content, pt, ct)
        return content


class GroqClient(BaseLLMClient):
    """
    Groq API client - FREE and FAST!

    Models available (with separate rate limits):
    - llama-3.3-70b-versatile (best quality) - 100k TPD
    - llama-3.1-8b-instant (fastest) - 500k TPD
    - mixtral-8x7b-32768 (good for long context) - 500k TPD
    - gemma2-9b-it (good alternative) - 500k TPD

    Auto-fallback: When primary model hits rate limit, automatically
    switches to fallback models with higher limits.

    Get free API key at: https://console.groq.com/
    """

    # Fallback order: if primary model hits rate limit, try these
    FALLBACK_MODELS = [
        "llama-3.1-8b-instant",  # 500k TPD, very fast
        "gemma2-9b-it",  # 500k TPD, good quality
        "mixtral-8x7b-32768",  # 500k TPD, long context
    ]

    # How long a rate-limited model stays benched before being retried.
    # Without this, a single transient 429 excluded the model for the entire
    # client lifetime (typically the whole session).
    RATE_LIMIT_COOLDOWN_S = 120.0

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.primary_model = config.model
        self.current_model = config.model
        self._rate_limited_until: dict[str, float] = {}  # model -> retry-after ts
        try:
            from groq import Groq

            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except ImportError as e:
            raise ImportError("Please install groq: pip install groq") from e

    def _is_rate_limited(self, model: str) -> bool:
        import time

        until = self._rate_limited_until.get(model)
        return until is not None and time.time() < until

    def _mark_rate_limited(self, model: str) -> None:
        import time

        self._rate_limited_until[model] = time.time() + self.RATE_LIMIT_COOLDOWN_S

    def _get_available_model(self) -> str:
        """Get an available model, falling back if primary is rate limited."""
        if not self._is_rate_limited(self.primary_model):
            return self.primary_model

        for fallback in self.FALLBACK_MODELS:
            if not self._is_rate_limited(fallback):
                if self.current_model != fallback:
                    print(f"  [WARN] Falling back to model: {fallback}")
                return fallback

        # All models rate limited, try primary anyway
        return self.primary_model

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat_with_messages(messages, temperature, max_tokens)

    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last_error = None
        temperature, max_tokens = self._resolve_params(temperature, max_tokens)

        # Try available models
        for _attempt in range(len(self.FALLBACK_MODELS) + 2):
            self.current_model = self._get_available_model()

            try:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = getattr(response, "usage", None)
                content = response.choices[0].message.content.strip()
                pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
                ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
                # Record even usage-less responses so the call count stays honest
                record_usage(
                    provider="groq",
                    model=self.current_model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
                _log_interaction("groq", self.current_model, messages, content, pt, ct)
                return content

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    self._mark_rate_limited(self.current_model)
                    print(f"  [WARN] Rate limited on {self.current_model}, trying fallback...")
                    continue
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        # All models exhausted
        raise last_error or Exception("All Groq models rate limited")


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client - FREE tier available!

    Free tier: 15 requests/minute, 1500 requests/day

    Models:
    - gemini-2.0-flash-exp (newest, fastest)
    - gemini-1.5-pro (best quality)
    - gemini-1.5-flash (balanced)

    Get free API key at: https://aistudio.google.com/apikey
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel(self.config.model)
        except ImportError as e:
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            ) from e

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = user_message
        if system_message:
            prompt = f"{system_message}\n\n{user_message}"

        temperature, max_tokens = self._resolve_params(temperature, max_tokens)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = self.model.generate_content(prompt, generation_config=generation_config)
        usage = getattr(response, "usage_metadata", None)
        pt = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        ct = getattr(usage, "candidates_token_count", 0) or 0 if usage else 0
        content = response.text.strip()
        record_usage(
            provider="gemini",
            model=self.config.model,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        messages = [{"role": "user", "content": prompt}]
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        _log_interaction("gemini", self.config.model, messages, content, pt, ct)
        return content

    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        return self.chat(prompt, temperature=temperature, max_tokens=max_tokens)


class OllamaClient(BaseLLMClient):
    """
    Ollama client - FREE, runs locally!

    Install: https://ollama.ai/
    Then: ollama pull llama3.2

    Models:
    - llama3.2 (8B, good balance)
    - codellama (specialized for code)
    - mistral (7B, fast)
    - deepseek-coder (great for code)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat_with_messages(messages, temperature, max_tokens)

    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import requests

        temperature, max_tokens = self._resolve_params(temperature, max_tokens)
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        prompt_tokens = data.get("prompt_eval_count") or 0
        completion_tokens = data.get("eval_count") or 0
        content = data["message"]["content"].strip()
        record_usage(
            provider="ollama",
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        _log_interaction(
            "ollama",
            self.config.model,
            messages,
            content,
            prompt_tokens,
            completion_tokens,
        )
        return content


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter API client - Access to ALL models!

    Pay-per-use with many cheap/free options.

    Get API key at: https://openrouter.ai/
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
        )

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat_with_messages(messages, temperature, max_tokens)

    def chat_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature, max_tokens = self._resolve_params(temperature, max_tokens)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        content = response.choices[0].message.content.strip()
        pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
        # Record even usage-less responses so the call count stays honest
        record_usage(
            provider="openrouter",
            model=self.config.model,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        _log_interaction("openrouter", self.config.model, messages, content, pt, ct)
        return content


# Provider registry
PROVIDERS = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
    "openrouter": OpenRouterClient,
}


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    role: str | None = None,
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.

    Args:
        provider: One of 'openai', 'groq', 'gemini', 'ollama', 'openrouter'
                  Defaults to LLM_PROVIDER env var or 'openai'
        model: Model name (uses provider default if not specified)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        role: Optional agent role for model routing. If set, the ModelRouter
              decides the provider/model unless ``provider``/``model`` is
              already given explicitly. See core.model_router.

    Returns:
        Configured LLM client

    Example:
        # Use free Groq API
        client = get_llm_client(provider="groq")
        response = client.chat("Write a hello world in Python")

        # Routed by role
        client = get_llm_client(role="documenter")  # picks a cheap model
    """
    # Apply role-based routing only when caller didn't already specify
    if role and (provider is None or model is None):
        from core.model_router import get_router

        choice = get_router().for_role(role)
        if choice is not None:
            if provider is None and choice.provider is not None:
                provider = choice.provider
            if model is None:
                model = choice.model

    # Default matches LLMConfig.provider and .env.example — previously this
    # fell back to "openai" while the router handed out Groq model names,
    # producing model-not-found errors when LLM_PROVIDER was unset.
    provider = provider or os.getenv("LLM_PROVIDER", "groq")

    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")

    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return PROVIDERS[provider](config)


# Convenience function for quick usage
def quick_chat(
    message: str,
    system: str | None = None,
    provider: str | None = None,
) -> str:
    """Quick one-off chat without managing client lifecycle"""
    client = get_llm_client(provider=provider)
    return client.chat(message, system_message=system)
