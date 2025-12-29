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

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str = "groq"  # Default to free Groq
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024
    max_retries: int = 3  # Retry configuration

    # Default models per provider
    DEFAULT_MODELS: dict[str, str] = field(default_factory=lambda: {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.0-flash-exp",
        "ollama": "llama3.2",
        "openai": "gpt-4o",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
    })

    def __post_init__(self):
        if self.model is None:
            self.model = self.DEFAULT_MODELS.get(self.provider, "llama-3.3-70b-versatile")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config

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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


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
        "llama-3.1-8b-instant",     # 500k TPD, very fast
        "gemma2-9b-it",             # 500k TPD, good quality
        "mixtral-8x7b-32768",       # 500k TPD, long context
    ]

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.primary_model = config.model
        self.current_model = config.model
        self._rate_limited_models = set()  # Track which models are rate limited
        try:
            from groq import Groq
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except ImportError:
            raise ImportError("Please install groq: pip install groq")

    def _get_available_model(self) -> str:
        """Get an available model, falling back if primary is rate limited."""
        if self.primary_model not in self._rate_limited_models:
            return self.primary_model

        for fallback in self.FALLBACK_MODELS:
            if fallback not in self._rate_limited_models:
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

        # Try available models
        for attempt in range(len(self.FALLBACK_MODELS) + 2):
            self.current_model = self._get_available_model()

            try:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    self._rate_limited_models.add(self.current_model)
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
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

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

        generation_config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_tokens,
        }

        response = self.model.generate_content(prompt, generation_config=generation_config)
        return response.text.strip()

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

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


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
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


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
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: One of 'openai', 'groq', 'gemini', 'ollama', 'openrouter'
                  Defaults to LLM_PROVIDER env var or 'openai'
        model: Model name (uses provider default if not specified)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    
    Returns:
        Configured LLM client
    
    Example:
        # Use free Groq API
        client = get_llm_client(provider="groq")
        response = client.chat("Write a hello world in Python")
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

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
