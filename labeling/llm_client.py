"""
LLM client for OpenRouter API.
Supports multiple models with fallback.
"""
import time
import httpx
from typing import List, Dict
from dataclasses import dataclass

from config.settings import Settings
from config.api_config import LLM_MODELS
from core.exceptions import APIError, RateLimitError


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float


class OpenRouterClient:
    """
    Client for OpenRouter API.
    
    Features:
    - Multiple model support
    - Automatic fallback on errors
    - Rate limiting
    - Token tracking
    """
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: str = None,
        default_model: str = None,
        timeout: float = 60.0
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (from settings if not provided)
            default_model: Default model to use
            timeout: Request timeout in seconds
        """
        settings = Settings()
        self.api_key = api_key or settings.openrouter_api_key
        self.default_model = default_model or LLM_MODELS["primary"]
        self.timeout = timeout
        
        # Request tracking
        self.total_tokens = 0
        self.request_count = 0
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        system_prompt: str = None
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            model: Model to use (default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        response = await self.generate_with_metadata(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt
        )
        return response.content
    
    async def generate_with_metadata(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        system_prompt: str = None
    ) -> LLMResponse:
        """
        Generate text with full response metadata.
        
        Args:
            prompt: User prompt
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Temperature
            system_prompt: System prompt
        
        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.default_model
        
        # Try primary model, then fallbacks
        models_to_try = [model]
        if model == LLM_MODELS["primary"]:
            models_to_try.extend([
                LLM_MODELS["fast"],
                LLM_MODELS["fallback"]
            ])
        
        last_error = None
        
        for try_model in models_to_try:
            try:
                return await self._make_request(
                    prompt=prompt,
                    model=try_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
            except RateLimitError as e:
                last_error = e
                # Wait and retry with same model
                await self._wait_for_rate_limit(e)
            except APIError as e:
                last_error = e
                # Try next model
                continue
        
        raise last_error or APIError("All models failed")
    
    async def _make_request(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str = None
    ) -> LLMResponse:
        """Make API request to OpenRouter."""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://keyword-clustering.streamlit.app",
            "X-Title": "AI Keyword Clustering"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.BASE_URL,
                json=payload,
                headers=headers
            )
        
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise RateLimitError(
                f"Rate limited. Retry after {retry_after}s",
                retry_after=int(retry_after)
            )
        
        if response.status_code != 200:
            raise APIError(
                f"OpenRouter API error: {response.status_code}",
                status_code=response.status_code
            )
        
        data = response.json()
        
        # Extract content
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        
        # Update tracking
        self.total_tokens += tokens
        self.request_count += 1
        
        return LLMResponse(
            content=content,
            model=model,
            tokens_used=tokens,
            latency_ms=latency
        )
    
    async def _wait_for_rate_limit(self, error: RateLimitError):
        """Wait for rate limit to reset."""
        import asyncio
        wait_time = min(error.retry_after or 60, 120)
        await asyncio.sleep(wait_time)
    
    async def generate_batch(
        self,
        prompts: List[str],
        model: str = None,
        max_tokens: int = 1000
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            model: Model to use
            max_tokens: Max tokens per response
        
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = await self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens
            )
            responses.append(response)
        return responses
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_tokens_per_request": (
                self.total_tokens / self.request_count
                if self.request_count > 0
                else 0
            )
        }


# Synchronous wrapper for non-async contexts
class SyncLLMClient:
    """Synchronous wrapper for OpenRouterClient."""
    
    def __init__(self, **kwargs):
        """Initialize with same args as OpenRouterClient."""
        self.async_client = OpenRouterClient(**kwargs)
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Synchronous generate."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.async_client.generate(prompt, **kwargs)
        )
    
    def get_stats(self) -> Dict:
        """Get usage stats."""
        return self.async_client.get_stats()