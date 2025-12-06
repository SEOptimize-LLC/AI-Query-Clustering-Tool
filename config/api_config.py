"""
API configuration and endpoint definitions.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class APIEndpoints:
    """API endpoint configurations."""
    
    # OpenAI
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_EMBEDDINGS: str = "/embeddings"
    
    # OpenRouter
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # DataForSEO
    DATAFORSEO_BASE_URL: str = "https://api.dataforseo.com/v3"
    DATAFORSEO_KEYWORDS: str = "/keywords_data/google_ads/search_volume/live"
    
    # Serper.dev
    SERPER_BASE_URL: str = "https://google.serper.dev"
    SERPER_SEARCH: str = "/search"


@dataclass
class RateLimits:
    """Rate limiting configuration per API."""
    
    # Requests per minute
    openai_rpm: int = 500
    openrouter_rpm: int = 100
    dataforseo_rpm: int = 2000
    serper_rpm: int = 100
    
    # Concurrent requests
    openai_concurrent: int = 10
    openrouter_concurrent: int = 5
    dataforseo_concurrent: int = 10
    serper_concurrent: int = 5


@dataclass
class APIConfig:
    """Complete API configuration."""
    
    endpoints: APIEndpoints
    rate_limits: RateLimits
    
    # Retry configuration
    max_retries: int = 3
    retry_delays: List[int] = None  # seconds between retries
    
    # Timeout configuration (seconds)
    connect_timeout: int = 10
    read_timeout: int = 60
    
    def __post_init__(self):
        if self.retry_delays is None:
            self.retry_delays = [1, 5, 15]
    
    @classmethod
    def default(cls) -> "APIConfig":
        """Create default API configuration."""
        return cls(
            endpoints=APIEndpoints(),
            rate_limits=RateLimits()
        )


# Location codes for DataForSEO (common ones)
LOCATION_CODES: Dict[str, int] = {
    "United States": 2840,
    "United Kingdom": 2826,
    "Canada": 2124,
    "Australia": 2036,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "Brazil": 2076,
    "Mexico": 2484,
    "India": 2356,
    "Japan": 2392,
}

# Language codes for DataForSEO
LANGUAGE_CODES: Dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Japanese": "ja",
    "Chinese": "zh",
    "Hindi": "hi",
}

# LLM Models available via OpenRouter
LLM_MODELS: Dict[str, Dict] = {
    "anthropic/claude-sonnet-4": {
        "name": "Claude Sonnet 4",
        "context_length": 200000,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "best_for": "Complex reasoning, nuanced labels"
    },
    "google/gemini-2.5-flash-preview-05-20": {
        "name": "Gemini 2.5 Flash",
        "context_length": 1000000,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "best_for": "Fast processing, cost-effective"
    },
    "openai/gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "context_length": 128000,
        "cost_per_1k_input": 0.0004,
        "cost_per_1k_output": 0.0016,
        "best_for": "Balanced cost/quality"
    }
}