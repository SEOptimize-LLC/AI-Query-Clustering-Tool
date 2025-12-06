"""
Application settings and configuration.
All API credentials are loaded from Streamlit Cloud Secrets.
"""
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class ClusteringSettings:
    """Settings for the clustering algorithm."""
    
    # Coarse clustering (MiniBatch K-Means)
    coarse_cluster_ratio: float = 0.5  # sqrt(n/ratio) clusters
    kmeans_batch_size: int = 1000
    kmeans_n_init: int = 3
    
    # Fine clustering (HDBSCAN)
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    
    # SERP validation
    serp_validation_count: int = 10  # Top N keywords by volume per cluster
    serp_overlap_threshold: float = 0.4  # Minimum SERP overlap for coherence
    
    # Outlier reassignment
    outlier_similarity_threshold: float = 0.7  # Cosine similarity threshold


@dataclass
class EmbeddingSettings:
    """Settings for embedding generation."""
    
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100  # OpenAI API batch size
    chunk_size: int = 5000  # Keywords per processing chunk


@dataclass
class LabelingSettings:
    """Settings for LLM-based label generation."""
    
    # Available models via OpenRouter
    available_models: dict = field(default_factory=lambda: {
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "gemini-2.5-flash": "google/gemini-2.5-flash-preview-05-20",
        "gpt-4.1-mini": "openai/gpt-4.1-mini"
    })
    default_model: str = "anthropic/claude-sonnet-4"
    temperature: float = 0.3  # Lower for consistent labeling
    max_clusters_per_batch: int = 50  # Clusters per labeling request


@dataclass
class CacheSettings:
    """Settings for caching strategy."""
    
    keyword_metrics_ttl_days: int = 7
    serp_data_ttl_hours: int = 24


@dataclass
class Settings:
    """
    Main application settings.
    Loads API credentials from Streamlit Cloud Secrets (st.secrets).
    """
    
    # App info
    app_name: str = "AI Keyword Clustering"
    app_version: str = "1.0.0"
    
    # Sub-settings
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    labeling: LabelingSettings = field(default_factory=LabelingSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    
    # Location settings (for DataForSEO/Serper)
    default_location_code: int = 2840  # US
    default_language_code: str = "en"
    default_location_name: str = "United States"
    
    def _get_secret(self, flat_key: str, nested_section: str, nested_key: str) -> str:
        """
        Get secret supporting both flat and nested formats.
        
        Flat: SUPABASE_URL = "..."
        Nested: [supabase]
                url = "..."
        """
        # Try flat format first
        if flat_key in st.secrets:
            return st.secrets[flat_key]
        
        # Try nested format
        try:
            if nested_section in st.secrets:
                section = st.secrets[nested_section]
                if nested_key in section:
                    return section[nested_key]
        except Exception:
            pass
        
        return ""
    
    @property
    def supabase_url(self) -> str:
        """Get Supabase URL from Streamlit secrets."""
        return self._get_secret("SUPABASE_URL", "supabase", "url")
    
    @property
    def supabase_key(self) -> str:
        """Get Supabase anon key from Streamlit secrets."""
        return self._get_secret("SUPABASE_KEY", "supabase", "key")
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from Streamlit secrets."""
        return self._get_secret("OPENAI_API_KEY", "openai", "api_key")
    
    @property
    def openrouter_api_key(self) -> str:
        """Get OpenRouter API key from Streamlit secrets."""
        return self._get_secret("OPENROUTER_API_KEY", "openrouter", "api_key")
    
    @property
    def dataforseo_login(self) -> str:
        """Get DataForSEO login from Streamlit secrets."""
        return self._get_secret("DATAFORSEO_LOGIN", "dataforseo", "login")
    
    @property
    def dataforseo_password(self) -> str:
        """Get DataForSEO password from Streamlit secrets."""
        return self._get_secret("DATAFORSEO_PASSWORD", "dataforseo", "password")
    
    @property
    def serper_api_key(self) -> str:
        """Get Serper API key from Streamlit secrets."""
        return self._get_secret("SERPER_API_KEY", "serper", "api_key")
    
    def validate_secrets(self) -> dict:
        """
        Validate that all required secrets are configured.
        Returns dict with service names and their status.
        """
        return {
            "Supabase": bool(self.supabase_url and self.supabase_key),
            "OpenAI": bool(self.openai_api_key),
            "OpenRouter": bool(self.openrouter_api_key),
            "DataForSEO": bool(self.dataforseo_login and self.dataforseo_password),
            "Serper": bool(self.serper_api_key)
        }
    
    def get_missing_secrets(self) -> list:
        """Return list of missing required secrets."""
        status = self.validate_secrets()
        return [service for service, configured in status.items() if not configured]


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid recreating settings on each call.
    """
    return Settings()