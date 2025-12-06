"""
Cache manager for API response caching.
Uses Supabase for persistent caching across sessions.
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from storage.supabase_client import SupabaseClient


class CacheManager:
    """
    Manages caching for API responses.
    Uses Supabase tables for persistent caching.
    """
    
    # Cache TTL settings
    CACHE_TTL = {
        "keyword_metrics": 7,   # days
        "serp_data": 24,        # hours
    }
    
    def __init__(self, supabase: SupabaseClient):
        """
        Initialize cache manager.
        
        Args:
            supabase: Supabase client instance
        """
        self.db = supabase
    
    # =========================================================================
    # Keyword Metrics Caching
    # =========================================================================
    
    def get_keyword_metrics(
        self,
        keywords: List[str]
    ) -> Dict[str, dict]:
        """
        Get cached keyword metrics.
        
        Args:
            keywords: List of keywords to look up
        
        Returns:
            Dict mapping keyword -> {search_volume, keyword_difficulty, cpc}
        """
        if not keywords:
            return {}
        
        return self.db.get_cached_metrics(
            keywords,
            max_age_days=self.CACHE_TTL["keyword_metrics"]
        )
    
    def save_keyword_metrics(self, metrics: Dict[str, dict]):
        """
        Cache keyword metrics.
        
        Args:
            metrics: Dict mapping keyword -> metrics dict
        """
        if metrics:
            self.db.cache_metrics(metrics)
    
    def get_uncached_keywords(
        self,
        keywords: List[str]
    ) -> List[str]:
        """
        Get list of keywords that need metrics fetching.
        
        Args:
            keywords: Full list of keywords
        
        Returns:
            List of keywords not in cache or with expired cache
        """
        cached = self.get_keyword_metrics(keywords)
        return [kw for kw in keywords if kw not in cached]
    
    # =========================================================================
    # SERP Data Caching
    # =========================================================================
    
    def get_serp_data(self, keyword: str) -> Optional[dict]:
        """
        Get cached SERP data for a keyword.
        
        Args:
            keyword: Keyword to look up
        
        Returns:
            SERP data dict or None if not cached
        """
        return self.db.get_cached_serp(
            keyword,
            max_age_hours=self.CACHE_TTL["serp_data"]
        )
    
    def save_serp_data(self, keyword: str, serp_data: dict):
        """
        Cache SERP data.
        
        Args:
            keyword: Keyword
            serp_data: SERP data to cache
        """
        self.db.cache_serp(keyword, serp_data)
    
    def get_serp_data_batch(
        self,
        keywords: List[str]
    ) -> Dict[str, dict]:
        """
        Get cached SERP data for multiple keywords.
        
        Args:
            keywords: List of keywords
        
        Returns:
            Dict mapping keyword -> SERP data
        """
        return self.db.get_cached_serps_batch(
            keywords,
            max_age_hours=self.CACHE_TTL["serp_data"]
        )
    
    def get_uncached_serp_keywords(
        self,
        keywords: List[str]
    ) -> List[str]:
        """
        Get list of keywords that need SERP fetching.
        
        Args:
            keywords: Full list of keywords
        
        Returns:
            List of keywords not in SERP cache
        """
        cached = self.get_serp_data_batch(keywords)
        return [kw for kw in keywords if kw not in cached]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_cache_stats(
        self,
        keywords: List[str]
    ) -> dict:
        """
        Get cache statistics for a set of keywords.
        
        Args:
            keywords: List of keywords to check
        
        Returns:
            Dict with cache hit/miss statistics
        """
        metrics_cached = self.get_keyword_metrics(keywords)
        serp_cached = self.get_serp_data_batch(keywords)
        
        total = len(keywords)
        
        return {
            "total_keywords": total,
            "metrics_cached": len(metrics_cached),
            "metrics_uncached": total - len(metrics_cached),
            "metrics_hit_rate": len(metrics_cached) / total if total else 0,
            "serp_cached": len(serp_cached),
            "serp_uncached": total - len(serp_cached),
            "serp_hit_rate": len(serp_cached) / total if total else 0
        }
    
    def estimate_api_savings(
        self,
        keywords: List[str]
    ) -> dict:
        """
        Estimate API cost savings from caching.
        
        Args:
            keywords: List of keywords
        
        Returns:
            Dict with estimated savings
        """
        stats = self.get_cache_stats(keywords)
        
        # Approximate costs
        dataforseo_cost_per_1k = 0.05  # $
        serper_cost_per_request = 0.001  # $
        
        metrics_savings = (
            stats["metrics_cached"] / 1000
        ) * dataforseo_cost_per_1k
        serp_savings = stats["serp_cached"] * serper_cost_per_request
        
        return {
            "metrics_api_calls_saved": stats["metrics_cached"],
            "serp_api_calls_saved": stats["serp_cached"],
            "estimated_savings_usd": round(
                metrics_savings + serp_savings, 2
            )
        }