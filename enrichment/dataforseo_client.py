"""
DataForSEO API client for keyword metrics.
"""
import streamlit as st
import asyncio
import aiohttp
import base64
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from core.exceptions import APIError, RateLimitError


@dataclass
class KeywordMetrics:
    """Keyword metrics from DataForSEO."""
    
    keyword: str
    search_volume: int = 0
    keyword_difficulty: float = 0.0
    cpc: float = 0.0
    competition: float = 0.0


class DataForSEOClient:
    """
    Client for DataForSEO Keywords Data API.
    Fetches search volume and keyword difficulty.
    """
    
    BASE_URL = "https://api.dataforseo.com/v3"
    BATCH_SIZE = 1000  # Max keywords per request
    
    def __init__(
        self,
        login: str = None,
        password: str = None
    ):
        """
        Initialize DataForSEO client.
        
        Args:
            login: DataForSEO login email
            password: DataForSEO API password
        """
        self.login = login or st.secrets.get("DATAFORSEO_LOGIN", "")
        self.password = password or st.secrets.get("DATAFORSEO_PASSWORD", "")
        
        # Create auth header
        credentials = f"{self.login}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.auth_header = f"Basic {encoded}"
    
    @property
    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.login and self.password)
    
    async def get_keyword_metrics(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        progress_callback: Callable = None
    ) -> Dict[str, KeywordMetrics]:
        """
        Fetch search volume and difficulty for keywords.
        
        Args:
            keywords: List of keywords
            location_code: DataForSEO location code (default: US)
            language_code: Language code
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict mapping keyword -> KeywordMetrics
        """
        if not self.is_configured:
            raise APIError(
                "DataForSEO credentials not configured",
                service="DataForSEO"
            )
        
        results = {}
        total_batches = (len(keywords) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        async with aiohttp.ClientSession() as session:
            for i, batch_start in enumerate(
                range(0, len(keywords), self.BATCH_SIZE)
            ):
                batch = keywords[batch_start:batch_start + self.BATCH_SIZE]
                
                try:
                    batch_results = await self._fetch_batch(
                        session,
                        batch,
                        location_code,
                        language_code
                    )
                    results.update(batch_results)
                except Exception as e:
                    # Log error but continue with other batches
                    st.warning(f"Batch {i+1} failed: {e}")
                
                if progress_callback:
                    progress_callback(
                        min(batch_start + self.BATCH_SIZE, len(keywords)),
                        len(keywords)
                    )
                
                # Small delay between batches to avoid rate limits
                if i < total_batches - 1:
                    await asyncio.sleep(0.5)
        
        return results
    
    async def _fetch_batch(
        self,
        session: aiohttp.ClientSession,
        keywords: List[str],
        location_code: int,
        language_code: str
    ) -> Dict[str, KeywordMetrics]:
        """
        Fetch metrics for a single batch of keywords.
        
        Args:
            session: aiohttp session
            keywords: Batch of keywords
            location_code: Location code
            language_code: Language code
        
        Returns:
            Dict mapping keyword -> KeywordMetrics
        """
        url = f"{self.BASE_URL}/keywords_data/google_ads/search_volume/live"
        
        payload = [{
            "keywords": keywords,
            "location_code": location_code,
            "language_code": language_code
        }]
        
        headers = {
            "Authorization": self.auth_header,
            "Content-Type": "application/json"
        }
        
        async with session.post(
            url,
            json=payload,
            headers=headers
        ) as response:
            if response.status == 429:
                raise RateLimitError(
                    service="DataForSEO",
                    retry_after=60
                )
            
            if response.status != 200:
                text = await response.text()
                raise APIError(
                    f"DataForSEO API error: {text}",
                    service="DataForSEO",
                    status_code=response.status,
                    response=text
                )
            
            data = await response.json()
            return self._parse_response(data)
    
    def _parse_response(
        self,
        data: dict
    ) -> Dict[str, KeywordMetrics]:
        """
        Parse DataForSEO API response.
        
        Args:
            data: API response JSON
        
        Returns:
            Dict mapping keyword -> KeywordMetrics
        """
        results = {}
        
        tasks = data.get("tasks", []) or []
        for task in tasks:
            task_result = task.get("result", []) or []
            for item in task_result or []:
                keyword = item.get("keyword", "")
                if not keyword:
                    continue
                
                results[keyword] = KeywordMetrics(
                    keyword=keyword,
                    search_volume=item.get("search_volume") or 0,
                    keyword_difficulty=item.get("keyword_difficulty") or 0.0,
                    cpc=item.get("cpc") or 0.0,
                    competition=item.get("competition") or 0.0
                )
        
        return results
    
    def get_metrics_sync(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        progress_callback: Callable = None
    ) -> Dict[str, KeywordMetrics]:
        """
        Synchronous wrapper for get_keyword_metrics.
        Use this in Streamlit (which has its own event loop).
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.get_keyword_metrics(
                    keywords,
                    location_code,
                    language_code,
                    progress_callback
                )
            )
        finally:
            loop.close()


def get_dataforseo_client() -> Optional[DataForSEOClient]:
    """
    Get DataForSEO client if configured.
    
    Returns:
        DataForSEOClient or None if not configured
    """
    client = DataForSEOClient()
    return client if client.is_configured else None