"""
DataForSEO API client for keyword metrics.
Uses synchronous requests to avoid Streamlit event loop conflicts.
"""
import streamlit as st
import requests
import base64
import time
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
    Uses synchronous requests for Streamlit compatibility.
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
    
    def get_keyword_metrics(
        self,
        keywords: List[str],
        location_code: int = 2840,
        language_code: str = "en",
        progress_callback: Callable = None
    ) -> Dict[str, KeywordMetrics]:
        """
        Fetch search volume and difficulty for keywords (synchronous).
        
        Args:
            keywords: List of keywords
            location_code: DataForSEO location code (default: US)
            language_code: Language code
            progress_callback: Optional callback(current, total)
        
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
        
        for i, batch_start in enumerate(
            range(0, len(keywords), self.BATCH_SIZE)
        ):
            batch = keywords[batch_start:batch_start + self.BATCH_SIZE]
            
            try:
                batch_results = self._fetch_batch(
                    batch,
                    location_code,
                    language_code,
                    is_first_batch=(i == 0)
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
                time.sleep(0.5)
        
        return results
    
    def _fetch_batch(
        self,
        keywords: List[str],
        location_code: int,
        language_code: str,
        is_first_batch: bool = False
    ) -> Dict[str, KeywordMetrics]:
        """
        Fetch metrics for a single batch of keywords (synchronous).
        
        Args:
            keywords: Batch of keywords
            location_code: Location code
            language_code: Language code
            is_first_batch: Whether to show debug info
        
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
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 429:
            raise RateLimitError(
                service="DataForSEO",
                retry_after=60
            )
        
        if response.status_code != 200:
            raise APIError(
                f"DataForSEO API error: {response.text}",
                service="DataForSEO",
                status_code=response.status_code,
                response=response.text
            )
        
        data = response.json()
        
        # Debug: Show full response for first batch
        if is_first_batch:
            st.caption(f"🔧 HTTP Status: {response.status_code}")
            st.caption(f"🔧 API status_code: {data.get('status_code')}")
            st.caption(f"🔧 API status_message: {data.get('status_message')}")
            
            tasks = data.get('tasks', []) or []
            if tasks:
                task = tasks[0]
                st.caption(f"🔧 Task status: {task.get('status_code')} - {task.get('status_message')}")
                result = task.get('result', []) or []
                st.caption(f"🔧 Results count: {len(result)}")
                
                # Show first result item structure
                if result:
                    first = result[0]
                    st.caption(f"🔧 First result keys: {list(first.keys())}")
                    st.caption(f"🔧 First kw: {first.get('keyword')} vol: {first.get('search_volume')}")
            else:
                st.caption(f"🔧 No tasks in response!")
                st.caption(f"🔧 Full response (first 500 chars): {response.text[:500]}")
        
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
    
    # Alias for backward compatibility
    get_metrics_sync = get_keyword_metrics


def get_dataforseo_client() -> Optional[DataForSEOClient]:
    """
    Get DataForSEO client if configured.
    
    Returns:
        DataForSEOClient or None if not configured
    """
    client = DataForSEOClient()
    return client if client.is_configured else None