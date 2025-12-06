"""
Serper.dev API client for Google SERP data.
"""
import streamlit as st
import asyncio
import aiohttp
import requests
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from core.exceptions import APIError, RateLimitError


@dataclass
class SERPResult:
    """SERP result for a keyword."""
    
    keyword: str
    organic: List[Dict] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    people_also_ask: List[str] = field(default_factory=list)
    related_searches: List[str] = field(default_factory=list)


class SerperClient:
    """
    Client for Serper.dev Google SERP API.
    Used for SERP validation during clustering.
    """
    
    BASE_URL = "https://google.serper.dev"
    
    def __init__(self, api_key: str = None):
        """
        Initialize Serper client.
        
        Args:
            api_key: Serper.dev API key
        """
        self.api_key = api_key or st.secrets.get("SERPER_API_KEY", "")
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    async def search(
        self,
        query: str,
        location: str = "United States",
        num_results: int = 10
    ) -> SERPResult:
        """
        Perform a single Google search.
        
        Args:
            query: Search query
            location: Geographic location
            num_results: Number of results to fetch
        
        Returns:
            SERPResult with organic results and related data
        """
        if not self.is_configured:
            raise APIError(
                "Serper API key not configured",
                service="Serper"
            )
        
        async with aiohttp.ClientSession() as session:
            return await self._fetch_serp(
                session,
                query,
                location,
                num_results
            )
    
    async def batch_search(
        self,
        queries: List[str],
        location: str = "United States",
        num_results: int = 10,
        concurrency: int = 5,
        progress_callback: Callable = None
    ) -> Dict[str, SERPResult]:
        """
        Perform batch SERP searches with rate limiting.
        
        Args:
            queries: List of search queries
            location: Geographic location
            num_results: Results per query
            concurrency: Max concurrent requests
            progress_callback: Optional progress callback
        
        Returns:
            Dict mapping query -> SERPResult
        """
        if not self.is_configured:
            raise APIError(
                "Serper API key not configured",
                service="Serper"
            )
        
        results = {}
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_search(query: str) -> tuple:
            async with semaphore:
                try:
                    result = await self.search(query, location, num_results)
                    return query, result
                except Exception as e:
                    # Return empty result on error
                    return query, SERPResult(keyword=query)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, query in enumerate(queries):
                task = asyncio.create_task(
                    self._fetch_serp(session, query, location, num_results)
                )
                tasks.append((query, task))
            
            for i, (query, task) in enumerate(tasks):
                try:
                    result = await task
                    results[query] = result
                except Exception:
                    results[query] = SERPResult(keyword=query)
                
                if progress_callback:
                    progress_callback(i + 1, len(queries))
        
        return results
    
    async def _fetch_serp(
        self,
        session: aiohttp.ClientSession,
        query: str,
        location: str,
        num_results: int
    ) -> SERPResult:
        """
        Fetch SERP data for a single query.
        
        Args:
            session: aiohttp session
            query: Search query
            location: Location string
            num_results: Number of results
        
        Returns:
            SERPResult
        """
        url = f"{self.BASE_URL}/search"
        
        payload = {
            "q": query,
            "location": location,
            "num": num_results
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        async with session.post(
            url,
            json=payload,
            headers=headers
        ) as response:
            if response.status == 429:
                raise RateLimitError(
                    service="Serper",
                    retry_after=60
                )
            
            if response.status != 200:
                text = await response.text()
                raise APIError(
                    f"Serper API error: {text}",
                    service="Serper",
                    status_code=response.status,
                    response=text
                )
            
            data = await response.json()
            return self._parse_response(query, data)
    
    def _parse_response(self, query: str, data: dict) -> SERPResult:
        """
        Parse Serper API response.
        
        Args:
            query: Original query
            data: API response JSON
        
        Returns:
            SERPResult
        """
        organic = data.get("organic", [])
        
        urls = [
            r.get("link", "")
            for r in organic
            if r.get("link")
        ]
        
        titles = [
            r.get("title", "")
            for r in organic
            if r.get("title")
        ]
        
        paa = [
            item.get("question", "")
            for item in data.get("peopleAlsoAsk", [])
            if item.get("question")
        ]
        
        related = [
            item.get("query", "")
            for item in data.get("relatedSearches", [])
            if item.get("query")
        ]
        
        return SERPResult(
            keyword=query,
            organic=[
                {
                    "url": r.get("link"),
                    "title": r.get("title"),
                    "position": r.get("position"),
                    "snippet": r.get("snippet")
                }
                for r in organic
            ],
            urls=urls,
            titles=titles,
            people_also_ask=paa,
            related_searches=related
        )
    
    def search_sync(
        self,
        query: str,
        location: str = "United States",
        num_results: int = 10
    ) -> SERPResult:
        """
        Synchronous search using requests library.
        """
        if not self.is_configured:
            raise APIError("Serper API key not configured", service="Serper")
        
        url = f"{self.BASE_URL}/search"
        payload = {"q": query, "location": location, "num": num_results}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 429:
            raise RateLimitError(service="Serper", retry_after=60)
        
        if response.status_code != 200:
            raise APIError(
                f"Serper API error: {response.text}",
                service="Serper",
                status_code=response.status_code,
                response=response.text
            )
        
        return self._parse_response(query, response.json())
    
    def batch_search_sync(
        self,
        queries: List[str],
        location: str = "United States",
        num_results: int = 10,
        concurrency: int = 5,
        progress_callback: Callable = None
    ) -> Dict[str, SERPResult]:
        """
        Synchronous batch search using requests library.
        """
        if not self.is_configured:
            raise APIError("Serper API key not configured", service="Serper")
        
        results = {}
        
        for i, query in enumerate(queries):
            try:
                results[query] = self.search_sync(query, location, num_results)
            except Exception:
                results[query] = SERPResult(keyword=query)
            
            if progress_callback:
                progress_callback(i + 1, len(queries))
        
        return results


def get_serper_client() -> Optional[SerperClient]:
    """
    Get Serper client if configured.
    
    Returns:
        SerperClient or None if not configured
    """
    client = SerperClient()
    return client if client.is_configured else None