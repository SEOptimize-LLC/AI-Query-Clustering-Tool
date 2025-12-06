"""
OpenAI embedding client for generating semantic embeddings.
"""
import streamlit as st
import asyncio
import openai
from typing import List, Optional, Callable
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from core.exceptions import EmbeddingError, RateLimitError


class OpenAIEmbedder:
    """
    OpenAI embedding client using text-embedding-3-large.
    Best quality embeddings for business-critical clustering.
    """
    
    MODEL = "text-embedding-3-large"
    DIMENSIONS = 3072
    BATCH_SIZE = 100  # Max texts per API request
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or st.secrets.get("OPENAI_API_KEY", "")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.sync_client = openai.OpenAI(api_key=self.api_key)
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError)
        )
    )
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed (max BATCH_SIZE)
        
        Returns:
            numpy array of shape (n_texts, DIMENSIONS)
        """
        if not self.is_configured:
            raise EmbeddingError("OpenAI API key not configured")
        
        if len(texts) > self.BATCH_SIZE:
            raise EmbeddingError(
                f"Batch too large: {len(texts)} > {self.BATCH_SIZE}"
            )
        
        try:
            response = await self.client.embeddings.create(
                model=self.MODEL,
                input=texts
            )
            
            # Extract embeddings in the same order as input
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            
            return np.array(embeddings, dtype=np.float32)
            
        except openai.RateLimitError as e:
            raise RateLimitError(
                service="OpenAI",
                retry_after=60
            )
        except openai.APIError as e:
            raise EmbeddingError(
                f"OpenAI API error: {str(e)}",
                batch_index=0
            )
    
    async def embed_all(
        self,
        texts: List[str],
        progress_callback: Callable = None
    ) -> np.ndarray:
        """
        Generate embeddings for all texts with batching.
        
        Args:
            texts: List of texts to embed
            progress_callback: Optional (current, total) callback
        
        Returns:
            numpy array of shape (n_texts, DIMENSIONS)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            
            embeddings = await self.embed_batch(batch)
            all_embeddings.append(embeddings)
            
            if progress_callback:
                progress_callback(
                    min(i + self.BATCH_SIZE, len(texts)),
                    len(texts)
                )
            
            # Small delay between batches
            if i + self.BATCH_SIZE < len(texts):
                await asyncio.sleep(0.1)
        
        return np.vstack(all_embeddings)
    
    def embed_batch_sync(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous version of embed_batch.
        """
        if not self.is_configured:
            raise EmbeddingError("OpenAI API key not configured")
        
        if len(texts) > self.BATCH_SIZE:
            raise EmbeddingError(
                f"Batch too large: {len(texts)} > {self.BATCH_SIZE}"
            )
        
        try:
            response = self.sync_client.embeddings.create(
                model=self.MODEL,
                input=texts
            )
            
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            
            return np.array(embeddings, dtype=np.float32)
            
        except openai.RateLimitError:
            raise RateLimitError(service="OpenAI", retry_after=60)
        except openai.APIError as e:
            raise EmbeddingError(f"OpenAI API error: {str(e)}")
    
    def embed_all_sync(
        self,
        texts: List[str],
        progress_callback: Callable = None
    ) -> np.ndarray:
        """
        Synchronous version of embed_all.
        Use this in Streamlit (which has its own event loop).
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            
            embeddings = self.embed_batch_sync(batch)
            all_embeddings.append(embeddings)
            
            if progress_callback:
                progress_callback(
                    min(i + self.BATCH_SIZE, len(texts)),
                    len(texts)
                )
        
        return np.vstack(all_embeddings)
    
    def estimate_cost(self, n_texts: int) -> float:
        """
        Estimate embedding cost.
        
        Args:
            n_texts: Number of texts to embed
        
        Returns:
            Estimated cost in USD
        """
        # text-embedding-3-large: $0.13 per 1M tokens
        # Assume average of 5 tokens per keyword
        avg_tokens = 5
        total_tokens = n_texts * avg_tokens
        cost_per_million = 0.13
        
        return (total_tokens / 1_000_000) * cost_per_million


def get_openai_embedder() -> Optional[OpenAIEmbedder]:
    """
    Get OpenAI embedder if configured.
    
    Returns:
        OpenAIEmbedder or None if not configured
    """
    embedder = OpenAIEmbedder()
    return embedder if embedder.is_configured else None