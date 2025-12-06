"""
Batch processor for large-scale embedding generation.
Handles chunking and checkpointing for 100K+ keywords.
"""
import streamlit as st
from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass

from embedding.openai_embedder import OpenAIEmbedder
from storage.supabase_client import SupabaseClient


@dataclass
class EmbeddingProgress:
    """Progress state for embedding generation."""
    
    total_keywords: int = 0
    processed_keywords: int = 0
    current_batch: int = 0
    total_batches: int = 0
    
    @property
    def progress(self) -> float:
        if self.total_keywords == 0:
            return 0.0
        return self.processed_keywords / self.total_keywords


class BatchEmbeddingProcessor:
    """
    Processes large keyword lists in batches with checkpointing.
    Designed for 100K+ keywords on Streamlit Cloud.
    """
    
    CHUNK_SIZE = 5000  # Keywords per processing chunk
    CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N keywords
    
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        supabase: SupabaseClient = None
    ):
        """
        Initialize batch processor.
        
        Args:
            embedder: OpenAI embedder instance
            supabase: Optional Supabase client for persistence
        """
        self.embedder = embedder
        self.db = supabase
    
    def process_keywords(
        self,
        keywords: List[str],
        job_id: str = None,
        progress_callback: Callable = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all keywords with chunking.
        
        Args:
            keywords: List of keywords
            job_id: Optional job ID for database storage
            progress_callback: Optional (EmbeddingProgress) callback
        
        Returns:
            Dict mapping keyword -> embedding array
        """
        progress = EmbeddingProgress(
            total_keywords=len(keywords),
            total_batches=(len(keywords) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        )
        
        results = {}
        
        for chunk_idx in range(0, len(keywords), self.CHUNK_SIZE):
            chunk = keywords[chunk_idx:chunk_idx + self.CHUNK_SIZE]
            progress.current_batch = chunk_idx // self.CHUNK_SIZE + 1
            
            # Generate embeddings for chunk
            chunk_embeddings = self.embedder.embed_all_sync(
                chunk,
                progress_callback=lambda c, t: self._update_progress(
                    progress,
                    chunk_idx + c,
                    progress_callback
                )
            )
            
            # Map keywords to embeddings
            for i, kw in enumerate(chunk):
                results[kw] = chunk_embeddings[i]
            
            # Save to database if available
            if self.db and job_id:
                self._save_chunk_embeddings(
                    job_id,
                    chunk,
                    chunk_embeddings
                )
            
            progress.processed_keywords = min(
                chunk_idx + self.CHUNK_SIZE,
                len(keywords)
            )
            
            if progress_callback:
                progress_callback(progress)
        
        return results
    
    def _update_progress(
        self,
        progress: EmbeddingProgress,
        current: int,
        callback: Callable
    ):
        """Update progress and notify callback."""
        progress.processed_keywords = current
        if callback:
            callback(progress)
    
    def _save_chunk_embeddings(
        self,
        job_id: str,
        keywords: List[str],
        embeddings: np.ndarray
    ):
        """Save embeddings to Supabase."""
        if not self.db:
            return
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            keyword_embeddings = {
                kw: embeddings[i].tolist()
                for i, kw in enumerate(keywords)
            }
            self.db.update_keyword_embeddings_batch(job_id, keyword_embeddings)
        except Exception as e:
            st.warning(f"Failed to save embeddings to database: {e}")
    
    def estimate_processing_time(
        self,
        n_keywords: int
    ) -> dict:
        """
        Estimate processing time and cost.
        
        Args:
            n_keywords: Number of keywords
        
        Returns:
            Dict with estimates
        """
        # Based on benchmarks:
        # - OpenAI API: ~100 embeddings per second
        # - Batch overhead: ~0.5 seconds per 100 keywords
        
        embedding_time = n_keywords / 100  # seconds
        overhead_time = (n_keywords / 100) * 0.5
        total_time = embedding_time + overhead_time
        
        # Cost estimate
        cost = self.embedder.estimate_cost(n_keywords)
        
        return {
            "keywords": n_keywords,
            "estimated_seconds": int(total_time),
            "estimated_minutes": round(total_time / 60, 1),
            "estimated_cost_usd": round(cost, 2),
            "batches": (n_keywords + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        }
    
    def get_memory_estimate(self, n_keywords: int) -> dict:
        """
        Estimate memory usage.
        
        Args:
            n_keywords: Number of keywords
        
        Returns:
            Memory estimate dict
        """
        # Embedding dimensions
        dims = self.embedder.DIMENSIONS  # 3072
        
        # float32 = 4 bytes per dimension
        bytes_per_embedding = dims * 4
        total_bytes = n_keywords * bytes_per_embedding
        
        # Add overhead for Python objects (~30%)
        total_with_overhead = total_bytes * 1.3
        
        return {
            "embeddings_mb": round(total_bytes / (1024 * 1024), 1),
            "with_overhead_mb": round(total_with_overhead / (1024 * 1024), 1),
            "fits_in_memory": total_with_overhead < 800 * 1024 * 1024  # 800MB
        }