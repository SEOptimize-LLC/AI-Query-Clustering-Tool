"""
Supabase database client for persistent storage.
All data is stored in Supabase PostgreSQL for persistence across sessions.
"""
import streamlit as st
from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np

from core.exceptions import DatabaseError


class SupabaseClient:
    """
    Client for interacting with Supabase PostgreSQL database.
    Handles all CRUD operations for jobs, keywords, clusters, and caching.
    """
    
    def __init__(self, url: str, key: str):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL
            key: Supabase anon/public key
        """
        self.client: Client = create_client(url, key)
    
    # =========================================================================
    # Job Management
    # =========================================================================
    
    def create_job(
        self,
        total_keywords: int,
        config: dict
    ) -> str:
        """
        Create a new clustering job.
        
        Args:
            total_keywords: Number of keywords to process
            config: Job configuration dictionary
        
        Returns:
            Job ID (UUID)
        """
        try:
            result = self.client.table("jobs").insert({
                "total_keywords": total_keywords,
                "config": config
            }).execute()
            return result.data[0]["id"]
        except Exception as e:
            raise DatabaseError(
                f"Failed to create job: {e}",
                operation="insert",
                table="jobs"
            )
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        phase: str = None,
        processed_keywords: int = None,
        error_message: str = None
    ):
        """Update job status and progress."""
        update_data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if phase:
            update_data["current_phase"] = phase
        if processed_keywords is not None:
            update_data["processed_keywords"] = processed_keywords
        if error_message:
            update_data["error_message"] = error_message
        
        try:
            self.client.table("jobs").update(
                update_data
            ).eq("id", job_id).execute()
        except Exception as e:
            raise DatabaseError(
                f"Failed to update job: {e}",
                operation="update",
                table="jobs"
            )
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details by ID."""
        try:
            result = self.client.table("jobs").select("*").eq(
                "id", job_id
            ).single().execute()
            return result.data
        except Exception:
            return None
    
    # =========================================================================
    # Keyword Management
    # =========================================================================
    
    def insert_keywords(
        self,
        job_id: str,
        keywords: List[str],
        batch_size: int = 500
    ):
        """
        Bulk insert keywords for a job.
        
        Args:
            job_id: Job ID
            keywords: List of keyword strings
            batch_size: Number of keywords per insert batch
        """
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            rows = [
                {"job_id": job_id, "keyword": kw}
                for kw in batch
            ]
            
            try:
                self.client.table("keywords").insert(rows).execute()
            except Exception as e:
                raise DatabaseError(
                    f"Failed to insert keywords batch {i}: {e}",
                    operation="insert",
                    table="keywords"
                )
    
    def update_keyword_metrics(
        self,
        job_id: str,
        metrics: Dict[str, dict]
    ):
        """
        Update keyword metrics from DataForSEO.
        
        Args:
            job_id: Job ID
            metrics: Dict mapping keyword -> {search_volume, kd, cpc}
        """
        for keyword, data in metrics.items():
            try:
                self.client.table("keywords").update({
                    "search_volume": data.get("search_volume", 0),
                    "keyword_difficulty": data.get("keyword_difficulty", 0),
                    "cpc": data.get("cpc", 0)
                }).eq("job_id", job_id).eq("keyword", keyword).execute()
            except Exception:
                continue  # Skip failures, log if needed
    
    def update_keyword_embeddings_batch(
        self,
        job_id: str,
        keyword_embeddings: Dict[str, List[float]]
    ):
        """
        Update embeddings for multiple keywords.
        
        Args:
            job_id: Job ID
            keyword_embeddings: Dict mapping keyword -> embedding list
        """
        for keyword, embedding in keyword_embeddings.items():
            try:
                # Store as string representation for pgvector
                self.client.table("keywords").update({
                    "embedding": embedding
                }).eq("job_id", job_id).eq("keyword", keyword).execute()
            except Exception:
                continue
    
    def get_keywords_for_job(
        self,
        job_id: str,
        with_embeddings: bool = False,
        limit: int = None
    ) -> List[dict]:
        """
        Get all keywords for a job.
        
        Args:
            job_id: Job ID
            with_embeddings: Include embedding vectors
            limit: Maximum number of keywords to return
        """
        select = "keyword, search_volume, keyword_difficulty, cluster_id"
        if with_embeddings:
            select += ", embedding"
        
        query = self.client.table("keywords").select(select).eq(
            "job_id", job_id
        )
        
        if limit:
            query = query.limit(limit)
        
        try:
            result = query.execute()
            return result.data or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to get keywords: {e}",
                operation="select",
                table="keywords"
            )
    
    def assign_cluster_to_keywords(
        self,
        job_id: str,
        cluster_id: int,
        keywords: List[str]
    ):
        """Assign keywords to a cluster."""
        for keyword in keywords:
            try:
                self.client.table("keywords").update({
                    "cluster_id": cluster_id
                }).eq("job_id", job_id).eq("keyword", keyword).execute()
            except Exception:
                continue
    
    # =========================================================================
    # Cluster Management
    # =========================================================================
    
    def save_cluster(
        self,
        job_id: str,
        label: str,
        intent: str,
        centroid_keyword: str,
        keyword_count: int,
        total_search_volume: int,
        avg_keyword_difficulty: float,
        serp_validation_score: float = None,
        quality_score: float = None,
        parent_cluster_id: int = None
    ) -> int:
        """
        Save a cluster and return its ID.
        
        Returns:
            Cluster ID
        """
        try:
            result = self.client.table("clusters").insert({
                "job_id": job_id,
                "label": label,
                "intent": intent,
                "centroid_keyword": centroid_keyword,
                "keyword_count": keyword_count,
                "total_search_volume": total_search_volume,
                "avg_keyword_difficulty": avg_keyword_difficulty,
                "serp_validation_score": serp_validation_score,
                "quality_score": quality_score,
                "parent_cluster_id": parent_cluster_id
            }).execute()
            return result.data[0]["id"]
        except Exception as e:
            raise DatabaseError(
                f"Failed to save cluster: {e}",
                operation="insert",
                table="clusters"
            )
    
    def get_clusters_for_job(self, job_id: str) -> List[dict]:
        """Get all clusters for a job."""
        try:
            result = self.client.table("clusters").select("*").eq(
                "job_id", job_id
            ).order("total_search_volume", desc=True).execute()
            return result.data or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to get clusters: {e}",
                operation="select",
                table="clusters"
            )
    
    def get_cluster_keywords(
        self,
        job_id: str,
        cluster_id: int
    ) -> List[dict]:
        """Get all keywords in a cluster."""
        try:
            result = self.client.table("keywords").select(
                "keyword, search_volume, keyword_difficulty"
            ).eq("job_id", job_id).eq("cluster_id", cluster_id).order(
                "search_volume", desc=True
            ).execute()
            return result.data or []
        except Exception as e:
            raise DatabaseError(
                f"Failed to get cluster keywords: {e}",
                operation="select",
                table="keywords"
            )
    
    # =========================================================================
    # Metrics Cache
    # =========================================================================
    
    def get_cached_metrics(
        self,
        keywords: List[str],
        max_age_days: int = 7
    ) -> Dict[str, dict]:
        """
        Get cached keyword metrics.
        
        Args:
            keywords: List of keywords to look up
            max_age_days: Maximum age of cached data in days
        
        Returns:
            Dict mapping keyword -> metrics
        """
        if not keywords:
            return {}
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        try:
            result = self.client.table("metrics_cache").select("*").in_(
                "keyword", keywords
            ).gte("fetched_at", cutoff.isoformat()).execute()
            
            return {
                row["keyword"]: {
                    "search_volume": row["search_volume"],
                    "keyword_difficulty": row["keyword_difficulty"],
                    "cpc": row["cpc"],
                    "competition": row["competition"]
                }
                for row in (result.data or [])
            }
        except Exception:
            return {}
    
    def cache_metrics(self, metrics: Dict[str, dict]):
        """
        Cache keyword metrics.
        
        Args:
            metrics: Dict mapping keyword -> {search_volume, kd, cpc, competition}
        """
        rows = [
            {
                "keyword": kw,
                "search_volume": data.get("search_volume", 0),
                "keyword_difficulty": data.get("keyword_difficulty", 0),
                "cpc": data.get("cpc", 0),
                "competition": data.get("competition", 0),
                "fetched_at": datetime.now().isoformat()
            }
            for kw, data in metrics.items()
        ]
        
        if rows:
            try:
                self.client.table("metrics_cache").upsert(rows).execute()
            except Exception:
                pass  # Caching failure shouldn't break the flow
    
    # =========================================================================
    # SERP Cache
    # =========================================================================
    
    def get_cached_serp(
        self,
        keyword: str,
        max_age_hours: int = 24
    ) -> Optional[dict]:
        """
        Get cached SERP data for a keyword.
        
        Args:
            keyword: Keyword to look up
            max_age_hours: Maximum age of cached data in hours
        
        Returns:
            SERP data dict or None if not cached/expired
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            result = self.client.table("serp_cache").select("*").eq(
                "keyword", keyword
            ).gte("fetched_at", cutoff.isoformat()).execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception:
            return None
    
    def cache_serp(self, keyword: str, serp_data: dict):
        """
        Cache SERP data for a keyword.
        
        Args:
            keyword: Keyword
            serp_data: SERP data with urls, titles, etc.
        """
        try:
            self.client.table("serp_cache").upsert({
                "keyword": keyword,
                "urls": serp_data.get("urls", []),
                "titles": serp_data.get("titles", []),
                "people_also_ask": serp_data.get("people_also_ask", []),
                "related_searches": serp_data.get("related_searches", []),
                "fetched_at": datetime.now().isoformat()
            }).execute()
        except Exception:
            pass  # Caching failure shouldn't break the flow
    
    def get_cached_serps_batch(
        self,
        keywords: List[str],
        max_age_hours: int = 24
    ) -> Dict[str, dict]:
        """Get cached SERP data for multiple keywords."""
        if not keywords:
            return {}
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            result = self.client.table("serp_cache").select("*").in_(
                "keyword", keywords
            ).gte("fetched_at", cutoff.isoformat()).execute()
            
            return {row["keyword"]: row for row in (result.data or [])}
        except Exception:
            return {}
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def delete_job_data(self, job_id: str):
        """Delete all data for a job (cascades to keywords/clusters)."""
        try:
            self.client.table("jobs").delete().eq("id", job_id).execute()
        except Exception as e:
            raise DatabaseError(
                f"Failed to delete job: {e}",
                operation="delete",
                table="jobs"
            )
    
    def get_job_statistics(self, job_id: str) -> dict:
        """Get statistics for a completed job."""
        try:
            # Get cluster stats
            clusters = self.get_clusters_for_job(job_id)
            
            total_volume = sum(c.get("total_search_volume", 0) for c in clusters)
            avg_kd = np.mean([
                c.get("avg_keyword_difficulty", 0) for c in clusters
            ]) if clusters else 0
            
            # Get keyword stats
            keywords = self.get_keywords_for_job(job_id)
            clustered = sum(1 for k in keywords if k.get("cluster_id"))
            
            return {
                "total_clusters": len(clusters),
                "total_keywords": len(keywords),
                "clustered_keywords": clustered,
                "unclustered_keywords": len(keywords) - clustered,
                "cluster_rate": clustered / len(keywords) if keywords else 0,
                "total_search_volume": total_volume,
                "avg_keyword_difficulty": avg_kd
            }
        except Exception:
            return {}


@lru_cache()
def get_supabase_client() -> Optional[SupabaseClient]:
    """
    Get cached Supabase client instance.
    Returns None if credentials are not configured.
    """
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    
    if url and key:
        return SupabaseClient(url, key)
    return None