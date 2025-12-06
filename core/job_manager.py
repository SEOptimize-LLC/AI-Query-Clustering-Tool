"""
Job management for clustering pipeline.
Handles job creation, state management, and persistence.
"""
import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
import json

from core.progress_tracker import ProgressTracker, ProcessingPhase
from core.exceptions import ProcessingError


class JobStatus(Enum):
    """Job status enumeration."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class JobConfig:
    """Configuration for a clustering job."""
    
    # Location settings
    location_code: int = 2840  # US
    location_name: str = "United States"
    language_code: str = "en"
    
    # Clustering settings
    serp_validation_count: int = 10
    serp_overlap_threshold: float = 0.4
    outlier_similarity_threshold: float = 0.7
    
    # LLM settings
    llm_model: str = "anthropic/claude-sonnet-4"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "location_code": self.location_code,
            "location_name": self.location_name,
            "language_code": self.language_code,
            "serp_validation_count": self.serp_validation_count,
            "serp_overlap_threshold": self.serp_overlap_threshold,
            "outlier_similarity_threshold": self.outlier_similarity_threshold,
            "llm_model": self.llm_model
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "JobConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ClusteringJob:
    """Represents a clustering job."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    config: JobConfig = field(default_factory=JobConfig)
    
    # Keyword data
    total_keywords: int = 0
    processed_keywords: int = 0
    keywords: List[str] = field(default_factory=list)
    
    # Progress tracking
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    current_phase: Optional[str] = None
    
    # Results
    cluster_count: int = 0
    total_search_volume: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    def start(self):
        """Start the job."""
        self.status = JobStatus.PROCESSING
        self.updated_at = datetime.now()
        self.progress.job_id = self.id
    
    def complete(self, cluster_count: int, total_search_volume: int):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.cluster_count = cluster_count
        self.total_search_volume = total_search_volume
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now()
    
    def pause(self):
        """Pause the job."""
        self.status = JobStatus.PAUSED
        self.updated_at = datetime.now()
    
    def resume(self):
        """Resume the job."""
        self.status = JobStatus.PROCESSING
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "total_keywords": self.total_keywords,
            "processed_keywords": self.processed_keywords,
            "current_phase": self.current_phase,
            "cluster_count": self.cluster_count,
            "total_search_volume": self.total_search_volume,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error_message": self.error_message,
            "progress": self.progress.to_dict()
        }


class JobManager:
    """
    Manages clustering jobs.
    Uses Streamlit session state for in-memory state
    and Supabase for persistence.
    """
    
    def __init__(self, supabase_client=None):
        self.db = supabase_client
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state for job management."""
        if "current_job" not in st.session_state:
            st.session_state.current_job = None
        if "job_history" not in st.session_state:
            st.session_state.job_history = []
    
    def create_job(
        self,
        keywords: List[str],
        config: JobConfig = None
    ) -> ClusteringJob:
        """
        Create a new clustering job.
        
        Args:
            keywords: List of keywords to cluster
            config: Job configuration
        
        Returns:
            ClusteringJob instance
        """
        job = ClusteringJob(
            config=config or JobConfig(),
            total_keywords=len(keywords),
            keywords=keywords
        )
        
        st.session_state.current_job = job
        
        # Persist to database if available
        if self.db:
            self._save_to_db(job)
        
        return job
    
    def get_current_job(self) -> Optional[ClusteringJob]:
        """Get the current active job."""
        return st.session_state.get("current_job")
    
    def update_job(self, job: ClusteringJob):
        """Update job state."""
        job.updated_at = datetime.now()
        st.session_state.current_job = job
        
        if self.db:
            self._update_in_db(job)
    
    def complete_job(
        self,
        cluster_count: int,
        total_search_volume: int
    ):
        """Complete the current job."""
        job = self.get_current_job()
        if job:
            job.complete(cluster_count, total_search_volume)
            self.update_job(job)
            st.session_state.job_history.append(job)
    
    def fail_job(self, error: str):
        """Mark current job as failed."""
        job = self.get_current_job()
        if job:
            job.fail(error)
            self.update_job(job)
    
    def _save_to_db(self, job: ClusteringJob):
        """Save job to Supabase database."""
        if not self.db:
            return
        
        try:
            self.db.client.table("jobs").insert({
                "id": job.id,
                "status": job.status.value,
                "total_keywords": job.total_keywords,
                "processed_keywords": job.processed_keywords,
                "current_phase": job.current_phase,
                "config": job.config.to_dict()
            }).execute()
        except Exception as e:
            st.warning(f"Could not save job to database: {e}")
    
    def _update_in_db(self, job: ClusteringJob):
        """Update job in Supabase database."""
        if not self.db:
            return
        
        try:
            self.db.client.table("jobs").update({
                "status": job.status.value,
                "processed_keywords": job.processed_keywords,
                "current_phase": job.current_phase,
                "error_message": job.error_message,
                "updated_at": "NOW()"
            }).eq("id", job.id).execute()
        except Exception as e:
            st.warning(f"Could not update job in database: {e}")
    
    def load_job_from_db(self, job_id: str) -> Optional[ClusteringJob]:
        """Load a job from database."""
        if not self.db:
            return None
        
        try:
            result = self.db.client.table("jobs").select("*").eq(
                "id", job_id
            ).single().execute()
            
            if result.data:
                return self._job_from_db_row(result.data)
        except Exception as e:
            st.warning(f"Could not load job from database: {e}")
        
        return None
    
    def _job_from_db_row(self, row: dict) -> ClusteringJob:
        """Create ClusteringJob from database row."""
        job = ClusteringJob(
            id=row["id"],
            status=JobStatus(row["status"]),
            config=JobConfig.from_dict(row.get("config", {})),
            total_keywords=row.get("total_keywords", 0),
            processed_keywords=row.get("processed_keywords", 0),
            current_phase=row.get("current_phase")
        )
        return job
    
    def get_resumable_jobs(self) -> List[Dict[str, Any]]:
        """Get list of jobs that can be resumed."""
        if not self.db:
            return []
        
        try:
            result = self.db.client.table("jobs").select(
                "id", "status", "total_keywords", 
                "processed_keywords", "created_at"
            ).in_(
                "status", ["processing", "paused"]
            ).order("created_at", desc=True).limit(10).execute()
            
            return result.data or []
        except Exception:
            return []