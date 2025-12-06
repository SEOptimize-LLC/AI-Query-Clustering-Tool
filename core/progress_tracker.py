"""
Progress tracking for multi-phase clustering pipeline.
"""
import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, List
from enum import Enum


class ProcessingPhase(Enum):
    """Processing pipeline phases."""
    
    UPLOAD = "upload"
    ENRICHMENT = "enrichment"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    SERP_VALIDATION = "serp_validation"
    LABELING = "labeling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PhaseProgress:
    """Progress state for a single phase."""
    
    phase: ProcessingPhase
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    current: int = 0
    total: int = 0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def start(self, total: int = 0, message: str = ""):
        """Mark phase as started."""
        self.status = "running"
        self.total = total
        self.message = message
        self.started_at = datetime.now()
        self.progress = 0.0
        self.current = 0
    
    def update(self, current: int, message: str = ""):
        """Update progress within phase."""
        self.current = current
        if self.total > 0:
            self.progress = min(1.0, current / self.total)
        if message:
            self.message = message
    
    def complete(self, message: str = ""):
        """Mark phase as completed."""
        self.status = "completed"
        self.progress = 1.0
        self.current = self.total
        self.completed_at = datetime.now()
        if message:
            self.message = message
    
    def fail(self, error: str):
        """Mark phase as failed."""
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.now()
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate time remaining in seconds."""
        if self.progress <= 0 or self.status != "running":
            return None
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return None
        return (elapsed / self.progress) - elapsed


@dataclass
class ProgressTracker:
    """
    Tracks progress across all phases of the clustering pipeline.
    Integrates with Streamlit for real-time UI updates.
    """
    
    job_id: Optional[str] = None
    phases: dict = field(default_factory=dict)
    current_phase: Optional[ProcessingPhase] = None
    _callbacks: List[Callable] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize phase progress for all phases."""
        for phase in ProcessingPhase:
            if phase not in [ProcessingPhase.COMPLETED, ProcessingPhase.FAILED]:
                self.phases[phase] = PhaseProgress(phase=phase)
    
    def start_phase(
        self,
        phase: ProcessingPhase,
        total: int = 0,
        message: str = ""
    ):
        """Start a new phase."""
        self.current_phase = phase
        self.phases[phase].start(total, message)
        self._notify()
    
    def update_phase(
        self,
        current: int,
        message: str = ""
    ):
        """Update current phase progress."""
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].update(current, message)
            self._notify()
    
    def complete_phase(self, message: str = ""):
        """Complete current phase."""
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].complete(message)
            self._notify()
    
    def fail_phase(self, error: str):
        """Mark current phase as failed."""
        if self.current_phase and self.current_phase in self.phases:
            self.phases[self.current_phase].fail(error)
            self._notify()
    
    def add_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self._callbacks.append(callback)
    
    def _notify(self):
        """Notify all callbacks of progress update."""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception:
                pass  # Don't let callback errors break tracking
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall progress across all phases."""
        active_phases = [
            p for p in ProcessingPhase 
            if p not in [ProcessingPhase.COMPLETED, ProcessingPhase.FAILED]
        ]
        
        total_progress = sum(
            self.phases[p].progress for p in active_phases
        )
        return total_progress / len(active_phases)
    
    @property
    def is_completed(self) -> bool:
        """Check if all phases are completed."""
        active_phases = [
            p for p in ProcessingPhase 
            if p not in [ProcessingPhase.COMPLETED, ProcessingPhase.FAILED]
        ]
        return all(
            self.phases[p].status == "completed" for p in active_phases
        )
    
    @property
    def is_failed(self) -> bool:
        """Check if any phase has failed."""
        return any(
            p.status == "failed" for p in self.phases.values()
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/serialization."""
        return {
            "job_id": self.job_id,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "overall_progress": self.overall_progress,
            "phases": {
                phase.value: {
                    "status": p.status,
                    "progress": p.progress,
                    "current": p.current,
                    "total": p.total,
                    "message": p.message,
                    "error": p.error
                }
                for phase, p in self.phases.items()
            }
        }
    
    def render_streamlit_progress(self, container=None):
        """
        Render progress UI in Streamlit.
        
        Args:
            container: Streamlit container to render in (default: main)
        """
        target = container or st
        
        # Overall progress bar
        target.progress(
            self.overall_progress,
            text=f"Overall Progress: {self.overall_progress:.1%}"
        )
        
        # Phase details
        phase_names = {
            ProcessingPhase.UPLOAD: "📤 Upload",
            ProcessingPhase.ENRICHMENT: "📊 Enrichment",
            ProcessingPhase.EMBEDDING: "🧠 Embedding",
            ProcessingPhase.CLUSTERING: "🎯 Clustering",
            ProcessingPhase.SERP_VALIDATION: "🔍 SERP Validation",
            ProcessingPhase.LABELING: "🏷️ Labeling"
        }
        
        cols = target.columns(len(phase_names))
        
        for i, (phase, name) in enumerate(phase_names.items()):
            p = self.phases[phase]
            with cols[i]:
                if p.status == "completed":
                    st.success(f"{name}\n✅ Done")
                elif p.status == "running":
                    st.info(f"{name}\n⏳ {p.progress:.0%}")
                elif p.status == "failed":
                    st.error(f"{name}\n❌ Failed")
                else:
                    st.write(f"{name}\n⏸️ Pending")
        
        # Current phase details
        if self.current_phase and self.current_phase in self.phases:
            p = self.phases[self.current_phase]
            if p.status == "running" and p.total > 0:
                target.caption(
                    f"{p.message} ({p.current:,}/{p.total:,})"
                )
                if p.eta_seconds:
                    mins = int(p.eta_seconds // 60)
                    secs = int(p.eta_seconds % 60)
                    target.caption(f"ETA: {mins}m {secs}s")