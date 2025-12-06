"""
Fine-grained clustering using HDBSCAN.
Second stage of the clustering pipeline.
"""
import numpy as np
import hdbscan
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class FineClusterResult:
    """Result of fine-grained clustering."""
    
    labels: np.ndarray  # -1 indicates outliers
    n_clusters: int
    outlier_indices: np.ndarray
    probabilities: np.ndarray  # Cluster membership probabilities
    
    @property
    def n_outliers(self) -> int:
        return len(self.outlier_indices)
    
    @property
    def outlier_rate(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return self.n_outliers / len(self.labels)
    
    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """Get indices of keywords in a cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster (excluding outliers)."""
        unique, counts = np.unique(
            self.labels[self.labels >= 0],
            return_counts=True
        )
        return dict(zip(unique, counts))


class FineClusterer:
    """
    Second-stage clustering using HDBSCAN.
    Automatically determines number of clusters and identifies outliers.
    Optimized for maximum coverage (minimize outliers).
    """
    
    def __init__(
        self,
        min_cluster_size: int = 3,  # Lowered from 5
        min_samples: int = 2,       # Lowered from 3
        cluster_selection_method: str = "leaf",  # Changed from "eom"
        metric: str = "euclidean"
    ):
        """
        Initialize fine clusterer.
        
        Args:
            min_cluster_size: Minimum keywords for a cluster (default: 3)
            min_samples: Samples for core point (default: 2)
            cluster_selection_method: "leaf" for more clusters, less outliers
            metric: Distance metric
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
    
    def fit(self, embeddings: np.ndarray) -> FineClusterResult:
        """
        Perform fine-grained clustering on embeddings.
        
        Args:
            embeddings: Keyword embeddings (n_keywords, n_dims)
        
        Returns:
            FineClusterResult with cluster assignments
        """
        if len(embeddings) < self.min_cluster_size:
            # Too few for clustering, treat all as one cluster
            return FineClusterResult(
                labels=np.zeros(len(embeddings), dtype=int),
                n_clusters=1,
                outlier_indices=np.array([]),
                probabilities=np.ones(len(embeddings))
            )
        
        # Initialize and fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True
        )
        
        labels = clusterer.fit_predict(embeddings)
        probabilities = clusterer.probabilities_
        
        # Get outlier indices (label == -1)
        outlier_indices = np.where(labels == -1)[0]
        
        # Count actual clusters (excluding outliers)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return FineClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            outlier_indices=outlier_indices,
            probabilities=probabilities
        )
    
    def fit_subcluster(
        self,
        embeddings: np.ndarray,
        parent_indices: np.ndarray,
        base_cluster_id: int
    ) -> Dict[int, np.ndarray]:
        """
        Subcluster a coarse cluster into finer groups.
        
        Args:
            embeddings: Full embedding matrix
            parent_indices: Indices belonging to parent cluster
            base_cluster_id: Starting ID for new clusters
        
        Returns:
            Dict mapping new_cluster_id -> original indices
        """
        if len(parent_indices) < self.min_cluster_size:
            # Too small to subcluster
            return {base_cluster_id: parent_indices}
        
        # Extract embeddings for this cluster
        cluster_embeddings = embeddings[parent_indices]
        
        # Fit HDBSCAN
        result = self.fit(cluster_embeddings)
        
        # Map back to original indices
        subclusters = {}
        
        for local_cluster_id in range(result.n_clusters):
            local_indices = result.get_cluster_indices(local_cluster_id)
            original_indices = parent_indices[local_indices]
            new_cluster_id = base_cluster_id + local_cluster_id
            subclusters[new_cluster_id] = original_indices
        
        # Handle outliers - put in separate "cluster"
        if result.n_outliers > 0:
            outlier_original = parent_indices[result.outlier_indices]
            outlier_id = base_cluster_id + result.n_clusters
            subclusters[outlier_id] = outlier_original
        
        return subclusters
    
    def adaptive_parameters(
        self,
        n_keywords: int
    ) -> Dict[str, int]:
        """
        Suggest adaptive parameters based on dataset size.
        Optimized for maximum coverage (fewer outliers).
        
        Args:
            n_keywords: Number of keywords in cluster
        
        Returns:
            Dict with suggested parameters
        """
        if n_keywords < 50:
            return {
                "min_cluster_size": 2,
                "min_samples": 1
            }
        elif n_keywords < 200:
            return {
                "min_cluster_size": 3,
                "min_samples": 2
            }
        elif n_keywords < 1000:
            return {
                "min_cluster_size": 4,
                "min_samples": 2
            }
        else:
            return {
                "min_cluster_size": 5,
                "min_samples": 3
            }