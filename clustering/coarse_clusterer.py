"""
Coarse clustering using MiniBatch K-Means.
First stage of the clustering pipeline.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CoarseClusterResult:
    """Result of coarse clustering."""
    
    labels: np.ndarray  # Cluster label for each keyword
    n_clusters: int
    centroids: np.ndarray  # Cluster centroids
    inertia: float  # Sum of squared distances
    
    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """Get indices of keywords in a cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


class CoarseClusterer:
    """
    First-stage clustering using MiniBatch K-Means.
    Efficient for large datasets, provides initial groupings.
    """
    
    def __init__(
        self,
        cluster_ratio: float = 0.5,
        min_clusters: int = 10,
        max_clusters: int = 1000,
        batch_size: int = 1000,
        n_init: int = 3,
        random_state: int = 42
    ):
        """
        Initialize coarse clusterer.
        
        Args:
            cluster_ratio: k = sqrt(n / cluster_ratio)
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            batch_size: Mini-batch size for K-Means
            n_init: Number of initializations
            random_state: Random seed for reproducibility
        """
        self.cluster_ratio = cluster_ratio
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.batch_size = batch_size
        self.n_init = n_init
        self.random_state = random_state
    
    def fit(self, embeddings: np.ndarray) -> CoarseClusterResult:
        """
        Perform coarse clustering on embeddings.
        
        Args:
            embeddings: Keyword embeddings (n_keywords, n_dims)
        
        Returns:
            CoarseClusterResult with cluster assignments
        """
        n_keywords = len(embeddings)
        
        # Calculate optimal number of clusters
        n_clusters = self._calculate_n_clusters(n_keywords)
        
        # Initialize and fit MiniBatch K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(self.batch_size, n_keywords),
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=100
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        return CoarseClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=kmeans.cluster_centers_,
            inertia=kmeans.inertia_
        )
    
    def _calculate_n_clusters(self, n_keywords: int) -> int:
        """
        Calculate optimal number of clusters.
        Uses sqrt(n / ratio) heuristic.
        
        Args:
            n_keywords: Number of keywords
        
        Returns:
            Number of clusters
        """
        k = int(np.sqrt(n_keywords / self.cluster_ratio))
        return max(self.min_clusters, min(k, self.max_clusters))
    
    def estimate_cluster_quality(
        self,
        embeddings: np.ndarray,
        result: CoarseClusterResult
    ) -> Dict[int, float]:
        """
        Estimate quality of each cluster using silhouette-like metric.
        
        Args:
            embeddings: Keyword embeddings
            result: Clustering result
        
        Returns:
            Dict mapping cluster_id -> quality score (0-1)
        """
        from sklearn.metrics import pairwise_distances
        
        quality = {}
        
        for cluster_id in range(result.n_clusters):
            indices = result.get_cluster_indices(cluster_id)
            if len(indices) < 2:
                quality[cluster_id] = 0.0
                continue
            
            cluster_embeddings = embeddings[indices]
            centroid = result.centroids[cluster_id]
            
            # Calculate average distance to centroid
            distances = np.linalg.norm(
                cluster_embeddings - centroid,
                axis=1
            )
            avg_distance = np.mean(distances)
            
            # Normalize to 0-1 (lower distance = higher quality)
            # Using exponential decay
            quality[cluster_id] = np.exp(-avg_distance)
        
        return quality
    
    def split_large_clusters(
        self,
        embeddings: np.ndarray,
        result: CoarseClusterResult,
        max_size: int = 500
    ) -> CoarseClusterResult:
        """
        Split clusters that are too large.
        
        Args:
            embeddings: Keyword embeddings
            result: Initial clustering result
            max_size: Maximum cluster size before splitting
        
        Returns:
            Updated clustering result
        """
        new_labels = result.labels.copy()
        new_centroids = list(result.centroids)
        next_cluster_id = result.n_clusters
        
        for cluster_id in range(result.n_clusters):
            indices = result.get_cluster_indices(cluster_id)
            
            if len(indices) <= max_size:
                continue
            
            # Split this cluster
            cluster_embeddings = embeddings[indices]
            n_subclusters = (len(indices) // max_size) + 1
            
            sub_kmeans = MiniBatchKMeans(
                n_clusters=n_subclusters,
                batch_size=min(100, len(indices)),
                n_init=2,
                random_state=self.random_state
            )
            
            sub_labels = sub_kmeans.fit_predict(cluster_embeddings)
            
            # Reassign labels (keep first subcluster as original)
            for sub_id in range(1, n_subclusters):
                sub_indices = indices[sub_labels == sub_id]
                new_labels[sub_indices] = next_cluster_id
                new_centroids.append(sub_kmeans.cluster_centers_[sub_id])
                next_cluster_id += 1
            
            # Update centroid for original cluster
            new_centroids[cluster_id] = sub_kmeans.cluster_centers_[0]
        
        return CoarseClusterResult(
            labels=new_labels,
            n_clusters=next_cluster_id,
            centroids=np.array(new_centroids),
            inertia=result.inertia  # Approximate
        )