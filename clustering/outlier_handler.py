"""
Outlier handling and reassignment.
Minimizes unclustered keywords.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class OutlierReassignment:
    """Result of outlier reassignment."""
    
    keyword_index: int
    original_cluster: int  # -1 for outliers
    new_cluster: int
    similarity: float
    confidence: str  # "high", "medium", "low"


class OutlierHandler:
    """
    Handles outlier reassignment to reduce unclustered keywords.
    Uses cosine similarity to find best cluster matches.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        confidence_thresholds: Dict[str, float] = None
    ):
        """
        Initialize outlier handler.
        
        Args:
            similarity_threshold: Minimum similarity for reassignment
            confidence_thresholds: Thresholds for confidence levels
        """
        self.similarity_threshold = similarity_threshold
        self.confidence_thresholds = confidence_thresholds or {
            "high": 0.85,
            "medium": 0.75,
            "low": 0.65
        }
    
    def reassign_outliers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray = None
    ) -> Tuple[np.ndarray, List[OutlierReassignment]]:
        """
        Reassign outliers to nearest clusters.
        
        Args:
            embeddings: All keyword embeddings
            labels: Current cluster labels (-1 for outliers)
            centroids: Cluster centroids (computed if not provided)
        
        Returns:
            Tuple of (new_labels, list of reassignments)
        """
        new_labels = labels.copy()
        reassignments = []
        
        # Find outlier indices
        outlier_indices = np.where(labels == -1)[0]
        
        if len(outlier_indices) == 0:
            return new_labels, reassignments
        
        # Get unique cluster IDs (excluding outliers)
        cluster_ids = np.unique(labels[labels >= 0])
        
        if len(cluster_ids) == 0:
            return new_labels, reassignments
        
        # Compute centroids if not provided
        if centroids is None:
            centroids = self._compute_centroids(embeddings, labels, cluster_ids)
        
        # Get outlier embeddings
        outlier_embeddings = embeddings[outlier_indices]
        
        # Calculate similarity to all centroids
        similarities = cosine_similarity(outlier_embeddings, centroids)
        
        for i, outlier_idx in enumerate(outlier_indices):
            # Find best matching cluster
            best_cluster_local = np.argmax(similarities[i])
            best_similarity = similarities[i, best_cluster_local]
            best_cluster = cluster_ids[best_cluster_local]
            
            if best_similarity >= self.similarity_threshold:
                # Reassign to this cluster
                new_labels[outlier_idx] = best_cluster
                
                # Determine confidence
                confidence = self._get_confidence(best_similarity)
                
                reassignment = OutlierReassignment(
                    keyword_index=outlier_idx,
                    original_cluster=-1,
                    new_cluster=best_cluster,
                    similarity=best_similarity,
                    confidence=confidence
                )
                reassignments.append(reassignment)
        
        return new_labels, reassignments
    
    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_ids: np.ndarray
    ) -> np.ndarray:
        """
        Compute cluster centroids.
        
        Args:
            embeddings: All embeddings
            labels: Cluster labels
            cluster_ids: Unique cluster IDs
        
        Returns:
            Array of centroids
        """
        centroids = []
        
        for cluster_id in cluster_ids:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def _get_confidence(self, similarity: float) -> str:
        """Get confidence level based on similarity."""
        if similarity >= self.confidence_thresholds["high"]:
            return "high"
        elif similarity >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def find_similar_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.8
    ) -> List[Tuple[int, int, float]]:
        """
        Find clusters that are similar and might be merged.
        
        Args:
            embeddings: All embeddings
            labels: Cluster labels
            threshold: Similarity threshold for merge candidates
        
        Returns:
            List of (cluster1, cluster2, similarity) tuples
        """
        cluster_ids = np.unique(labels[labels >= 0])
        
        if len(cluster_ids) < 2:
            return []
        
        # Compute centroids
        centroids = self._compute_centroids(embeddings, labels, cluster_ids)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(centroids)
        
        # Find similar pairs
        merge_candidates = []
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                sim = similarities[i, j]
                if sim >= threshold:
                    merge_candidates.append((
                        int(cluster_ids[i]),
                        int(cluster_ids[j]),
                        float(sim)
                    ))
        
        # Sort by similarity descending
        merge_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return merge_candidates
    
    def get_outlier_stats(
        self,
        original_labels: np.ndarray,
        new_labels: np.ndarray
    ) -> dict:
        """
        Get statistics about outlier handling.
        
        Args:
            original_labels: Labels before reassignment
            new_labels: Labels after reassignment
        
        Returns:
            Statistics dict
        """
        original_outliers = np.sum(original_labels == -1)
        remaining_outliers = np.sum(new_labels == -1)
        reassigned = original_outliers - remaining_outliers
        
        return {
            "original_outliers": int(original_outliers),
            "remaining_outliers": int(remaining_outliers),
            "reassigned": int(reassigned),
            "reassignment_rate": (
                reassigned / original_outliers
                if original_outliers > 0 else 0
            ),
            "final_cluster_rate": (
                1 - remaining_outliers / len(new_labels)
                if len(new_labels) > 0 else 0
            )
        }