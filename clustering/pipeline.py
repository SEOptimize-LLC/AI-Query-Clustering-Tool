"""
Main clustering pipeline orchestrator.
Coordinates all stages of the clustering process.

Pipeline with cluster merging to prevent over-fragmentation:
1. Coarse clustering (MiniBatch K-Means)
2. Fine-grained sub-clustering (HDBSCAN)
3. CLUSTER MERGING (combines semantically similar clusters)
4. SERP validation (optional)
5. Outlier reassignment
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field

from clustering.coarse_clusterer import CoarseClusterer, CoarseClusterResult
from clustering.fine_clusterer import FineClusterer
from clustering.serp_validator import SERPValidator, ClusterValidation
from clustering.outlier_handler import OutlierHandler


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    
    cluster_id: int
    keyword_indices: np.ndarray
    keywords: List[str]
    centroid: np.ndarray
    size: int
    serp_validation: Optional[ClusterValidation] = None
    quality_score: float = 0.0


@dataclass
class ClusteringResult:
    """Complete result of clustering pipeline."""
    
    # Core results
    labels: np.ndarray
    n_clusters: int
    clusters: Dict[int, ClusterInfo] = field(default_factory=dict)
    
    # Statistics
    total_keywords: int = 0
    clustered_keywords: int = 0
    outlier_keywords: int = 0
    cluster_rate: float = 0.0
    
    # Validation
    validations: List[ClusterValidation] = field(default_factory=list)
    avg_serp_overlap: float = 0.0
    
    def get_cluster_keywords(
        self,
        cluster_id: int
    ) -> List[str]:
        """Get keywords in a cluster."""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id].keywords
        return []


class ClusteringPipeline:
    """
    Orchestrates the multi-stage clustering process with cluster merging.
    
    Pipeline:
    1. Coarse clustering (MiniBatch K-Means)
    2. Fine-grained sub-clustering (HDBSCAN)
    3. **CLUSTER MERGING** - Combines semantically similar clusters
    4. SERP validation (optional)
    5. Outlier reassignment
    
    Cluster merging prevents over-fragmentation of semantically related
    keywords like "odoo pricing" and "odoo online pricing".
    """
    
    def __init__(
        self,
        coarse_clusterer: CoarseClusterer = None,
        fine_clusterer: FineClusterer = None,
        serp_validator: SERPValidator = None,
        outlier_handler: OutlierHandler = None,
        merge_threshold: float = 0.80  # Merge clusters with >80% similarity
    ):
        """
        Initialize clustering pipeline.
        
        Args:
            coarse_clusterer: First-stage clusterer
            fine_clusterer: Second-stage clusterer
            serp_validator: SERP validation (optional)
            outlier_handler: Outlier reassignment handler
            merge_threshold: Similarity threshold for merging clusters
        """
        self.coarse = coarse_clusterer or CoarseClusterer()
        self.fine = fine_clusterer or FineClusterer()
        self.serp_validator = serp_validator
        self.outlier_handler = outlier_handler or OutlierHandler()
        self.merge_threshold = merge_threshold
    
    def run(
        self,
        embeddings: np.ndarray,
        keywords: List[str],
        keyword_data: Dict[str, dict] = None,
        progress_callback: Callable = None,
        skip_serp_validation: bool = False
    ) -> ClusteringResult:
        """
        Run the full clustering pipeline.
        
        Args:
            embeddings: Keyword embeddings (n_keywords, n_dims)
            keywords: List of keyword strings
            keyword_data: Optional dict with keyword metrics
            progress_callback: Progress callback (stage, progress, message)
            skip_serp_validation: Skip SERP validation step
        
        Returns:
            ClusteringResult with all cluster information
        """
        keyword_data = keyword_data or {}
        
        # Stage 1: Coarse clustering
        if progress_callback:
            progress_callback("clustering", 0.1, "Stage 1: Coarse clustering...")
        
        coarse_result = self.coarse.fit(embeddings)
        
        # Stage 2: Fine-grained sub-clustering
        if progress_callback:
            progress_callback("clustering", 0.25, "Stage 2: Sub-clustering...")
        
        fine_labels, fine_centroids = self._run_fine_clustering(
            embeddings,
            coarse_result,
            progress_callback
        )
        
        # Stage 3: CLUSTER MERGING - Combine similar clusters
        if progress_callback:
            progress_callback("clustering", 0.5, "Stage 3: Merging clusters...")
        
        merged_labels, merged_centroids = self._merge_similar_clusters(
            embeddings,
            fine_labels,
            fine_centroids
        )
        
        # Stage 4: SERP validation (if enabled)
        validations = []
        if self.serp_validator and not skip_serp_validation:
            if progress_callback:
                progress_callback(
                    "serp_validation", 0.0, "Stage 4: SERP validation..."
                )
            
            validations = self._run_serp_validation(
                keywords,
                keyword_data,
                merged_labels,
                progress_callback
            )
        
        # Stage 5: Outlier reassignment
        if progress_callback:
            progress_callback(
                "clustering", 0.75, "Stage 5: Outlier reassignment..."
            )
        
        final_labels, reassignments = self.outlier_handler.reassign_outliers(
            embeddings,
            merged_labels,
            merged_centroids
        )
        
        # Recompute centroids after final labels
        final_centroids = self._compute_all_centroids(embeddings, final_labels)
        
        # Build result
        if progress_callback:
            progress_callback("clustering", 0.95, "Finalizing results...")
        
        result = self._build_result(
            embeddings,
            keywords,
            final_labels,
            final_centroids,
            validations
        )
        
        if progress_callback:
            progress_callback("clustering", 1.0, "Clustering complete!")
        
        return result
    
    def _run_fine_clustering(
        self,
        embeddings: np.ndarray,
        coarse_result: CoarseClusterResult,
        progress_callback: Callable = None
    ) -> tuple:
        """Run fine-grained clustering on each coarse cluster."""
        final_labels = np.full(len(embeddings), -1, dtype=int)
        all_centroids = []
        next_cluster_id = 0
        
        for coarse_id in range(coarse_result.n_clusters):
            indices = coarse_result.get_cluster_indices(coarse_id)
            
            if len(indices) < self.fine.min_cluster_size:
                # Too small for sub-clustering, keep as is
                final_labels[indices] = next_cluster_id
                all_centroids.append(coarse_result.centroids[coarse_id])
                next_cluster_id += 1
                continue
            
            # Run HDBSCAN on this cluster
            cluster_embeddings = embeddings[indices]
            fine_result = self.fine.fit(cluster_embeddings)
            
            # Map fine cluster labels to global labels
            for local_id in range(fine_result.n_clusters):
                local_indices = fine_result.get_cluster_indices(local_id)
                global_indices = indices[local_indices]
                final_labels[global_indices] = next_cluster_id
                
                # Compute centroid for this sub-cluster
                centroid = np.mean(
                    embeddings[global_indices],
                    axis=0
                )
                all_centroids.append(centroid)
                next_cluster_id += 1
            
            # Fine-clustering outliers stay as -1
            if fine_result.n_outliers > 0:
                outlier_global = indices[fine_result.outlier_indices]
                final_labels[outlier_global] = -1
            
            if progress_callback:
                progress = 0.3 + (0.4 * (coarse_id + 1) / coarse_result.n_clusters)
                progress_callback(
                    "clustering",
                    progress,
                    f"Sub-clustering {coarse_id + 1}/{coarse_result.n_clusters}"
                )
        
        centroids = np.array(all_centroids) if all_centroids else np.array([])
        return final_labels, centroids
    
    def _merge_similar_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> tuple:
        """
        Merge clusters with similar centroids to prevent over-fragmentation.
        
        This is critical for grouping semantically related keywords like:
        - "odoo pricing" and "odoo online pricing europe"
        - "api odoo" and "odoo api"
        
        Args:
            embeddings: All keyword embeddings
            labels: Current cluster labels
            centroids: Current cluster centroids
        
        Returns:
            Tuple of (merged_labels, merged_centroids)
        """
        if len(centroids) < 2:
            return labels, centroids
        
        new_labels = labels.copy()
        cluster_ids = np.unique(labels[labels >= 0])
        
        if len(cluster_ids) < 2:
            return labels, centroids
        
        # Compute centroids for current clusters
        current_centroids = {}
        for cid in cluster_ids:
            mask = labels == cid
            current_centroids[int(cid)] = np.mean(embeddings[mask], axis=0)
        
        # Compute pairwise similarities
        centroid_list = list(current_centroids.values())
        centroid_ids = list(current_centroids.keys())
        centroid_matrix = np.array(centroid_list)
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        similarities = cos_sim(centroid_matrix)
        
        # Build merge mapping using Union-Find approach
        merge_map = {cid: cid for cid in centroid_ids}
        
        # Find clusters to merge (similarity > threshold)
        for i in range(len(centroid_ids)):
            for j in range(i + 1, len(centroid_ids)):
                if similarities[i, j] >= self.merge_threshold:
                    # Merge cluster j into cluster i (keep lower id)
                    cid_i = centroid_ids[i]
                    cid_j = centroid_ids[j]
                    
                    # Find roots
                    root_i = self._find_root(merge_map, cid_i)
                    root_j = self._find_root(merge_map, cid_j)
                    
                    if root_i != root_j:
                        # Merge: point j's root to i's root
                        merge_map[root_j] = root_i
        
        # Apply merges to labels
        for cid in centroid_ids:
            root = self._find_root(merge_map, cid)
            if root != cid:
                # Merge this cluster into root
                mask = new_labels == cid
                new_labels[mask] = root
        
        # Renumber clusters to be consecutive
        unique_new = np.unique(new_labels[new_labels >= 0])
        id_mapping = {old: new for new, old in enumerate(unique_new)}
        
        for old_id, new_id in id_mapping.items():
            mask = new_labels == old_id
            new_labels[mask] = new_id
        
        # Recompute centroids
        new_centroids = self._compute_all_centroids(embeddings, new_labels)
        
        return new_labels, new_centroids
    
    def _find_root(self, merge_map: Dict[int, int], cid: int) -> int:
        """Find root of cluster in Union-Find structure."""
        if merge_map[cid] == cid:
            return cid
        # Path compression
        merge_map[cid] = self._find_root(merge_map, merge_map[cid])
        return merge_map[cid]
    
    def _compute_all_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Compute centroids for all clusters."""
        cluster_ids = np.unique(labels[labels >= 0])
        centroids = []
        
        for cid in sorted(cluster_ids):
            mask = labels == cid
            centroid = np.mean(embeddings[mask], axis=0)
            centroids.append(centroid)
        
        return np.array(centroids) if centroids else np.array([])
    
    def _run_serp_validation(
        self,
        keywords: List[str],
        keyword_data: Dict[str, dict],
        labels: np.ndarray,
        progress_callback: Callable = None
    ) -> List[ClusterValidation]:
        """Run SERP validation on clusters."""
        validations = []
        cluster_ids = np.unique(labels[labels >= 0])
        
        for i, cluster_id in enumerate(cluster_ids):
            indices = np.where(labels == cluster_id)[0]
            
            # Build cluster keyword data
            cluster_keywords = [
                {
                    "keyword": keywords[idx],
                    "search_volume": keyword_data.get(
                        keywords[idx], {}
                    ).get("search_volume", 0)
                }
                for idx in indices
            ]
            
            # Validate
            validation = self.serp_validator.validate_cluster(
                cluster_keywords,
                cluster_id=int(cluster_id)
            )
            validations.append(validation)
            
            if progress_callback:
                progress = (i + 1) / len(cluster_ids)
                progress_callback(
                    "serp_validation",
                    progress,
                    f"Validating cluster {i + 1}/{len(cluster_ids)}"
                )
        
        return validations
    
    def _build_result(
        self,
        embeddings: np.ndarray,
        keywords: List[str],
        labels: np.ndarray,
        centroids: np.ndarray,
        validations: List[ClusterValidation]
    ) -> ClusteringResult:
        """Build final clustering result."""
        cluster_ids = np.unique(labels[labels >= 0])
        n_clusters = len(cluster_ids)
        
        # Build cluster info
        clusters = {}
        validation_map = {
            v.cluster_id: v for v in validations
        }
        
        for idx, cluster_id in enumerate(cluster_ids):
            indices = np.where(labels == cluster_id)[0]
            
            centroid = (
                centroids[idx]
                if idx < len(centroids)
                else np.mean(embeddings[indices], axis=0)
            )
            
            clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                keyword_indices=indices,
                keywords=[keywords[i] for i in indices],
                centroid=centroid,
                size=len(indices),
                serp_validation=validation_map.get(cluster_id),
                quality_score=self._calculate_quality(
                    embeddings[indices],
                    centroid,
                    validation_map.get(cluster_id)
                )
            )
        
        # Calculate statistics
        total = len(labels)
        outliers = np.sum(labels == -1)
        clustered = total - outliers
        
        # Average SERP overlap
        overlaps = [
            v.average_overlap for v in validations
            if v.average_overlap > 0
        ]
        avg_overlap = np.mean(overlaps) if overlaps else 0
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            clusters=clusters,
            total_keywords=total,
            clustered_keywords=clustered,
            outlier_keywords=outliers,
            cluster_rate=clustered / total if total > 0 else 0,
            validations=validations,
            avg_serp_overlap=avg_overlap
        )
    
    def _calculate_quality(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray,
        validation: Optional[ClusterValidation]
    ) -> float:
        """Calculate cluster quality score (0-100)."""
        # Semantic coherence (based on distance to centroid)
        if len(embeddings) > 0:
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            avg_distance = np.mean(distances)
            coherence_score = max(0, 30 * np.exp(-avg_distance))
        else:
            coherence_score = 0
        
        # SERP validation score
        if validation and validation.confidence > 0:
            serp_score = validation.average_overlap * 30
        else:
            serp_score = 15  # Neutral if no validation
        
        # Size balance score
        size = len(embeddings)
        if 5 <= size <= 100:
            size_score = 20
        elif size < 5:
            size_score = size * 4
        else:
            size_score = max(10, 20 - (size - 100) * 0.1)
        
        # Intent clarity (based on validation confidence)
        if validation:
            intent_score = validation.confidence * 20
        else:
            intent_score = 10
        
        return min(100, coherence_score + serp_score + size_score + intent_score)