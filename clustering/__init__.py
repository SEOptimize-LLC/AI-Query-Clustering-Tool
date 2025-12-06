"""
Clustering module for AI Keyword Clustering.

Multi-stage pipeline:
1. Coarse clustering (MiniBatch K-Means)
2. Fine-grained sub-clustering (HDBSCAN)
3. SERP validation
4. Outlier reassignment
"""
from clustering.coarse_clusterer import CoarseClusterer, CoarseClusterResult
from clustering.fine_clusterer import FineClusterer, FineClusterResult
from clustering.serp_validator import (
    SERPValidator,
    ClusterValidation
)
from clustering.outlier_handler import OutlierHandler
from clustering.pipeline import (
    ClusteringPipeline,
    ClusteringResult,
    ClusterInfo
)

__all__ = [
    # Pipeline
    "ClusteringPipeline",
    "ClusteringResult",
    "ClusterInfo",
    # Coarse clustering
    "CoarseClusterer",
    "CoarseClusterResult",
    # Fine clustering
    "FineClusterer",
    "FineClusterResult",
    # SERP validation
    "SERPValidator",
    "ClusterValidation",
    # Outlier handling
    "OutlierHandler",
]