"""
SERP-based cluster validation.
Uses SERP overlap to validate cluster coherence.
"""
import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass

from enrichment.serper_client import SerperClient, SERPResult


@dataclass
class ClusterValidation:
    """Validation result for a cluster."""
    
    cluster_id: int
    keywords_checked: List[str]
    serp_data: Dict[str, SERPResult]
    url_overlap_matrix: np.ndarray
    average_overlap: float
    common_urls: List[str]
    is_coherent: bool
    confidence: float


class SERPValidator:
    """
    Validates cluster coherence using SERP overlap.
    If keywords share ranking URLs, they likely belong together.
    """
    
    def __init__(
        self,
        serper_client: SerperClient,
        overlap_threshold: float = 0.4,
        validation_count: int = 10
    ):
        """
        Initialize SERP validator.
        
        Args:
            serper_client: Serper.dev client
            overlap_threshold: Minimum average overlap for coherence
            validation_count: Number of top keywords to validate per cluster
        """
        self.serper = serper_client
        self.overlap_threshold = overlap_threshold
        self.validation_count = validation_count
    
    def validate_cluster(
        self,
        cluster_keywords: List[Dict],
        cluster_id: int = 0
    ) -> ClusterValidation:
        """
        Validate cluster using top keywords by search volume.
        
        Args:
            cluster_keywords: List of dicts with keyword and search_volume
            cluster_id: Cluster identifier
        
        Returns:
            ClusterValidation with coherence assessment
        """
        # Sort by search volume and take top N
        sorted_keywords = sorted(
            cluster_keywords,
            key=lambda x: x.get("search_volume", 0),
            reverse=True
        )[:self.validation_count]
        
        if len(sorted_keywords) < 2:
            # Can't validate with < 2 keywords
            return ClusterValidation(
                cluster_id=cluster_id,
                keywords_checked=[
                    kw["keyword"] for kw in sorted_keywords
                ],
                serp_data={},
                url_overlap_matrix=np.array([]),
                average_overlap=1.0,
                common_urls=[],
                is_coherent=True,
                confidence=0.0
            )
        
        # Fetch SERP data
        queries = [kw["keyword"] for kw in sorted_keywords]
        serp_results = self.serper.batch_search_sync(queries)
        
        # Calculate URL overlap matrix
        overlap_matrix = self._calculate_overlap_matrix(serp_results)
        avg_overlap = self._calculate_average_overlap(overlap_matrix)
        
        # Find commonly ranking URLs
        common_urls = self._find_common_urls(serp_results)
        
        # Calculate confidence based on data availability
        data_coverage = sum(
            1 for r in serp_results.values() if r.urls
        ) / len(queries)
        confidence = data_coverage * min(1.0, len(sorted_keywords) / 5)
        
        return ClusterValidation(
            cluster_id=cluster_id,
            keywords_checked=queries,
            serp_data=serp_results,
            url_overlap_matrix=overlap_matrix,
            average_overlap=avg_overlap,
            common_urls=common_urls,
            is_coherent=avg_overlap >= self.overlap_threshold,
            confidence=confidence
        )
    
    def _calculate_overlap_matrix(
        self,
        serp_results: Dict[str, SERPResult]
    ) -> np.ndarray:
        """
        Calculate pairwise URL overlap between keywords.
        
        Args:
            serp_results: Dict of keyword -> SERPResult
        
        Returns:
            Square matrix of overlap scores (0-1)
        """
        keywords = list(serp_results.keys())
        n = len(keywords)
        matrix = np.zeros((n, n))
        
        for i, kw1 in enumerate(keywords):
            urls1 = set(serp_results[kw1].urls[:10])  # Top 10 URLs
            
            for j, kw2 in enumerate(keywords):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                
                urls2 = set(serp_results[kw2].urls[:10])
                
                if not urls1 or not urls2:
                    matrix[i, j] = 0.0
                else:
                    # Jaccard similarity
                    intersection = len(urls1 & urls2)
                    union = len(urls1 | urls2)
                    matrix[i, j] = intersection / union if union > 0 else 0
        
        return matrix
    
    def _calculate_average_overlap(
        self,
        overlap_matrix: np.ndarray
    ) -> float:
        """
        Calculate average pairwise overlap.
        
        Args:
            overlap_matrix: Overlap matrix
        
        Returns:
            Average overlap score
        """
        if overlap_matrix.size == 0:
            return 0.0
        
        n = len(overlap_matrix)
        if n < 2:
            return 1.0
        
        # Get upper triangle (excluding diagonal)
        upper_tri = overlap_matrix[np.triu_indices(n, k=1)]
        
        return float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    
    def _find_common_urls(
        self,
        serp_results: Dict[str, SERPResult]
    ) -> List[str]:
        """
        Find URLs that appear in multiple keyword SERPs.
        
        Args:
            serp_results: Dict of keyword -> SERPResult
        
        Returns:
            List of commonly ranking URLs
        """
        url_counts: Dict[str, int] = {}
        
        for result in serp_results.values():
            for url in result.urls[:10]:
                url_counts[url] = url_counts.get(url, 0) + 1
        
        # URLs appearing in 30%+ of keywords
        min_appearances = max(2, len(serp_results) * 0.3)
        common = [
            url for url, count in url_counts.items()
            if count >= min_appearances
        ]
        
        # Sort by frequency
        common.sort(key=lambda u: url_counts[u], reverse=True)
        
        return common[:10]  # Top 10 common URLs
    
    def suggest_cluster_split(
        self,
        validation: ClusterValidation
    ) -> List[List[str]]:
        """
        Suggest how to split an incoherent cluster.
        
        Args:
            validation: ClusterValidation result
        
        Returns:
            List of keyword groups (suggested sub-clusters)
        """
        if validation.is_coherent or len(validation.keywords_checked) < 3:
            return [validation.keywords_checked]
        
        # Use overlap matrix to identify natural groupings
        overlap = validation.url_overlap_matrix
        keywords = validation.keywords_checked
        n = len(keywords)
        
        # Simple greedy grouping based on overlap
        groups = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            group = [keywords[i]]
            assigned.add(i)
            
            # Add keywords with high overlap to this group
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                if overlap[i, j] >= self.overlap_threshold:
                    group.append(keywords[j])
                    assigned.add(j)
            
            groups.append(group)
        
        return groups