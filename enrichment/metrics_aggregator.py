"""
Metrics aggregation utilities.
"""
from typing import List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class ClusterMetrics:
    """Aggregated metrics for a cluster."""
    
    total_search_volume: int = 0
    avg_search_volume: float = 0.0
    max_search_volume: int = 0
    min_search_volume: int = 0
    
    avg_keyword_difficulty: float = 0.0
    max_keyword_difficulty: float = 0.0
    min_keyword_difficulty: float = 0.0
    
    avg_cpc: float = 0.0
    total_cpc_opportunity: float = 0.0
    
    keyword_count: int = 0


class MetricsAggregator:
    """
    Aggregates metrics for clusters.
    """
    
    def aggregate_cluster(
        self,
        keywords: List[Dict]
    ) -> ClusterMetrics:
        """
        Aggregate metrics for a cluster of keywords.
        
        Args:
            keywords: List of keyword dicts with metrics
                Each dict should have: keyword, search_volume, 
                keyword_difficulty, cpc
        
        Returns:
            ClusterMetrics with aggregated values
        """
        if not keywords:
            return ClusterMetrics()
        
        volumes = [
            kw.get("search_volume", 0) or 0
            for kw in keywords
        ]
        difficulties = [
            kw.get("keyword_difficulty", 0) or 0
            for kw in keywords
        ]
        cpcs = [
            kw.get("cpc", 0) or 0
            for kw in keywords
        ]
        
        return ClusterMetrics(
            total_search_volume=sum(volumes),
            avg_search_volume=np.mean(volumes) if volumes else 0,
            max_search_volume=max(volumes) if volumes else 0,
            min_search_volume=min(volumes) if volumes else 0,
            
            avg_keyword_difficulty=np.mean(difficulties) if difficulties else 0,
            max_keyword_difficulty=max(difficulties) if difficulties else 0,
            min_keyword_difficulty=min(difficulties) if difficulties else 0,
            
            avg_cpc=np.mean(cpcs) if cpcs else 0,
            total_cpc_opportunity=sum(
                v * c for v, c in zip(volumes, cpcs)
            ),
            
            keyword_count=len(keywords)
        )
    
    def aggregate_all_clusters(
        self,
        clusters: Dict[int, List[Dict]]
    ) -> Dict[int, ClusterMetrics]:
        """
        Aggregate metrics for multiple clusters.
        
        Args:
            clusters: Dict mapping cluster_id -> list of keyword dicts
        
        Returns:
            Dict mapping cluster_id -> ClusterMetrics
        """
        return {
            cluster_id: self.aggregate_cluster(keywords)
            for cluster_id, keywords in clusters.items()
        }
    
    def get_summary_statistics(
        self,
        cluster_metrics: Dict[int, ClusterMetrics]
    ) -> dict:
        """
        Get summary statistics across all clusters.
        
        Args:
            cluster_metrics: Dict of cluster_id -> ClusterMetrics
        
        Returns:
            Summary statistics dict
        """
        if not cluster_metrics:
            return {
                "total_clusters": 0,
                "total_keywords": 0,
                "total_search_volume": 0,
                "avg_cluster_size": 0,
                "avg_keyword_difficulty": 0
            }
        
        metrics = list(cluster_metrics.values())
        
        total_volume = sum(m.total_search_volume for m in metrics)
        total_keywords = sum(m.keyword_count for m in metrics)
        
        # Weighted average KD (by keyword count)
        weighted_kd_sum = sum(
            m.avg_keyword_difficulty * m.keyword_count
            for m in metrics
        )
        avg_kd = weighted_kd_sum / total_keywords if total_keywords else 0
        
        return {
            "total_clusters": len(metrics),
            "total_keywords": total_keywords,
            "total_search_volume": total_volume,
            "avg_cluster_size": total_keywords / len(metrics),
            "avg_keyword_difficulty": avg_kd,
            "largest_cluster_volume": max(
                m.total_search_volume for m in metrics
            ),
            "smallest_cluster_volume": min(
                m.total_search_volume for m in metrics
            ),
            "total_cpc_opportunity": sum(
                m.total_cpc_opportunity for m in metrics
            )
        }
    
    def rank_clusters(
        self,
        cluster_metrics: Dict[int, ClusterMetrics],
        by: str = "opportunity"
    ) -> List[tuple]:
        """
        Rank clusters by specified criteria.
        
        Args:
            cluster_metrics: Dict of cluster_id -> ClusterMetrics
            by: Ranking criteria
                - "volume": Total search volume
                - "difficulty": Lowest avg difficulty first
                - "opportunity": Volume / Difficulty ratio
                - "size": Keyword count
        
        Returns:
            List of (cluster_id, score) tuples, sorted descending
        """
        rankings = []
        
        for cluster_id, metrics in cluster_metrics.items():
            if by == "volume":
                score = metrics.total_search_volume
            elif by == "difficulty":
                # Lower difficulty = higher score
                score = 100 - metrics.avg_keyword_difficulty
            elif by == "opportunity":
                # Volume / difficulty ratio (avoid division by zero)
                kd = max(1, metrics.avg_keyword_difficulty)
                score = metrics.total_search_volume / kd
            elif by == "size":
                score = metrics.keyword_count
            else:
                score = metrics.total_search_volume
            
            rankings.append((cluster_id, score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def calculate_priority_score(
        self,
        metrics: ClusterMetrics,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate a priority score for a cluster.
        
        Args:
            metrics: ClusterMetrics instance
            weights: Optional custom weights for factors
                Default: volume=0.4, difficulty=0.3, size=0.3
        
        Returns:
            Priority score (0-100)
        """
        weights = weights or {
            "volume": 0.4,
            "difficulty": 0.3,
            "size": 0.3
        }
        
        # Normalize factors to 0-100 scale
        # Volume: log scale for better distribution
        volume_score = min(100, np.log1p(metrics.total_search_volume) * 10)
        
        # Difficulty: inverse (lower is better)
        difficulty_score = 100 - metrics.avg_keyword_difficulty
        
        # Size: scale based on reasonable cluster size
        size_score = min(100, metrics.keyword_count * 2)
        
        # Calculate weighted score
        priority = (
            volume_score * weights["volume"] +
            difficulty_score * weights["difficulty"] +
            size_score * weights["size"]
        )
        
        return round(priority, 1)