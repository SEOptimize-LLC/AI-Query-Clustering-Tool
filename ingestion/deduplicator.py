"""
Keyword deduplication utilities.
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class DeduplicationResult:
    """Result of keyword deduplication."""
    
    unique_keywords: List[str]
    duplicates: Dict[str, List[str]]  # canonical -> duplicates
    metrics_merged: Dict[str, dict]  # canonical -> merged metrics
    
    @property
    def unique_count(self) -> int:
        return len(self.unique_keywords)
    
    @property
    def duplicate_count(self) -> int:
        return sum(len(dups) for dups in self.duplicates.values())
    
    @property
    def original_count(self) -> int:
        return self.unique_count + self.duplicate_count
    
    @property
    def reduction_rate(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.duplicate_count / self.original_count


class KeywordDeduplicator:
    """
    Deduplicates keywords using various strategies.
    """
    
    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
        strip_special_chars: bool = False
    ):
        """
        Initialize deduplicator.
        
        Args:
            case_sensitive: Keep case distinctions
            normalize_whitespace: Collapse multiple spaces
            strip_special_chars: Remove special characters
        """
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self.strip_special_chars = strip_special_chars
    
    def deduplicate(
        self,
        keywords: List[str],
        metrics: Dict[str, dict] = None
    ) -> DeduplicationResult:
        """
        Deduplicate keywords.
        
        When duplicates have metrics, keeps the highest search volume.
        
        Args:
            keywords: List of keywords
            metrics: Optional dict mapping keyword -> metrics
        
        Returns:
            DeduplicationResult with unique keywords and duplicate info
        """
        metrics = metrics or {}
        
        # Track canonical form -> original forms
        canonical_map: Dict[str, List[str]] = {}
        # Track canonical form -> best original (by search volume)
        best_originals: Dict[str, str] = {}
        # Track canonical form -> merged metrics
        merged_metrics: Dict[str, dict] = {}
        
        for kw in keywords:
            canonical = self._canonicalize(kw)
            
            if canonical not in canonical_map:
                canonical_map[canonical] = []
                best_originals[canonical] = kw
                merged_metrics[canonical] = metrics.get(kw, {})
            else:
                # Track as duplicate
                canonical_map[canonical].append(kw)
                
                # Keep the one with higher search volume
                current_vol = merged_metrics[canonical].get("search_volume", 0)
                new_vol = metrics.get(kw, {}).get("search_volume", 0)
                
                if new_vol > current_vol:
                    best_originals[canonical] = kw
                    # Merge metrics, keeping higher values
                    merged_metrics[canonical] = self._merge_metrics(
                        merged_metrics[canonical],
                        metrics.get(kw, {})
                    )
        
        # Build results
        unique_keywords = [
            best_originals[canonical]
            for canonical in canonical_map.keys()
        ]
        
        duplicates = {
            best_originals[canonical]: variants
            for canonical, variants in canonical_map.items()
            if variants  # Only include if there were duplicates
        }
        
        final_metrics = {
            best_originals[canonical]: merged_metrics[canonical]
            for canonical in canonical_map.keys()
            if merged_metrics[canonical]
        }
        
        return DeduplicationResult(
            unique_keywords=unique_keywords,
            duplicates=duplicates,
            metrics_merged=final_metrics
        )
    
    def _canonicalize(self, keyword: str) -> str:
        """
        Convert keyword to canonical form for comparison.
        
        Args:
            keyword: Original keyword
        
        Returns:
            Canonical form
        """
        result = keyword.strip()
        
        if not self.case_sensitive:
            result = result.lower()
        
        if self.normalize_whitespace:
            result = re.sub(r'\s+', ' ', result)
        
        if self.strip_special_chars:
            result = re.sub(r'[^\w\s]', '', result)
        
        return result
    
    def _merge_metrics(
        self,
        existing: dict,
        new: dict
    ) -> dict:
        """
        Merge metrics from duplicate keywords.
        
        Strategy:
        - search_volume: Sum (total opportunity)
        - keyword_difficulty: Average
        - cpc: Maximum (best opportunity)
        
        Args:
            existing: Existing metrics
            new: New metrics to merge
        
        Returns:
            Merged metrics
        """
        merged = dict(existing)
        
        # Sum search volumes
        if "search_volume" in new:
            merged["search_volume"] = (
                merged.get("search_volume", 0) + new["search_volume"]
            )
        
        # Average keyword difficulty
        if "keyword_difficulty" in new:
            existing_kd = merged.get("keyword_difficulty", 0)
            new_kd = new["keyword_difficulty"]
            if existing_kd > 0 and new_kd > 0:
                merged["keyword_difficulty"] = (existing_kd + new_kd) / 2
            else:
                merged["keyword_difficulty"] = new_kd or existing_kd
        
        # Max CPC
        if "cpc" in new:
            merged["cpc"] = max(merged.get("cpc", 0), new["cpc"])
        
        return merged
    
    def get_stats(self, result: DeduplicationResult) -> dict:
        """
        Get deduplication statistics.
        
        Args:
            result: DeduplicationResult object
        
        Returns:
            Statistics dictionary
        """
        return {
            "original_count": result.original_count,
            "unique_count": result.unique_count,
            "duplicate_count": result.duplicate_count,
            "reduction_rate": f"{result.reduction_rate:.1%}",
            "top_duplicates": self._get_top_duplicates(result, 5)
        }
    
    def _get_top_duplicates(
        self,
        result: DeduplicationResult,
        n: int = 5
    ) -> List[dict]:
        """
        Get keywords with most duplicates.
        
        Args:
            result: DeduplicationResult
            n: Number of top entries
        
        Returns:
            List of dicts with keyword and duplicate count
        """
        sorted_dups = sorted(
            result.duplicates.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:n]
        
        return [
            {
                "keyword": kw,
                "duplicate_count": len(dups),
                "duplicates": dups[:3]  # Show first 3 duplicates
            }
            for kw, dups in sorted_dups
        ]