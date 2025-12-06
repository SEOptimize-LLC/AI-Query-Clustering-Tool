"""
Keyword validation utilities.
"""
from typing import List, Tuple
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of keyword validation."""
    
    valid_keywords: List[str]
    invalid_keywords: List[Tuple[str, str]]  # (keyword, reason)
    warnings: List[str]
    
    @property
    def valid_count(self) -> int:
        return len(self.valid_keywords)
    
    @property
    def invalid_count(self) -> int:
        return len(self.invalid_keywords)
    
    @property
    def total_count(self) -> int:
        return self.valid_count + self.invalid_count
    
    @property
    def validity_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.valid_count / self.total_count


class KeywordValidator:
    """
    Validates keywords for clustering.
    Removes invalid entries and provides warnings.
    """
    
    # Validation settings
    MIN_LENGTH = 1
    MAX_LENGTH = 200  # Most search engines truncate at ~200 chars
    
    # Patterns to reject
    INVALID_PATTERNS = [
        r'^https?://',  # URLs
        r'^\d+$',       # Pure numbers
        r'^[^\w]+$',    # No alphanumeric characters
    ]
    
    def __init__(
        self,
        min_length: int = None,
        max_length: int = None,
        allow_urls: bool = False,
        allow_numbers: bool = False
    ):
        """
        Initialize validator.
        
        Args:
            min_length: Minimum keyword length
            max_length: Maximum keyword length
            allow_urls: Allow URL-like keywords
            allow_numbers: Allow pure numeric keywords
        """
        self.min_length = min_length or self.MIN_LENGTH
        self.max_length = max_length or self.MAX_LENGTH
        self.allow_urls = allow_urls
        self.allow_numbers = allow_numbers
    
    def validate(self, keywords: List[str]) -> ValidationResult:
        """
        Validate a list of keywords.
        
        Args:
            keywords: List of keyword strings
        
        Returns:
            ValidationResult with valid/invalid keywords and warnings
        """
        valid = []
        invalid = []
        warnings = []
        
        for kw in keywords:
            is_valid, reason = self._validate_keyword(kw)
            
            if is_valid:
                valid.append(kw)
            else:
                invalid.append((kw, reason))
        
        # Generate warnings
        if len(invalid) > 0:
            warnings.append(
                f"{len(invalid)} keywords were filtered out"
            )
        
        # Check for potential issues
        avg_length = (
            sum(len(k) for k in valid) / len(valid)
            if valid else 0
        )
        if avg_length > 50:
            warnings.append(
                "Average keyword length is high (>50 chars). "
                "Consider checking for data issues."
            )
        
        # Check for duplicate-looking entries
        lowercase_set = set()
        potential_duplicates = 0
        for kw in valid:
            lower = kw.lower()
            if lower in lowercase_set:
                potential_duplicates += 1
            lowercase_set.add(lower)
        
        if potential_duplicates > 0:
            warnings.append(
                f"{potential_duplicates} potential case-insensitive "
                "duplicates detected. Consider deduplication."
            )
        
        return ValidationResult(
            valid_keywords=valid,
            invalid_keywords=invalid,
            warnings=warnings
        )
    
    def validate_batch(
        self,
        keywords_data: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """
        Validate a list of keyword dictionaries.
        
        Args:
            keywords_data: List of dicts with 'keyword' key
        
        Returns:
            Tuple of (valid_keywords, invalid_keywords) as dicts
        """
        valid = []
        invalid = []
        
        for kw_dict in keywords_data:
            keyword = kw_dict.get("keyword", "")
            is_valid, reason = self._validate_keyword(keyword)
            
            if is_valid:
                valid.append(kw_dict)
            else:
                invalid.append(kw_dict)
        
        return valid, invalid
    
    def _validate_keyword(self, keyword: str) -> Tuple[bool, str]:
        """
        Validate a single keyword.
        
        Args:
            keyword: Keyword string
        
        Returns:
            Tuple of (is_valid, reason if invalid)
        """
        # Check for empty/whitespace
        if not keyword or not keyword.strip():
            return False, "Empty or whitespace-only"
        
        keyword = keyword.strip()
        
        # Check length
        if len(keyword) < self.min_length:
            return False, f"Too short (min {self.min_length} chars)"
        
        if len(keyword) > self.max_length:
            return False, f"Too long (max {self.max_length} chars)"
        
        # Check patterns
        if not self.allow_urls:
            if re.match(r'^https?://', keyword, re.IGNORECASE):
                return False, "URL detected"
        
        if not self.allow_numbers:
            if re.match(r'^\d+$', keyword):
                return False, "Pure number"
        
        # Check for no alphanumeric content
        if not re.search(r'\w', keyword):
            return False, "No alphanumeric characters"
        
        return True, ""
    
    def get_stats(self, result: ValidationResult) -> dict:
        """
        Get validation statistics.
        
        Args:
            result: ValidationResult object
        
        Returns:
            Statistics dictionary
        """
        # Categorize invalid keywords by reason
        reason_counts = {}
        for _, reason in result.invalid_keywords:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            "total_input": result.total_count,
            "valid": result.valid_count,
            "invalid": result.invalid_count,
            "validity_rate": f"{result.validity_rate:.1%}",
            "invalid_reasons": reason_counts,
            "warnings": result.warnings
        }