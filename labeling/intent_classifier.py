"""
Intent classification for keywords and clusters.
Uses pattern matching and LLM for accurate classification.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Types of search intent."""
    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: IntentType
    confidence: float
    modifiers: List[str]
    explanation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "primary_intent": self.primary_intent.value,
            "confidence": self.confidence,
            "modifiers": self.modifiers,
            "explanation": self.explanation
        }


class IntentClassifier:
    """
    Classifies search intent using pattern matching and LLM.
    
    Intent types:
    - Informational: User wants to learn
    - Transactional: User wants to buy/act
    - Navigational: User wants specific site
    - Commercial: User is researching purchase
    """
    
    # Pattern definitions for rule-based classification
    INFORMATIONAL_PATTERNS = [
        r'\bhow\s+to\b',
        r'\bwhat\s+is\b',
        r'\bwhat\s+are\b',
        r'\bwhy\s+',
        r'\bwhen\s+',
        r'\bwhere\s+',
        r'\bguide\b',
        r'\btutorial\b',
        r'\bexample[s]?\b',
        r'\bexplain\b',
        r'\bdefinition\b',
        r'\bmeaning\b',
        r'\btips\b',
        r'\bideas\b',
        r'\blearn\b',
    ]
    
    TRANSACTIONAL_PATTERNS = [
        r'\bbuy\b',
        r'\bpurchase\b',
        r'\border\b',
        r'\bdownload\b',
        r'\bfree\s+download\b',
        r'\bget\s+',
        r'\bsubscribe\b',
        r'\bsign\s+up\b',
        r'\bdiscount\b',
        r'\bcoupon\b',
        r'\bdeal[s]?\b',
        r'\bsale\b',
        r'\bprice\b',
        r'\bcheap\b',
        r'\baffordable\b',
    ]
    
    NAVIGATIONAL_PATTERNS = [
        r'\blogin\b',
        r'\blog\s*in\b',
        r'\bsign\s*in\b',
        r'\bwebsite\b',
        r'\bofficial\b',
        r'\bsite\b',
        r'\bapp\b',
        r'\bdownload\s+app\b',
        r'\bcontact\b',
        r'\bsupport\b',
        r'\bhelp\s+center\b',
    ]
    
    COMMERCIAL_PATTERNS = [
        r'\bbest\b',
        r'\btop\s+\d+\b',
        r'\breview[s]?\b',
        r'\bcompare\b',
        r'\bcomparison\b',
        r'\bvs\.?\b',
        r'\bversus\b',
        r'\balternative[s]?\b',
        r'\brecommend',
        r'\brating[s]?\b',
        r'\bpros\s+and\s+cons\b',
    ]
    
    # Modifier patterns
    MODIFIER_PATTERNS = {
        "local": [
            r'\bnear\s+me\b',
            r'\bin\s+[A-Z][a-z]+\b',
            r'\blocal\b',
            r'\b\d{5}\b',  # ZIP code
        ],
        "comparison": [
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bcompare\b',
            r'\bor\b',
            r'\bdifference\b',
        ],
        "question": [
            r'^how\b',
            r'^what\b',
            r'^why\b',
            r'^when\b',
            r'^where\b',
            r'^which\b',
            r'^who\b',
            r'\?$',
        ],
        "branded": [],  # Will be populated dynamically
        "long_tail": [],  # Keywords with 5+ words
    }
    
    def __init__(self, llm_client=None, brand_names: List[str] = None):
        """
        Initialize intent classifier.
        
        Args:
            llm_client: Optional LLM client for advanced classification
            brand_names: List of brand names to detect branded intent
        """
        self.llm_client = llm_client
        self.brand_names = brand_names or []
        
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.info_re = [
            re.compile(p, re.IGNORECASE)
            for p in self.INFORMATIONAL_PATTERNS
        ]
        self.trans_re = [
            re.compile(p, re.IGNORECASE)
            for p in self.TRANSACTIONAL_PATTERNS
        ]
        self.nav_re = [
            re.compile(p, re.IGNORECASE)
            for p in self.NAVIGATIONAL_PATTERNS
        ]
        self.comm_re = [
            re.compile(p, re.IGNORECASE)
            for p in self.COMMERCIAL_PATTERNS
        ]
        
        # Modifier patterns
        self.modifier_re = {}
        for mod, patterns in self.MODIFIER_PATTERNS.items():
            if patterns:
                self.modifier_re[mod] = [
                    re.compile(p, re.IGNORECASE)
                    for p in patterns
                ]
    
    def classify_keyword(
        self,
        keyword: str,
        use_llm: bool = False
    ) -> IntentResult:
        """
        Classify intent for a single keyword.
        
        Args:
            keyword: Keyword to classify
            use_llm: Whether to use LLM for classification
        
        Returns:
            IntentResult with classification
        """
        # Rule-based classification
        scores = self._calculate_scores(keyword)
        modifiers = self._detect_modifiers(keyword)
        
        # Determine primary intent
        intent, confidence = self._determine_intent(scores)
        
        return IntentResult(
            primary_intent=intent,
            confidence=confidence,
            modifiers=modifiers
        )
    
    def classify_keywords(
        self,
        keywords: List[str],
        use_llm: bool = False
    ) -> Dict[str, IntentResult]:
        """
        Classify intent for multiple keywords.
        
        Args:
            keywords: List of keywords
            use_llm: Whether to use LLM
        
        Returns:
            Dict mapping keyword to IntentResult
        """
        results = {}
        for keyword in keywords:
            results[keyword] = self.classify_keyword(keyword, use_llm)
        return results
    
    def classify_cluster(
        self,
        keywords: List[str]
    ) -> IntentResult:
        """
        Determine overall intent for a cluster of keywords.
        
        Args:
            keywords: Keywords in the cluster
        
        Returns:
            IntentResult for the cluster
        """
        # Classify all keywords
        results = self.classify_keywords(keywords)
        
        # Aggregate intents
        intent_counts = {}
        all_modifiers = set()
        total_confidence = 0
        
        for kw, result in results.items():
            intent = result.primary_intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            all_modifiers.update(result.modifiers)
            total_confidence += result.confidence
        
        # Find dominant intent
        if intent_counts:
            dominant = max(intent_counts.items(), key=lambda x: x[1])
            primary_intent = dominant[0]
            vote_ratio = dominant[1] / len(keywords)
            avg_confidence = total_confidence / len(keywords)
            
            # Confidence is combo of vote ratio and avg classification confidence
            confidence = (vote_ratio + avg_confidence) / 2
        else:
            primary_intent = IntentType.UNKNOWN
            confidence = 0.0
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            modifiers=list(all_modifiers),
            explanation=f"Based on {len(keywords)} keywords"
        )
    
    def _calculate_scores(self, keyword: str) -> Dict[IntentType, float]:
        """Calculate intent scores for a keyword."""
        scores = {
            IntentType.INFORMATIONAL: 0.0,
            IntentType.TRANSACTIONAL: 0.0,
            IntentType.NAVIGATIONAL: 0.0,
            IntentType.COMMERCIAL: 0.0,
        }
        
        # Check each pattern category
        for pattern in self.info_re:
            if pattern.search(keyword):
                scores[IntentType.INFORMATIONAL] += 1.0
        
        for pattern in self.trans_re:
            if pattern.search(keyword):
                scores[IntentType.TRANSACTIONAL] += 1.0
        
        for pattern in self.nav_re:
            if pattern.search(keyword):
                scores[IntentType.NAVIGATIONAL] += 1.0
        
        for pattern in self.comm_re:
            if pattern.search(keyword):
                scores[IntentType.COMMERCIAL] += 1.0
        
        # Check for brand names
        kw_lower = keyword.lower()
        for brand in self.brand_names:
            if brand.lower() in kw_lower:
                scores[IntentType.NAVIGATIONAL] += 0.5
                break
        
        return scores
    
    def _determine_intent(
        self,
        scores: Dict[IntentType, float]
    ) -> Tuple[IntentType, float]:
        """Determine primary intent from scores."""
        total = sum(scores.values())
        
        if total == 0:
            # No strong signals, default to informational
            return IntentType.INFORMATIONAL, 0.3
        
        # Find highest score
        max_intent = max(scores.items(), key=lambda x: x[1])
        confidence = max_intent[1] / total
        
        # Boost confidence if clear winner
        if max_intent[1] >= 2:
            confidence = min(0.95, confidence + 0.2)
        
        return max_intent[0], confidence
    
    def _detect_modifiers(self, keyword: str) -> List[str]:
        """Detect intent modifiers in keyword."""
        modifiers = []
        
        for mod, patterns in self.modifier_re.items():
            for pattern in patterns:
                if pattern.search(keyword):
                    modifiers.append(mod)
                    break
        
        # Check long-tail
        if len(keyword.split()) >= 5:
            modifiers.append("long_tail")
        
        # Check branded
        kw_lower = keyword.lower()
        for brand in self.brand_names:
            if brand.lower() in kw_lower:
                modifiers.append("branded")
                break
        
        return list(set(modifiers))
    
    async def classify_with_llm(
        self,
        keywords: List[str]
    ) -> IntentResult:
        """
        Use LLM for advanced intent classification.
        
        Args:
            keywords: Keywords to classify
        
        Returns:
            IntentResult from LLM
        """
        if not self.llm_client:
            return self.classify_cluster(keywords)
        
        from labeling.prompts import format_intent_prompt
        
        prompt = format_intent_prompt(keywords)
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            intent_str = result.get("primary_intent", "UNKNOWN")
            intent = IntentType(intent_str.lower())
            
            return IntentResult(
                primary_intent=intent,
                confidence=result.get("confidence", 0.8),
                modifiers=result.get("modifiers", []),
                explanation=result.get("explanation", "")
            )
        except Exception:
            # Fall back to rule-based
            return self.classify_cluster(keywords)