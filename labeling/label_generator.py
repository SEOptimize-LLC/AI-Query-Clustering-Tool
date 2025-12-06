"""
Label generator with intelligent fallback mechanism.
Generates consistent, semantic labels for clusters based on user intent.

Best Practices Applied:
- Focus on User Intent: Labels signal what the user will find
- Be Specific: Avoid broad topics
- Actionable & Descriptive: Include action or solution keywords provide
- Avoid generic labels like "Cluster 1" or "Miscellaneous"
"""
import json
import re
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from collections import Counter

from labeling.intent_classifier import IntentClassifier, IntentResult


@dataclass
class ClusterLabel:
    """Generated label for a cluster."""
    cluster_id: int
    label: str
    confidence: float
    intent: IntentResult
    summary: str = ""
    description: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "confidence": self.confidence,
            "intent": self.intent.to_dict(),
            "summary": self.summary,
            "description": self.description
        }


@dataclass
class LabelingResult:
    """Result of label generation for all clusters."""
    labels: Dict[int, ClusterLabel] = field(default_factory=dict)
    consistency_score: float = 0.0
    
    def get_label(self, cluster_id: int) -> str:
        """Get label for a cluster."""
        if cluster_id in self.labels:
            return self.labels[cluster_id].label
        return f"Cluster {cluster_id}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "labels": {
                cid: lbl.to_dict()
                for cid, lbl in self.labels.items()
            },
            "consistency_score": self.consistency_score
        }


class LabelGenerator:
    """
    Generates consistent cluster labels using two-phase approach
    with intelligent fallback.
    
    Phase 1: Attempt LLM-based labeling
    Phase 2: Smart fallback using keyword analysis if LLM fails
    
    Labels follow SEO best practices:
    - User intent focused (Informational, Transactional, etc.)
    - Specific and descriptive (2-5 words)
    - Action-oriented when appropriate
    """
    
    # Intent prefixes DISABLED per user request
    # Labels should be clean topic names without prefixes
    INTENT_PREFIXES = {
        "informational": ["", "", "", ""],  # No prefix
        "transactional": ["", "", "", ""],  # No prefix
        "commercial": ["", "", "", ""],     # No prefix
        "navigational": ["", "", "", ""],   # No prefix
    }
    
    def __init__(
        self,
        llm_client,
        intent_classifier: IntentClassifier = None
    ):
        """
        Initialize label generator.
        
        Args:
            llm_client: LLM client for generating labels
            intent_classifier: Optional intent classifier
        """
        self.llm_client = llm_client
        self.intent_classifier = intent_classifier or IntentClassifier()
    
    async def generate_labels(
        self,
        clusters: Dict[int, dict],
        progress_callback: Callable = None
    ) -> LabelingResult:
        """
        Generate consistent labels for all clusters.
        
        Args:
            clusters: Dict mapping cluster_id to cluster data
                Each cluster should have:
                - keywords: List of keywords
                - top_keywords: Optional list by search volume
                - serp_data: Optional SERP validation data
            progress_callback: Progress callback (phase, progress, message)
        
        Returns:
            LabelingResult with all labels
        """
        if not clusters:
            return LabelingResult()
        
        # Phase 1: Analyze all clusters
        if progress_callback:
            progress_callback("labeling", 0.1, "Analyzing clusters...")
        
        cluster_analyses = self._analyze_clusters(clusters)
        
        # Phase 2: Try LLM labeling
        if progress_callback:
            progress_callback("labeling", 0.3, "Generating labels with AI...")
        
        llm_labels = await self._generate_llm_labels(cluster_analyses)
        
        # Phase 3: Build final labels with smart fallback
        if progress_callback:
            progress_callback("labeling", 0.8, "Finalizing labels...")
        
        labels = self._build_final_labels(
            cluster_analyses,
            llm_labels,
            clusters
        )
        
        # Calculate consistency score
        consistency = self._calculate_consistency(labels)
        
        if progress_callback:
            progress_callback("labeling", 1.0, "Labeling complete!")
        
        return LabelingResult(
            labels=labels,
            consistency_score=consistency
        )
    
    def _analyze_clusters(
        self,
        clusters: Dict[int, dict]
    ) -> List[dict]:
        """Analyze all clusters for labeling."""
        analyses = []
        
        for cluster_id, data in clusters.items():
            keywords = data.get("keywords", [])
            top_keywords = data.get("top_keywords", keywords[:10])
            
            # Classify intent
            intent = self.intent_classifier.classify_cluster(keywords)
            
            # Extract key themes
            theme = self._extract_theme(keywords, top_keywords)
            
            analyses.append({
                "cluster_id": cluster_id,
                "keywords": keywords,
                "top_keywords": top_keywords[:5],
                "intent": intent,
                "theme": theme,
                "keyword_count": len(keywords)
            })
        
        return analyses
    
    def _extract_theme(
        self,
        keywords: List[str],
        top_keywords: List[str]
    ) -> str:
        """
        Extract main theme from keywords.
        Uses word frequency analysis.
        """
        # Combine all keywords
        all_text = " ".join(keywords[:50])  # Use first 50 for efficiency
        
        # Tokenize and count (simple approach)
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'with', 'you', 'this', 'that',
            'from', 'have', 'your', 'what', 'how', 'why', 'when', 'where',
            'can', 'will', 'all', 'best', 'top', 'get', 'make', 'use',
            'does', 'has', 'was', 'were', 'been', 'being', 'more', 'most'
        }
        words = [w for w in words if w not in stopwords]
        
        # Get most common words
        word_counts = Counter(words)
        common = word_counts.most_common(3)
        
        if common:
            # Build theme from top words
            theme_words = [w[0] for w in common]
            return " ".join(theme_words[:2]).title()
        
        # Fallback: use first top keyword
        if top_keywords:
            first_kw = top_keywords[0] if top_keywords else ""
            # Capitalize words
            return " ".join(w.capitalize() for w in first_kw.split()[:3])
        
        return ""
    
    async def _generate_llm_labels(
        self,
        analyses: List[dict]
    ) -> Dict[int, str]:
        """Try to generate labels using LLM."""
        labels = {}
        
        # Build prompt with all cluster summaries
        prompt = self._build_labeling_prompt(analyses)
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse response
            labels = self._parse_llm_response(response, analyses)
            
        except Exception as e:
            # LLM failed - will use fallback
            print(f"LLM labeling failed: {e}")
        
        return labels
    
    def _build_labeling_prompt(self, analyses: List[dict]) -> str:
        """Build the LLM prompt for labeling."""
        cluster_summaries = []
        
        for a in analyses:
            intent_type = a["intent"].primary_intent.value
            keywords_sample = ", ".join(a["top_keywords"][:5])
            
            cluster_summaries.append(
                f"Cluster {a['cluster_id']} ({a['keyword_count']} keywords):\n"
                f"  Intent: {intent_type}\n"
                f"  Theme: {a['theme']}\n"
                f"  Sample: {keywords_sample}"
            )
        
        prompt = f"""You are an SEO expert creating cluster labels.

TASK: Create a short label (2-4 words) for each keyword cluster.

RULES:
1. Labels must be DESCRIPTIVE topic names (not "Cluster 1")
2. NO prefixes like "Guide:", "Best:", "How-To:" - just the topic
3. Labels should be the main TOPIC/THEME of the keywords
4. Labels must be UNIQUE (no duplicates)
5. Use Title Case

EXAMPLES of good labels:
- "Odoo Pricing" (not "Guide: Odoo Pricing")
- "Running Shoes Reviews" (not "Best: Running Shoes")
- "API Integration" (not "How-To: API Integration")

CLUSTERS:
{chr(10).join(cluster_summaries)}

OUTPUT (JSON only):
{{"labels": [{{"cluster_id": 0, "label": "Topic Name"}}]}}

JSON:"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        analyses: List[dict]
    ) -> Dict[int, str]:
        """Parse LLM response and extract labels."""
        labels = {}
        
        # Try multiple JSON extraction strategies
        json_data = None
        
        # Strategy 1: Direct JSON parse
        try:
            json_data = json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON object in response
        if not json_data:
            try:
                # Find content between first { and last }
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    json_data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON array
        if not json_data:
            try:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    arr = json.loads(json_str)
                    json_data = {"labels": arr}
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Line-by-line extraction
        if not json_data:
            json_data = self._extract_labels_from_text(response, analyses)
        
        # Extract labels from parsed data
        if json_data and "labels" in json_data:
            for item in json_data["labels"]:
                cid = item.get("cluster_id")
                label = item.get("label", "")
                
                # Validate label
                if cid is not None and label and len(label) > 2:
                    # Clean up label
                    label = label.strip().strip('"').strip("'")
                    if not self._is_generic_label(label):
                        labels[cid] = label
        
        return labels
    
    def _extract_labels_from_text(
        self,
        response: str,
        analyses: List[dict]
    ) -> Optional[dict]:
        """Extract labels from non-JSON text response."""
        labels_list = []
        
        # Try to find patterns like "Cluster 0: Label" or "0. Label"
        patterns = [
            r'[Cc]luster\s*(\d+)[:\s]+["\']?([^"\'\n,]+)["\']?',
            r'(\d+)[.)\s]+["\']?([^"\'\n,]+)["\']?',
            r'"cluster_id":\s*(\d+)[^"]*"label":\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                for match in matches:
                    cid = int(match[0])
                    label = match[1].strip()
                    if label and len(label) > 2:
                        labels_list.append({
                            "cluster_id": cid,
                            "label": label
                        })
                break
        
        if labels_list:
            return {"labels": labels_list}
        return None
    
    def _is_generic_label(self, label: str) -> bool:
        """Check if label is too generic."""
        generic_patterns = [
            r'^cluster\s*\d*$',
            r'^cluster\s+of\s+\d+',
            r'^group\s*\d*$',
            r'^category\s*\d*$',
            r'^misc',
            r'^other',
            r'^\d+\s+keywords?',
        ]
        
        label_lower = label.lower().strip()
        for pattern in generic_patterns:
            if re.match(pattern, label_lower):
                return True
        return False
    
    def _build_final_labels(
        self,
        analyses: List[dict],
        llm_labels: Dict[int, str],
        clusters: Dict[int, dict]
    ) -> Dict[int, ClusterLabel]:
        """Build final labels with smart fallback."""
        labels = {}
        used_labels = set()  # Track to avoid duplicates
        
        for analysis in analyses:
            cluster_id = analysis["cluster_id"]
            intent = analysis["intent"]
            
            # Try LLM label first
            label_text = llm_labels.get(cluster_id)
            
            # If no LLM label or it's generic, use smart fallback
            if not label_text or self._is_generic_label(label_text):
                label_text = self._generate_smart_fallback(
                    analysis,
                    used_labels
                )
            
            # Ensure uniqueness
            original_label = label_text
            counter = 1
            while label_text.lower() in used_labels:
                counter += 1
                label_text = f"{original_label} {counter}"
            
            used_labels.add(label_text.lower())
            
            # Determine confidence
            confidence = 0.9 if cluster_id in llm_labels else 0.7
            
            labels[cluster_id] = ClusterLabel(
                cluster_id=cluster_id,
                label=label_text,
                confidence=confidence,
                intent=intent,
                summary=analysis.get("theme", "")
            )
        
        return labels
    
    def _generate_smart_fallback(
        self,
        analysis: dict,
        used_labels: set
    ) -> str:
        """
        Generate a smart fallback label based on:
        1. Intent type + theme
        2. Top keywords analysis
        """
        intent_type = analysis["intent"].primary_intent.value.lower()
        theme = analysis.get("theme", "")
        top_keywords = analysis.get("top_keywords", [])
        keywords = analysis.get("keywords", [])
        
        # Strategy 1: Use theme directly (no prefix)
        if theme and len(theme) > 2:
            label = theme.strip()
            if label.lower() not in used_labels:
                return label
        
        # Strategy 2: Use top keyword directly
        if top_keywords:
            # Find best keyword (not too long, not too short)
            for kw in top_keywords[:5]:
                if isinstance(kw, str) and 3 <= len(kw.split()) <= 5:
                    label = " ".join(
                        w.capitalize() for w in kw.split()[:4]
                    )
                    if label.lower() not in used_labels:
                        return label
        
        # Strategy 3: Extract common phrase from keywords
        common_phrase = self._find_common_phrase(keywords[:20])
        if common_phrase:
            label = " ".join(
                w.capitalize() for w in common_phrase.split()[:4]
            )
            if label.lower() not in used_labels:
                return label
        
        # Strategy 4: First keyword capitalized
        if top_keywords:
            first_kw = top_keywords[0]
            if isinstance(first_kw, str):
                words = first_kw.split()[:4]
                label = " ".join(w.capitalize() for w in words)
                if label.lower() not in used_labels:
                    return label
        
        # Strategy 5: Theme + count (last resort, but still descriptive)
        if theme:
            return f"{theme} Topics"
        
        # Final fallback using first keyword
        if keywords:
            first = keywords[0] if isinstance(keywords[0], str) else ""
            return " ".join(w.capitalize() for w in first.split()[:3])
        
        return f"Topic Group {analysis['cluster_id']}"
    
    def _find_common_phrase(self, keywords: List[str]) -> Optional[str]:
        """Find common 2-3 word phrase across keywords."""
        if not keywords:
            return None
        
        # Extract 2-grams and 3-grams
        ngrams = []
        for kw in keywords:
            if isinstance(kw, str):
                words = kw.lower().split()
                for n in [2, 3]:
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i+n])
                        ngrams.append(ngram)
        
        if not ngrams:
            return None
        
        # Find most common
        ngram_counts = Counter(ngrams)
        common = ngram_counts.most_common(1)
        
        if common and common[0][1] >= 3:  # Appears at least 3 times
            return common[0][0]
        
        return None
    
    def _calculate_consistency(
        self,
        labels: Dict[int, ClusterLabel]
    ) -> float:
        """Calculate label consistency score."""
        if not labels:
            return 0.0
        
        label_texts = [lbl.label for lbl in labels.values()]
        
        # Check for uniqueness
        unique_count = len(set(label_texts))
        uniqueness_score = unique_count / len(label_texts)
        
        # Check for generic labels (should be 0)
        generic_count = sum(
            1 for lbl in label_texts
            if self._is_generic_label(lbl)
        )
        generic_penalty = 1 - (generic_count / len(label_texts))
        
        # Check label length consistency
        lengths = [len(lbl.split()) for lbl in label_texts]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            length_variance = sum(
                (l - avg_len) ** 2 for l in lengths
            ) / len(lengths)
            length_score = max(0, 1 - (length_variance / 10))
        else:
            length_score = 0.5
        
        # Weighted average
        return (
            uniqueness_score * 0.4 +
            generic_penalty * 0.4 +
            length_score * 0.2
        )
    
    async def generate_descriptions(
        self,
        labels: Dict[int, ClusterLabel],
        clusters: Dict[int, dict]
    ) -> Dict[int, ClusterLabel]:
        """
        Generate detailed descriptions for labeled clusters.
        
        Args:
            labels: Existing labels
            clusters: Cluster data with metrics
        
        Returns:
            Updated labels with descriptions
        """
        for cluster_id, label in labels.items():
            cluster_data = clusters.get(cluster_id, {})
            keywords = cluster_data.get("keywords", [])
            
            # Generate simple description without LLM
            intent_type = label.intent.primary_intent.value
            label.description = (
                f"{intent_type.capitalize()} content cluster "
                f"with {len(keywords)} keywords."
            )
        
        return labels