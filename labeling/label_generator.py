"""
Label generator with two-phase consistency mechanism.
Generates consistent, non-overlapping labels for clusters.
"""
import json
import asyncio
from typing import List, Dict, Callable
from dataclasses import dataclass, field

from labeling.intent_classifier import IntentClassifier, IntentResult
from labeling.prompts import (
    format_cluster_summary_prompt,
    format_consistent_labeling_prompt,
    format_description_prompt
)


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
    Generates consistent cluster labels using two-phase approach.
    
    Phase 1: Generate summaries for each cluster independently
    Phase 2: Generate all labels in single LLM call with global context
    
    This ensures:
    - Labels are mutually exclusive
    - Consistent naming conventions
    - No orphan clusters without proper labels
    """
    
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
        
        # Phase 1: Generate summaries
        if progress_callback:
            progress_callback("labeling", 0.1, "Phase 1: Generating summaries...")
        
        summaries = await self._generate_summaries(
            clusters,
            progress_callback
        )
        
        # Phase 2: Generate consistent labels
        if progress_callback:
            progress_callback("labeling", 0.6, "Phase 2: Generating labels...")
        
        labels = await self._generate_consistent_labels(
            summaries,
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
    
    async def _generate_summaries(
        self,
        clusters: Dict[int, dict],
        progress_callback: Callable = None
    ) -> List[dict]:
        """Generate summaries for all clusters."""
        summaries = []
        total = len(clusters)
        
        for i, (cluster_id, data) in enumerate(clusters.items()):
            keywords = data.get("keywords", [])
            top_keywords = data.get("top_keywords", keywords[:10])
            serp_data = data.get("serp_data")
            
            # Classify intent
            intent = self.intent_classifier.classify_cluster(keywords)
            
            # Generate summary prompt
            prompt = format_cluster_summary_prompt(
                keywords=keywords,
                intent_type=intent.primary_intent.value,
                top_keywords=top_keywords,
                serp_data=serp_data
            )
            
            # Get LLM summary
            try:
                summary = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=200
                )
            except Exception as e:
                summary = f"Cluster of {len(keywords)} keywords"
            
            summaries.append({
                "cluster_id": cluster_id,
                "summary": summary.strip(),
                "intent": intent,
                "keyword_count": len(keywords)
            })
            
            if progress_callback:
                progress = 0.1 + (0.5 * (i + 1) / total)
                progress_callback(
                    "labeling",
                    progress,
                    f"Summarized {i + 1}/{total} clusters"
                )
        
        return summaries
    
    async def _generate_consistent_labels(
        self,
        summaries: List[dict],
        clusters: Dict[int, dict]
    ) -> Dict[int, ClusterLabel]:
        """Generate consistent labels using global context."""
        # Build the consistent labeling prompt
        prompt = format_consistent_labeling_prompt(
            cluster_summaries=summaries,
            n_clusters=len(summaries)
        )
        
        # Single LLM call for all labels
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000
            )
            
            # Parse JSON response
            labels_data = self._parse_labels_response(response)
        except Exception as e:
            # Fallback: generate simple labels
            labels_data = self._generate_fallback_labels(summaries)
        
        # Build ClusterLabel objects
        labels = {}
        summary_map = {s["cluster_id"]: s for s in summaries}
        
        for label_info in labels_data:
            cluster_id = label_info["cluster_id"]
            summary_data = summary_map.get(cluster_id, {})
            
            labels[cluster_id] = ClusterLabel(
                cluster_id=cluster_id,
                label=label_info["label"],
                confidence=label_info.get("confidence", 0.8),
                intent=summary_data.get("intent", IntentResult(
                    primary_intent=IntentClassifier().classify_keyword(
                        label_info["label"]
                    ).primary_intent,
                    confidence=0.5,
                    modifiers=[]
                )),
                summary=summary_data.get("summary", "")
            )
        
        return labels
    
    def _parse_labels_response(self, response: str) -> List[dict]:
        """Parse LLM response to extract labels."""
        # Try to extract JSON
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                if "labels" in data:
                    return data["labels"]
        except json.JSONDecodeError:
            pass
        
        # Try array format
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _generate_fallback_labels(
        self,
        summaries: List[dict]
    ) -> List[dict]:
        """Generate fallback labels when LLM fails."""
        labels = []
        
        for summary in summaries:
            cluster_id = summary["cluster_id"]
            text = summary.get("summary", "")
            
            # Extract key phrase from summary
            words = text.split()[:3] if text else []
            label = " ".join(words).title() if words else f"Cluster {cluster_id}"
            
            # Clean up label
            label = label.replace(".", "").replace(",", "")[:50]
            
            labels.append({
                "cluster_id": cluster_id,
                "label": label,
                "confidence": 0.5
            })
        
        return labels
    
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
        
        # Check grammatical consistency
        structures = []
        for label in label_texts:
            words = label.split()
            if len(words) > 0:
                # Simple structure detection
                structure = (
                    "question" if words[0].lower() in [
                        "how", "what", "why", "when", "where"
                    ]
                    else "noun" if words[0][0].isupper()
                    else "other"
                )
                structures.append(structure)
        
        if structures:
            from collections import Counter
            most_common = Counter(structures).most_common(1)[0]
            consistency_score = most_common[1] / len(structures)
        else:
            consistency_score = 0.5
        
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
            consistency_score * 0.3 +
            length_score * 0.3
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
            metrics = cluster_data.get("metrics", {})
            
            prompt = format_description_prompt(
                label=label.label,
                sample_keywords=keywords[:15],
                count=len(keywords),
                total_volume=metrics.get("total_volume", 0),
                avg_difficulty=metrics.get("avg_difficulty", 50),
                intent=label.intent.primary_intent.value
            )
            
            try:
                description = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=200
                )
                label.description = description.strip()
            except Exception:
                label.description = f"Cluster containing {len(keywords)} keywords"
        
        return labels