"""
LLM prompt templates for label generation.
Implements two-phase labeling for consistency.
"""

# Phase 1: Generate cluster summaries
CLUSTER_SUMMARY_PROMPT = """Analyze this cluster of keywords and provide a brief summary.

Keywords in cluster:
{keywords}

Search intent indicators:
- Primary intent type: {intent_type}
- Top keywords by volume: {top_keywords}
{serp_info}

Provide a 1-2 sentence summary describing:
1. The main topic/theme
2. The user's GOAL or what action they want to take (not just "learn" or "find")
3. The search intent (informational, transactional, navigational, commercial)
4. Key distinguishing characteristics that make this cluster unique

Focus on identifying the specific user need or action, not just the subject matter.

Summary:"""


# Phase 2: Generate consistent labels with global context
CONSISTENT_LABELING_PROMPT = """You are an SEO expert creating cluster labels for \
keyword research.

CONTEXT:
You have {n_clusters} keyword clusters to label. Each cluster needs:
1. A short, action-oriented label (2-5 words)
2. Labels must be MUTUALLY EXCLUSIVE (no overlap in meaning)
3. Labels must follow a CONSISTENT naming convention
4. Labels must describe the USER'S GOAL, not just the topic

CLUSTER SUMMARIES:
{cluster_summaries}

CRITICAL RULES FOR ACTION-ORIENTED LABELS:
- START with an action verb that matches the search intent:
  * INFORMATIONAL: "Learn", "Understand", "Discover", "Explore"
  * COMMERCIAL: "Compare", "Choose", "Find Best", "Evaluate"
  * TRANSACTIONAL: "Find", "Get", "Buy", "Hire", "Book"
  * NAVIGATIONAL: Use brand/site name directly
- Examples of GOOD labels:
  * "Calculate Fireplace BTU Requirements" (informational)
  * "Find Fireplace Repair Services" (transactional)
  * "Compare Gas Fireplace Options" (commercial)
  * "Learn About Fireplace Safety" (informational)
- Examples of BAD labels (avoid these):
  * "Fireplace BTU" (too generic, no action)
  * "Fireplace Repair" (no user goal)
  * "Gas Fireplaces" (just a topic, not actionable)
- Make labels specific enough to distinguish clusters
- Avoid generic labels like "Other" or "Miscellaneous"

Generate a label for each cluster. Output as JSON:
{{
    "labels": [
        {{"cluster_id": 0, "label": "Your Label Here", "confidence": 0.9}},
        ...
    ]
}}

Labels:"""


# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """Classify the search intent for these keywords.

Keywords:
{keywords}

Classify as one of:
- INFORMATIONAL: User wants to learn/understand something
- TRANSACTIONAL: User wants to buy/download/sign up
- NAVIGATIONAL: User wants to find a specific website/page
- COMMERCIAL: User is researching before a purchase decision

Also identify any modifiers:
- Local intent (location-specific)
- Comparison intent (vs, compare, best)
- Question intent (how, what, why)
- Branded intent (contains brand names)

Output as JSON:
{{
    "primary_intent": "INTENT_TYPE",
    "confidence": 0.9,
    "modifiers": ["modifier1", "modifier2"],
    "explanation": "Brief explanation"
}}

Analysis:"""


# Label refinement prompt (for post-processing)
LABEL_REFINEMENT_PROMPT = """Review and refine these cluster labels for consistency.

Current labels:
{labels}

Issues to fix:
1. Inconsistent naming conventions
2. Overlapping meanings
3. Labels that are too vague or too specific
4. Grammatical inconsistencies

For each label that needs refinement, provide the correction.
Output as JSON:
{{
    "refinements": [
        {{"cluster_id": 0, "original": "Old Label", "refined": "New Label"}},
        ...
    ],
    "unchanged": [1, 2, 3]
}}

Refinements:"""


# Cluster description prompt
CLUSTER_DESCRIPTION_PROMPT = """Generate a detailed description for this cluster.

Label: {label}
Keywords ({count} total):
{sample_keywords}

Metrics:
- Total search volume: {total_volume:,}
- Average keyword difficulty: {avg_difficulty:.1f}
- Intent: {intent}

Write a 2-3 sentence description explaining:
1. What topics/themes these keywords cover
2. What type of content would target these keywords
3. The user intent and funnel stage

Description:"""


# Batch label generation (for efficiency)
BATCH_LABEL_PROMPT = """Generate ACTION-ORIENTED labels for multiple keyword clusters.

Clusters to label:
{clusters_data}

Requirements:
1. Each label should be 2-5 words
2. START with an action verb that matches search intent:
   - Informational: Learn, Understand, Discover, Calculate
   - Commercial: Compare, Choose, Find Best, Evaluate
   - Transactional: Find, Get, Buy, Hire, Book
3. Labels must describe the USER'S GOAL, not just the topic
4. Labels must be unique and non-overlapping
5. Use consistent naming conventions

Examples:
- "Calculate Fireplace BTU Requirements" (NOT "Fireplace BTU")
- "Find Fireplace Repair Services" (NOT "Fireplace Repair")
- "Compare Gas Fireplace Options" (NOT "Gas Fireplaces")

Output as JSON array:
[
    {{"cluster_id": 0, "label": "Label", "intent": "TYPE", "confidence": 0.9}},
    ...
]

Labels:"""


def format_cluster_summary_prompt(
    keywords: list,
    intent_type: str = "Unknown",
    top_keywords: list = None,
    serp_data: dict = None
) -> str:
    """Format the cluster summary prompt with data."""
    top_kw_str = ", ".join(top_keywords[:5]) if top_keywords else "N/A"
    
    serp_info = ""
    if serp_data:
        common_domains = serp_data.get("common_domains", [])
        if common_domains:
            serp_info = f"- Common ranking domains: {', '.join(common_domains[:5])}"
    
    return CLUSTER_SUMMARY_PROMPT.format(
        keywords="\n".join(f"- {kw}" for kw in keywords[:20]),
        intent_type=intent_type,
        top_keywords=top_kw_str,
        serp_info=serp_info
    )


def format_consistent_labeling_prompt(
    cluster_summaries: list,
    n_clusters: int
) -> str:
    """Format the consistent labeling prompt with all summaries."""
    summaries_text = "\n\n".join(
        f"Cluster {s['cluster_id']}:\n{s['summary']}"
        for s in cluster_summaries
    )
    
    return CONSISTENT_LABELING_PROMPT.format(
        n_clusters=n_clusters,
        cluster_summaries=summaries_text
    )


def format_intent_prompt(keywords: list) -> str:
    """Format the intent classification prompt."""
    return INTENT_CLASSIFICATION_PROMPT.format(
        keywords="\n".join(f"- {kw}" for kw in keywords[:30])
    )


def format_description_prompt(
    label: str,
    sample_keywords: list,
    count: int,
    total_volume: int,
    avg_difficulty: float,
    intent: str
) -> str:
    """Format the cluster description prompt."""
    return CLUSTER_DESCRIPTION_PROMPT.format(
        label=label,
        count=count,
        sample_keywords="\n".join(f"- {kw}" for kw in sample_keywords[:15]),
        total_volume=total_volume,
        avg_difficulty=avg_difficulty,
        intent=intent
    )