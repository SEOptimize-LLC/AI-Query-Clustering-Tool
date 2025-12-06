"""
Metrics display components for Streamlit.
Renders cluster metrics and statistics.
"""
import streamlit as st
from typing import Dict, List, Any


def display_summary_metrics(
    total_keywords: int,
    total_clusters: int,
    cluster_rate: float,
    total_volume: int,
    avg_difficulty: float
):
    """
    Display top-level summary metrics.
    
    Args:
        total_keywords: Total number of keywords
        total_clusters: Number of clusters
        cluster_rate: Percentage of keywords clustered
        total_volume: Total search volume
        avg_difficulty: Average keyword difficulty
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Keywords",
            value=f"{total_keywords:,}"
        )
    
    with col2:
        st.metric(
            label="Clusters",
            value=f"{total_clusters:,}"
        )
    
    with col3:
        st.metric(
            label="Cluster Rate",
            value=f"{cluster_rate:.1%}",
            delta="Good" if cluster_rate > 0.8 else "Low"
        )
    
    with col4:
        st.metric(
            label="Total Volume",
            value=f"{total_volume:,}"
        )
    
    with col5:
        st.metric(
            label="Avg. Difficulty",
            value=f"{avg_difficulty:.1f}",
            delta="Easy" if avg_difficulty < 30 else None,
            delta_color="normal"
        )


def display_cluster_card(
    cluster_id: int,
    label: str,
    keywords: List[str],
    metrics: Dict[str, Any],
    intent: str = None,
    quality_score: float = None,
    expanded: bool = False
):
    """
    Display a cluster card with details.
    
    Args:
        cluster_id: Cluster identifier
        label: Cluster label
        keywords: Keywords in cluster
        metrics: Cluster metrics (volume, difficulty, etc.)
        intent: Search intent type
        quality_score: Quality score (0-100)
        expanded: Whether to show expanded view
    """
    volume = metrics.get("total_volume", 0)
    difficulty = metrics.get("avg_difficulty", 50)
    size = len(keywords)
    
    # Card header with metrics
    with st.expander(
        f"**{label}** ({size} keywords)",
        expanded=expanded
    ):
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volume", f"{volume:,}")
        
        with col2:
            diff_color = (
                "green" if difficulty < 30
                else "orange" if difficulty < 60
                else "red"
            )
            st.markdown(
                f"**Difficulty:** :{diff_color}[{difficulty:.1f}]"
            )
        
        with col3:
            if intent:
                intent_emoji = {
                    "informational": "📚",
                    "transactional": "💰",
                    "commercial": "🛒",
                    "navigational": "🧭"
                }.get(intent.lower(), "❓")
                st.markdown(f"**Intent:** {intent_emoji} {intent.title()}")
        
        with col4:
            if quality_score is not None:
                q_color = (
                    "green" if quality_score >= 70
                    else "orange" if quality_score >= 40
                    else "red"
                )
                st.markdown(
                    f"**Quality:** :{q_color}[{quality_score:.0f}/100]"
                )
        
        # Keywords list
        st.markdown("**Top Keywords:**")
        
        # Show top 10 keywords
        for kw in keywords[:10]:
            st.markdown(f"• {kw}")
        
        if len(keywords) > 10:
            st.markdown(f"*...and {len(keywords) - 10} more*")


def display_cluster_table(
    clusters: List[Dict],
    on_select: callable = None
):
    """
    Display clusters in a data table format.
    
    Args:
        clusters: List of cluster dictionaries
        on_select: Callback when cluster is selected
    """
    import pandas as pd
    
    # Build dataframe
    data = []
    for cluster in clusters:
        data.append({
            "Cluster": cluster.get("label", f"Cluster {cluster['id']}"),
            "Keywords": cluster.get("size", 0),
            "Volume": cluster.get("total_volume", 0),
            "Avg. KD": cluster.get("avg_difficulty", 0),
            "Intent": cluster.get("intent", "Unknown").title(),
            "Quality": cluster.get("quality_score", 0)
        })
    
    df = pd.DataFrame(data)
    
    # Style the dataframe
    styled_df = df.style.background_gradient(
        subset=["Volume"],
        cmap="Blues"
    ).background_gradient(
        subset=["Quality"],
        cmap="RdYlGn",
        vmin=0,
        vmax=100
    ).format({
        "Volume": "{:,.0f}",
        "Avg. KD": "{:.1f}",
        "Quality": "{:.0f}"
    })
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )


def display_unclustered_keywords(
    keywords: List[str],
    limit: int = 50
):
    """
    Display unclustered keywords.
    
    Args:
        keywords: List of unclustered keywords
        limit: Maximum to show
    """
    st.markdown(f"### Unclustered Keywords ({len(keywords)} total)")
    
    if not keywords:
        st.success("All keywords were successfully clustered!")
        return
    
    # Warning message
    if len(keywords) > 0:
        rate = len(keywords)
        st.warning(
            f"{rate} keywords could not be assigned to clusters. "
            "Consider reviewing them manually."
        )
    
    # Show keywords
    col1, col2 = st.columns(2)
    
    half = min(limit // 2, len(keywords) // 2)
    
    with col1:
        for kw in keywords[:half]:
            st.markdown(f"• {kw}")
    
    with col2:
        for kw in keywords[half:limit]:
            st.markdown(f"• {kw}")
    
    if len(keywords) > limit:
        st.info(f"Showing {limit} of {len(keywords)} unclustered keywords")


def display_processing_stats(
    stats: Dict[str, Any]
):
    """
    Display processing statistics.
    
    Args:
        stats: Processing statistics dictionary
    """
    st.markdown("### Processing Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Timing:**")
        st.markdown(
            f"• Total time: {stats.get('total_time', 0):.1f}s"
        )
        st.markdown(
            f"• Embedding: {stats.get('embedding_time', 0):.1f}s"
        )
        st.markdown(
            f"• Clustering: {stats.get('clustering_time', 0):.1f}s"
        )
        st.markdown(
            f"• Labeling: {stats.get('labeling_time', 0):.1f}s"
        )
    
    with col2:
        st.markdown("**API Usage:**")
        st.markdown(
            f"• OpenAI tokens: {stats.get('openai_tokens', 0):,}"
        )
        st.markdown(
            f"• LLM tokens: {stats.get('llm_tokens', 0):,}"
        )
        st.markdown(
            f"• SERP queries: {stats.get('serp_queries', 0)}"
        )
        st.markdown(
            f"• DataForSEO calls: {stats.get('dataforseo_calls', 0)}"
        )


def display_quality_summary(
    cluster_rate: float,
    avg_quality: float,
    serp_overlap: float,
    consistency_score: float
):
    """
    Display quality summary with gauges.
    
    Args:
        cluster_rate: Percentage of keywords clustered
        avg_quality: Average cluster quality score
        serp_overlap: Average SERP overlap
        consistency_score: Label consistency score
    """
    st.markdown("### Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _display_gauge(
            "Cluster Rate",
            cluster_rate * 100,
            suffix="%"
        )
    
    with col2:
        _display_gauge(
            "Avg Quality",
            avg_quality,
            suffix="/100"
        )
    
    with col3:
        _display_gauge(
            "SERP Coherence",
            serp_overlap * 100,
            suffix="%"
        )
    
    with col4:
        _display_gauge(
            "Label Consistency",
            consistency_score * 100,
            suffix="%"
        )


def _display_gauge(
    label: str,
    value: float,
    suffix: str = ""
):
    """Display a simple gauge using progress bar."""
    color = (
        "green" if value >= 70
        else "orange" if value >= 40
        else "red"
    )
    
    st.markdown(f"**{label}**")
    st.progress(min(value / 100, 1.0))
    st.markdown(f":{color}[{value:.1f}{suffix}]")


def display_intent_distribution(
    intent_counts: Dict[str, int]
):
    """
    Display intent distribution as horizontal bars.
    
    Args:
        intent_counts: Count of clusters per intent type
    """
    st.markdown("### Intent Distribution")
    
    total = sum(intent_counts.values())
    if total == 0:
        st.info("No intent data available")
        return
    
    intent_colors = {
        "informational": "🟢",
        "transactional": "🔴",
        "commercial": "🟠",
        "navigational": "🔵"
    }
    
    for intent, count in sorted(
        intent_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        pct = count / total * 100
        emoji = intent_colors.get(intent.lower(), "⚪")
        st.markdown(f"{emoji} **{intent.title()}:** {count} ({pct:.1f}%)")
        st.progress(count / total)