"""
Chart generation for cluster visualization.
Uses Plotly for interactive charts.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def create_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    keywords: List[str],
    cluster_names: Dict[int, str] = None,
    method: str = "tsne",
    perplexity: int = 30
) -> go.Figure:
    """
    Create interactive 2D scatter plot of clusters.
    
    Args:
        embeddings: Keyword embeddings (n_keywords, n_dims)
        labels: Cluster labels
        keywords: Keyword strings for hover
        cluster_names: Optional mapping of cluster_id to name
        method: Dimensionality reduction method (tsne/pca)
        perplexity: t-SNE perplexity parameter
    
    Returns:
        Plotly figure
    """
    # Reduce dimensions
    if embeddings.shape[0] > 5000:
        # Use PCA for large datasets (faster)
        method = "pca"
    
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=42,
            n_iter=1000
        )
    else:
        reducer = PCA(n_components=2)
    
    coords = reducer.fit_transform(embeddings)
    
    # Prepare labels for display
    cluster_names = cluster_names or {}
    display_labels = [
        cluster_names.get(lbl, f"Cluster {lbl}" if lbl >= 0 else "Unclustered")
        for lbl in labels
    ]
    
    # Create color mapping
    unique_labels = sorted(set(labels))
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    color_map = {
        lbl: colors[i % len(colors)]
        for i, lbl in enumerate(unique_labels)
    }
    color_map[-1] = "#cccccc"  # Gray for unclustered
    
    point_colors = [color_map[lbl] for lbl in labels]
    
    # Create figure
    fig = go.Figure()
    
    # Add points by cluster for legend
    for lbl in unique_labels:
        mask = labels == lbl
        name = cluster_names.get(lbl, f"Cluster {lbl}")
        if lbl == -1:
            name = "Unclustered"
        
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=name,
            text=[keywords[i] for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=6,
                color=color_map[lbl],
                opacity=0.7
            )
        ))
    
    fig.update_layout(
        title="Keyword Clusters Visualization",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        height=600
    )
    
    return fig


def create_cluster_distribution(
    cluster_sizes: Dict[int, int],
    cluster_names: Dict[int, str] = None,
    top_n: int = 20
) -> go.Figure:
    """
    Create bar chart of cluster size distribution.
    
    Args:
        cluster_sizes: Mapping of cluster_id to size
        cluster_names: Optional mapping of cluster_id to name
        top_n: Show top N clusters
    
    Returns:
        Plotly figure
    """
    cluster_names = cluster_names or {}
    
    # Sort by size
    sorted_clusters = sorted(
        cluster_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    ids = [c[0] for c in sorted_clusters]
    sizes = [c[1] for c in sorted_clusters]
    names = [cluster_names.get(cid, f"Cluster {cid}") for cid in ids]
    
    fig = go.Figure(go.Bar(
        x=names,
        y=sizes,
        marker_color=px.colors.qualitative.Set3[:len(names)],
        text=sizes,
        textposition="outside"
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Clusters by Size",
        xaxis_title="Cluster",
        yaxis_title="Number of Keywords",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig


def create_volume_treemap(
    cluster_data: List[dict],
    cluster_names: Dict[int, str] = None
) -> go.Figure:
    """
    Create treemap showing search volume by cluster.
    
    Args:
        cluster_data: List of dicts with cluster_id, volume, size
        cluster_names: Optional cluster name mapping
    
    Returns:
        Plotly figure
    """
    cluster_names = cluster_names or {}
    
    labels = []
    parents = []
    values = []
    colors = []
    
    # Root
    total_volume = sum(c.get("volume", 0) for c in cluster_data)
    
    for cluster in sorted(
        cluster_data,
        key=lambda x: x.get("volume", 0),
        reverse=True
    ):
        cid = cluster["cluster_id"]
        volume = cluster.get("volume", 0)
        size = cluster.get("size", 0)
        
        name = cluster_names.get(cid, f"Cluster {cid}")
        labels.append(f"{name}<br>{volume:,} vol | {size} kw")
        parents.append("")
        values.append(volume)
        colors.append(volume)
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Volume")
        ),
        textinfo="label",
        hovertemplate="<b>%{label}</b><extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Search Volume Distribution (Total: {total_volume:,})",
        template="plotly_white",
        height=600
    )
    
    return fig


def create_difficulty_distribution(
    cluster_data: List[dict],
    cluster_names: Dict[int, str] = None
) -> go.Figure:
    """
    Create scatter plot of volume vs difficulty by cluster.
    
    Args:
        cluster_data: List with cluster_id, volume, difficulty, size
        cluster_names: Optional cluster name mapping
    
    Returns:
        Plotly figure
    """
    cluster_names = cluster_names or {}
    
    volumes = []
    difficulties = []
    sizes = []
    names = []
    
    for cluster in cluster_data:
        cid = cluster["cluster_id"]
        volumes.append(cluster.get("volume", 0))
        difficulties.append(cluster.get("avg_difficulty", 50))
        sizes.append(cluster.get("size", 10))
        names.append(cluster_names.get(cid, f"Cluster {cid}"))
    
    # Size scale
    max_size = max(sizes) if sizes else 1
    marker_sizes = [10 + (s / max_size) * 40 for s in sizes]
    
    fig = go.Figure(go.Scatter(
        x=difficulties,
        y=volumes,
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(
            size=marker_sizes,
            color=difficulties,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Difficulty")
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Volume: %{y:,}<br>"
            "Difficulty: %{x:.1f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title="Cluster Opportunity Matrix",
        xaxis_title="Average Keyword Difficulty",
        yaxis_title="Total Search Volume",
        template="plotly_white",
        height=600,
        showlegend=False
    )
    
    # Add quadrant lines
    avg_diff = np.mean(difficulties) if difficulties else 50
    avg_vol = np.median(volumes) if volumes else 0
    
    fig.add_hline(y=avg_vol, line_dash="dash", line_color="gray")
    fig.add_vline(x=avg_diff, line_dash="dash", line_color="gray")
    
    # Quadrant labels
    fig.add_annotation(
        x=avg_diff * 0.3, y=avg_vol * 2,
        text="Quick Wins", showarrow=False,
        font=dict(size=12, color="green")
    )
    fig.add_annotation(
        x=avg_diff * 1.5, y=avg_vol * 2,
        text="High Value", showarrow=False,
        font=dict(size=12, color="orange")
    )
    
    return fig


def create_intent_breakdown(
    intent_counts: Dict[str, int]
) -> go.Figure:
    """
    Create pie chart of search intent distribution.
    
    Args:
        intent_counts: Mapping of intent type to count
    
    Returns:
        Plotly figure
    """
    intent_colors = {
        "informational": "#4CAF50",
        "transactional": "#F44336",
        "commercial": "#FF9800",
        "navigational": "#2196F3",
        "unknown": "#9E9E9E"
    }
    
    labels = list(intent_counts.keys())
    values = list(intent_counts.values())
    colors = [intent_colors.get(lbl.lower(), "#9E9E9E") for lbl in labels]
    
    fig = go.Figure(go.Pie(
        labels=[lbl.title() for lbl in labels],
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} clusters<extra></extra>"
    ))
    
    fig.update_layout(
        title="Cluster Intent Distribution",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_metrics_gauge(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    thresholds: Dict[str, float] = None
) -> go.Figure:
    """
    Create gauge chart for metrics.
    
    Args:
        value: Current value
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: Color thresholds
    
    Returns:
        Plotly figure
    """
    thresholds = thresholds or {
        "green": 70,
        "yellow": 40
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title),
        gauge=dict(
            axis=dict(range=[min_val, max_val]),
            bar=dict(color="darkblue"),
            steps=[
                dict(range=[min_val, thresholds.get("yellow", 40)], color="red"),
                dict(
                    range=[thresholds.get("yellow", 40), thresholds.get("green", 70)],
                    color="yellow"
                ),
                dict(range=[thresholds.get("green", 70), max_val], color="green"),
            ]
        )
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=250
    )
    
    return fig


def create_cluster_quality_chart(
    clusters: List[dict]
) -> go.Figure:
    """
    Create horizontal bar chart of cluster quality scores.
    
    Args:
        clusters: List with cluster_id, label, quality_score
    
    Returns:
        Plotly figure
    """
    # Sort by quality
    sorted_clusters = sorted(
        clusters,
        key=lambda x: x.get("quality_score", 0),
        reverse=True
    )[:20]
    
    labels = [c.get("label", f"Cluster {c['cluster_id']}") for c in sorted_clusters]
    scores = [c.get("quality_score", 0) for c in sorted_clusters]
    
    colors = [
        "#4CAF50" if s >= 70
        else "#FF9800" if s >= 40
        else "#F44336"
        for s in scores
    ]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Cluster Quality Scores",
        xaxis_title="Quality Score",
        yaxis_title="",
        template="plotly_white",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig