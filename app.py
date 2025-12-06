"""
AI Keyword Clustering Application

A Streamlit app for semantic keyword clustering with:
- LLM-powered consistent labeling
- SERP validation for cluster coherence
- DataForSEO integration for keyword metrics
- Interactive visualizations
"""
import streamlit as st
import asyncio
import time
from datetime import datetime

# Page config must be first
st.set_page_config(
    page_title="AI Keyword Clustering",
    page_icon="🔑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from config.settings import Settings
from config.api_config import LOCATION_CODES, LANGUAGE_CODES
from core.job_manager import JobManager
from core.progress_tracker import ProgressTracker
from storage.supabase_client import SupabaseClient
from storage.cache_manager import CacheManager
from ingestion.csv_parser import CSVParser, preview_uploaded_file
from ingestion.validator import KeywordValidator
from ingestion.deduplicator import KeywordDeduplicator
from enrichment.dataforseo_client import DataForSEOClient
from enrichment.serper_client import SerperClient
from enrichment.metrics_aggregator import MetricsAggregator
from embedding.openai_embedder import OpenAIEmbedder
from embedding.batch_processor import BatchEmbeddingProcessor
from clustering.pipeline import ClusteringPipeline
from clustering.coarse_clusterer import CoarseClusterer
from clustering.fine_clusterer import FineClusterer
from clustering.serp_validator import SERPValidator
from clustering.outlier_handler import OutlierHandler
from labeling.label_generator import LabelGenerator
from labeling.llm_client import OpenRouterClient
from labeling.intent_classifier import IntentClassifier
from visualization.charts import (
    create_cluster_scatter,
    create_cluster_distribution,
    create_volume_treemap,
    create_difficulty_distribution
)
from visualization.metrics_display import (
    display_summary_metrics,
    display_cluster_table,
    display_quality_summary
)
from export.csv_exporter import CSVExporter
from export.excel_exporter import ExcelExporter


def init_session_state():
    """Initialize session state variables."""
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "error" not in st.session_state:
        st.session_state.error = None
    if "csv_preview" not in st.session_state:
        st.session_state.csv_preview = None
    if "csv_parser" not in st.session_state:
        st.session_state.csv_parser = None
    if "keywords_ready" not in st.session_state:
        st.session_state.keywords_ready = None


def check_api_keys() -> dict:
    """Check which API keys are configured."""
    settings = Settings()
    return {
        "supabase": bool(settings.supabase_url and settings.supabase_key),
        "openai": bool(settings.openai_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "dataforseo": bool(
            settings.dataforseo_login and settings.dataforseo_password
        ),
        "serper": bool(settings.serper_api_key)
    }


def render_sidebar():
    """Render sidebar with configuration."""
    st.sidebar.title("🔑 AI Keyword Clustering")
    
    # API Status
    st.sidebar.markdown("### API Status")
    api_status = check_api_keys()
    
    for api, configured in api_status.items():
        icon = "✅" if configured else "❌"
        st.sidebar.markdown(f"{icon} {api.title()}")
    
    if not all(api_status.values()):
        st.sidebar.warning(
            "Some APIs not configured. Add keys in Streamlit Secrets."
        )
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.markdown("### Settings")
    
    location = st.sidebar.selectbox(
        "Target Location",
        options=list(LOCATION_CODES.keys()),
        index=0  # United States
    )
    
    language = st.sidebar.selectbox(
        "Language",
        options=list(LANGUAGE_CODES.keys()),
        index=0  # English
    )
    
    st.sidebar.markdown("---")
    
    # LLM Model Selection
    st.sidebar.markdown("### LLM Model")
    llm_models = {
        "Claude 4 Sonnet (Recommended)": "anthropic/claude-sonnet-4",
        "Gemini 2.5 Flash": "google/gemini-2.5-flash-preview-05-20",
        "GPT-4.1 Mini": "openai/gpt-4.1-mini"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Model for Labeling",
        options=list(llm_models.keys()),
        index=0,
        help="LLM used via OpenRouter for cluster labeling"
    )
    selected_model = llm_models[selected_model_name]
    
    st.sidebar.markdown("---")
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=2,
            max_value=20,
            value=5
        )
        
        serp_validation = st.checkbox(
            "Enable SERP Validation",
            value=True,
            help="Validate clusters using SERP overlap"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Threshold for outlier reassignment"
        )
    
    return {
        "location": location,
        "language": language,
        "llm_model": selected_model,
        "min_cluster_size": min_cluster_size,
        "serp_validation": serp_validation,
        "similarity_threshold": similarity_threshold
    }


def render_upload_section():
    """Render file upload section with column selection."""
    st.markdown("## 📤 Upload Keywords")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with keywords",
            type=["csv"],
            help="Any CSV file - you'll select the keyword column"
        )
    
    with col2:
        st.markdown("**Supported Formats:**")
        st.markdown(
            "- Google Search Console exports\n"
            "- Ahrefs/SEMrush exports\n"
            "- Custom keyword lists\n"
            "- Any CSV with a keyword column"
        )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Check if we need to preview (new file or different file)
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if (st.session_state.csv_preview is None or
                st.session_state.get("file_key") != file_key):
            try:
                parser, preview = preview_uploaded_file(uploaded_file)
                st.session_state.csv_preview = preview
                st.session_state.csv_parser = parser
                st.session_state.file_key = file_key
                st.session_state.keywords_ready = None
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None
        
        preview = st.session_state.csv_preview
        parser = st.session_state.csv_parser
        
        # Show file info
        st.success(
            f"📄 Loaded **{preview['row_count']:,}** rows "
            f"with **{len(preview['columns'])}** columns"
        )
        
        # Column selection UI
        st.markdown("### Select Columns")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            # Keyword column (required)
            kw_options = [""] + preview['columns']
            default_kw = 0
            if preview['detected_keyword_column']:
                try:
                    default_kw = kw_options.index(
                        preview['detected_keyword_column']
                    )
                except ValueError:
                    default_kw = 0
            
            keyword_col = st.selectbox(
                "**Keyword Column** (required)",
                options=kw_options,
                index=default_kw,
                help="Column containing keywords/search queries"
            )
        
        with col_b:
            # Volume column (optional)
            vol_options = ["(none)"] + preview['columns']
            default_vol = 0
            if preview['detected_volume_column']:
                try:
                    default_vol = vol_options.index(
                        preview['detected_volume_column']
                    )
                except ValueError:
                    default_vol = 0
            
            volume_col = st.selectbox(
                "Search Volume Column (optional)",
                options=vol_options,
                index=default_vol,
                help="Will fetch from DataForSEO if not provided"
            )
        
        with col_c:
            # KD column (optional)
            kd_options = ["(none)"] + preview['columns']
            default_kd = 0
            if preview['detected_kd_column']:
                try:
                    default_kd = kd_options.index(
                        preview['detected_kd_column']
                    )
                except ValueError:
                    default_kd = 0
            
            kd_col = st.selectbox(
                "Difficulty Column (optional)",
                options=kd_options,
                index=default_kd,
                help="Will fetch from DataForSEO if not provided"
            )
        
        # Show data preview
        with st.expander("📋 Preview Data", expanded=True):
            import pandas as pd
            sample_df = pd.DataFrame(preview['sample_data'])
            st.dataframe(sample_df, use_container_width=True)
        
        # Load keywords button
        if keyword_col and keyword_col != "":
            if st.button("✅ Confirm Column Selection", type="secondary"):
                try:
                    vol = volume_col if volume_col != "(none)" else None
                    kd = kd_col if kd_col != "(none)" else None
                    
                    keywords_data = parser.parse_with_selection(
                        keyword_column=keyword_col,
                        volume_column=vol,
                        kd_column=kd
                    )
                    st.session_state.keywords_ready = keywords_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing keywords: {e}")
        else:
            st.warning("⚠️ Please select a keyword column")
        
        return st.session_state.keywords_ready
    else:
        # Reset state when no file
        st.session_state.csv_preview = None
        st.session_state.csv_parser = None
        st.session_state.keywords_ready = None
        st.session_state.file_key = None
    
    return None


async def process_keywords(
    keywords_data: list,
    config: dict,
    progress_container
):
    """
    Main processing pipeline.
    
    Args:
        keywords_data: List of keyword dictionaries
        config: Processing configuration
        progress_container: Streamlit container for progress
    """
    settings = Settings()
    start_time = time.time()
    
    # Initialize components
    supabase = SupabaseClient(settings.supabase_url, settings.supabase_key)
    cache = CacheManager(supabase)
    job_manager = JobManager(supabase)
    
    # Create job
    job_id = job_manager.create_job(
        total_keywords=len(keywords_data),
        config=config
    )
    st.session_state.job_id = job_id
    
    # Progress tracker
    progress = ProgressTracker(total_keywords=len(keywords_data))
    progress_placeholder = progress_container.empty()
    
    def update_progress(stage, pct, message):
        progress.update_stage(stage, pct, message)
        with progress_placeholder:
            progress.render()
    
    try:
        # Stage 1: Fetch metrics
        update_progress("metrics", 0.0, "Fetching keyword metrics...")
        
        keywords_text = [kw["keyword"] for kw in keywords_data]
        
        # Check cache first
        cached_metrics = cache.get_cached_metrics(keywords_text)
        uncached = [kw for kw in keywords_text if kw not in cached_metrics]
        
        if uncached:
            dataforseo = DataForSEOClient(
                login=settings.dataforseo_login,
                password=settings.dataforseo_password
            )
            
            location_code = LOCATION_CODES.get(config["location"], 2840)
            language_code = LANGUAGE_CODES.get(config["language"], "en")
            
            new_metrics = await dataforseo.get_keyword_metrics(
                keywords=uncached,
                location_code=location_code,
                language_code=language_code,
                progress_callback=lambda p: update_progress(
                    "metrics", p, f"Fetching metrics: {int(p*100)}%"
                )
            )
            
            # Cache new metrics
            cache.cache_metrics(new_metrics)
            cached_metrics.update(new_metrics)
        
        # Merge metrics into keyword data
        for kw in keywords_data:
            if kw["keyword"] in cached_metrics:
                metrics = cached_metrics[kw["keyword"]]
                kw["search_volume"] = metrics.get("search_volume", 0)
                kw["keyword_difficulty"] = metrics.get("keyword_difficulty", 0)
        
        update_progress("metrics", 1.0, "Metrics fetched!")
        
        # Stage 2: Generate embeddings
        update_progress("embedding", 0.0, "Generating embeddings...")
        
        embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
        batch_processor = BatchEmbeddingProcessor(embedder, supabase)
        
        # process_keywords is sync, returns Dict[str, np.ndarray]
        embeddings_dict = batch_processor.process_keywords(
            keywords=keywords_text,
            job_id=job_id,
            progress_callback=lambda p: update_progress(
                "embedding", p.progress, f"Embedding: {int(p.progress*100)}%"
            )
        )
        
        # Convert dict to ordered numpy array
        import numpy as np
        embeddings = np.array([
            embeddings_dict[kw] for kw in keywords_text
        ])
        
        update_progress("embedding", 1.0, "Embeddings complete!")
        
        # Stage 3: Clustering
        update_progress("clustering", 0.0, "Clustering keywords...")
        
        # Build clustering pipeline
        coarse = CoarseClusterer()
        fine = FineClusterer(min_cluster_size=config["min_cluster_size"])
        outlier = OutlierHandler(threshold=config["similarity_threshold"])
        
        serp_val = None
        if config["serp_validation"]:
            serper = SerperClient(api_key=settings.serper_api_key)
            serp_val = SERPValidator(serper, cache)
        
        pipeline = ClusteringPipeline(
            coarse_clusterer=coarse,
            fine_clusterer=fine,
            serp_validator=serp_val,
            outlier_handler=outlier
        )
        
        # Build keyword data dict
        kw_data = {kw["keyword"]: kw for kw in keywords_data}
        
        cluster_result = pipeline.run(
            embeddings=embeddings,
            keywords=keywords_text,
            keyword_data=kw_data,
            progress_callback=lambda s, p, m: update_progress(s, p, m),
            skip_serp_validation=not config["serp_validation"]
        )
        
        update_progress("clustering", 1.0, "Clustering complete!")
        
        # Stage 4: Label generation
        update_progress("labeling", 0.0, "Generating cluster labels...")
        
        llm = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            default_model=config.get("llm_model", "anthropic/claude-sonnet-4")
        )
        intent_classifier = IntentClassifier()
        label_gen = LabelGenerator(llm, intent_classifier)
        
        # Prepare cluster data for labeling
        cluster_data = {}
        for cid, info in cluster_result.clusters.items():
            cluster_data[cid] = {
                "keywords": info.keywords,
                "top_keywords": sorted(
                    info.keywords,
                    key=lambda k: kw_data.get(k, {}).get("search_volume", 0),
                    reverse=True
                )[:10],
                "serp_data": (
                    info.serp_validation.top_domains
                    if info.serp_validation
                    else None
                )
            }
        
        label_result = await label_gen.generate_labels(
            clusters=cluster_data,
            progress_callback=lambda s, p, m: update_progress(s, p, m)
        )
        
        update_progress("labeling", 1.0, "Labels generated!")
        
        # Stage 5: Aggregate metrics
        update_progress("aggregating", 0.5, "Aggregating metrics...")
        
        aggregator = MetricsAggregator()
        
        # Build final results
        final_clusters = []
        for cid, info in cluster_result.clusters.items():
            label = label_result.labels.get(cid)
            
            # Get keyword metrics for cluster
            cluster_kw_data = [
                kw_data.get(kw, {"keyword": kw})
                for kw in info.keywords
            ]
            
            agg = aggregator.aggregate_cluster(cluster_kw_data)
            
            final_clusters.append({
                "id": cid,
                "label": label.label if label else f"Cluster {cid}",
                "keywords": cluster_kw_data,
                "size": info.size,
                "total_volume": agg["total_volume"],
                "avg_difficulty": agg["avg_difficulty"],
                "intent": (
                    label.intent.primary_intent.value
                    if label else "unknown"
                ),
                "quality_score": info.quality_score,
                "priority_score": agg["priority_score"]
            })
        
        # Get unclustered
        outlier_indices = cluster_result.labels == -1
        unclustered = [
            kw_data.get(keywords_text[i], {"keyword": keywords_text[i]})
            for i in range(len(keywords_text))
            if outlier_indices[i]
        ]
        
        # Calculate totals
        total_volume = sum(c["total_volume"] for c in final_clusters)
        avg_difficulty = (
            sum(c["avg_difficulty"] * c["size"] for c in final_clusters) /
            sum(c["size"] for c in final_clusters)
            if final_clusters else 0
        )
        
        # Intent distribution
        intent_dist = {}
        for c in final_clusters:
            intent = c["intent"]
            intent_dist[intent] = intent_dist.get(intent, 0) + 1
        
        # Build results
        results = {
            "job_id": job_id,
            "clusters": sorted(
                final_clusters,
                key=lambda x: x["priority_score"],
                reverse=True
            ),
            "unclustered": unclustered,
            "total_keywords": len(keywords_data),
            "total_clusters": cluster_result.n_clusters,
            "clustered_keywords": cluster_result.clustered_keywords,
            "unclustered_count": cluster_result.outlier_keywords,
            "cluster_rate": cluster_result.cluster_rate,
            "total_volume": total_volume,
            "avg_difficulty": avg_difficulty,
            "avg_quality": (
                sum(c["quality_score"] for c in final_clusters) /
                len(final_clusters) if final_clusters else 0
            ),
            "serp_overlap": cluster_result.avg_serp_overlap,
            "consistency_score": label_result.consistency_score,
            "intent_distribution": intent_dist,
            "processing_time": time.time() - start_time,
            "embeddings": embeddings
        }
        
        # Save to job
        job_manager.update_job(job_id, "completed", results)
        
        update_progress("aggregating", 1.0, "Processing complete!")
        
        return results
        
    except Exception as e:
        job_manager.update_job(job_id, "failed", {"error": str(e)})
        raise


def render_results(results: dict):
    """Render clustering results."""
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Summary metrics
    display_summary_metrics(
        total_keywords=results["total_keywords"],
        total_clusters=results["total_clusters"],
        cluster_rate=results["cluster_rate"],
        total_volume=results["total_volume"],
        avg_difficulty=results["avg_difficulty"]
    )
    
    # Quality metrics
    st.markdown("### Quality Metrics")
    display_quality_summary(
        cluster_rate=results["cluster_rate"],
        avg_quality=results["avg_quality"],
        serp_overlap=results["serp_overlap"],
        consistency_score=results["consistency_score"]
    )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Clusters",
        "📈 Charts",
        "🔍 Unclustered",
        "📥 Export"
    ])
    
    with tab1:
        st.markdown("### Cluster Overview")
        display_cluster_table(results["clusters"])
        
        # Expandable cluster details
        st.markdown("### Cluster Details")
        for cluster in results["clusters"][:20]:
            with st.expander(
                f"**{cluster['label']}** "
                f"({cluster['size']} kw, {cluster['total_volume']:,} vol)"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Keywords", cluster["size"])
                with col2:
                    st.metric("Volume", f"{cluster['total_volume']:,}")
                with col3:
                    st.metric("Avg. KD", f"{cluster['avg_difficulty']:.1f}")
                
                st.markdown("**Keywords:**")
                for kw in cluster["keywords"][:15]:
                    if isinstance(kw, dict):
                        vol = kw.get("search_volume", 0)
                        st.markdown(f"• {kw['keyword']} ({vol:,})")
                    else:
                        st.markdown(f"• {kw}")
    
    with tab2:
        st.markdown("### Visualizations")
        
        # Cluster scatter plot
        if "embeddings" in results:
            st.markdown("#### Cluster Map")
            labels_array = []
            names_map = {}
            for c in results["clusters"]:
                names_map[c["id"]] = c["label"]
            
            # Reconstruct labels
            import numpy as np
            all_kw = []
            for c in results["clusters"]:
                for kw in c["keywords"]:
                    if isinstance(kw, dict):
                        all_kw.append(kw["keyword"])
                    else:
                        all_kw.append(kw)
            
            fig = create_cluster_scatter(
                embeddings=results["embeddings"][:min(5000, len(all_kw))],
                labels=np.array([0] * min(5000, len(all_kw))),
                keywords=all_kw[:5000],
                cluster_names=names_map
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution chart
        st.markdown("#### Cluster Size Distribution")
        sizes = {c["id"]: c["size"] for c in results["clusters"]}
        names = {c["id"]: c["label"] for c in results["clusters"]}
        fig = create_cluster_distribution(sizes, names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume treemap
        st.markdown("#### Volume Distribution")
        fig = create_volume_treemap(
            [{"cluster_id": c["id"], "volume": c["total_volume"], "size": c["size"]}
             for c in results["clusters"]],
            names
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Opportunity matrix
        st.markdown("#### Opportunity Matrix")
        fig = create_difficulty_distribution(
            [{"cluster_id": c["id"], "volume": c["total_volume"],
              "avg_difficulty": c["avg_difficulty"], "size": c["size"]}
             for c in results["clusters"]],
            names
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Unclustered Keywords")
        unclustered = results.get("unclustered", [])
        
        if unclustered:
            st.warning(f"{len(unclustered)} keywords could not be clustered")
            
            for kw in unclustered[:50]:
                if isinstance(kw, dict):
                    st.markdown(f"• {kw.get('keyword', '')}")
                else:
                    st.markdown(f"• {kw}")
            
            if len(unclustered) > 50:
                st.info(f"Showing 50 of {len(unclustered)}")
        else:
            st.success("All keywords were successfully clustered!")
    
    with tab4:
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CSV Export")
            
            csv_exporter = CSVExporter()
            
            # Full export
            csv_full = csv_exporter.export_clusters(
                results["clusters"],
                include_keywords=True
            )
            st.download_button(
                label="📄 Download Full CSV",
                data=csv_full,
                file_name=f"clusters_{results['job_id']}.csv",
                mime="text/csv"
            )
            
            # Summary export
            csv_summary = csv_exporter.export_clusters(
                results["clusters"],
                include_keywords=False
            )
            st.download_button(
                label="📋 Download Summary CSV",
                data=csv_summary,
                file_name=f"clusters_summary_{results['job_id']}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("#### Excel Export")
            
            excel_exporter = ExcelExporter()
            excel_data = excel_exporter.export_full_report(results)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"clustering_report_{results['job_id']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument"
                     ".spreadsheetml.sheet"
            )


def main():
    """Main application entry point."""
    init_session_state()
    
    # Sidebar
    config = render_sidebar()
    
    # Main content
    st.title("🔑 AI Keyword Clustering")
    st.markdown(
        "Cluster keywords by semantic similarity and search intent "
        "using AI-powered analysis."
    )
    
    # Upload section - returns keywords_data when ready
    keywords_data = render_upload_section()
    
    if keywords_data is not None:
        st.markdown("---")
        st.markdown("### 🔍 Keyword Validation")
        
        # Validate
        validator = KeywordValidator()
        valid, invalid = validator.validate_batch(keywords_data)
        
        if invalid:
            st.warning(f"{len(invalid)} invalid keywords removed")
        
        # Deduplicate
        dedup = KeywordDeduplicator()
        unique = dedup.deduplicate(valid)
        
        if len(unique) < len(valid):
            st.info(f"Removed {len(valid) - len(unique)} duplicates")
        
        # Show ready count with metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Keywords Ready", f"{len(unique):,}")
        with col2:
            has_vol = sum(
                1 for kw in unique
                if kw.get("search_volume", 0) > 0
            )
            st.metric("With Volume Data", f"{has_vol:,}")
        with col3:
            has_kd = sum(
                1 for kw in unique
                if kw.get("keyword_difficulty", 0) > 0
            )
            st.metric("With KD Data", f"{has_kd:,}")
        
        # Process button
        if st.button("🚀 Start Clustering", type="primary"):
            st.session_state.processing = True
            st.session_state.error = None
            
            progress_container = st.container()
            
            try:
                results = asyncio.run(
                    process_keywords(unique, config, progress_container)
                )
                st.session_state.results = results
                st.session_state.processing = False
                st.rerun()
                
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.processing = False
                st.error(f"Error: {e}")
    
    # Show results if available
    if st.session_state.results:
        render_results(st.session_state.results)
    
    # Show error if any
    if st.session_state.error:
        st.error(f"Processing failed: {st.session_state.error}")


if __name__ == "__main__":
    main()