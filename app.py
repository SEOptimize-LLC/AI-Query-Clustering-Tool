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
from io import BytesIO

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
from core.job_manager import JobManager, JobConfig
from core.progress_tracker import ProgressTracker
from storage.supabase_client import SupabaseClient
from storage.cache_manager import CacheManager
from ingestion.csv_parser import CSVParser
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
    defaults = {
        "job_id": None,
        "results": None,
        "processing": False,
        "fetching_metrics": False,
        "error": None,
        "csv_preview": None,
        "csv_parser": None,
        "keywords_raw": None,
        "keywords_validated": None,
        "keywords_enriched": None,
        "metrics_fetched": False,
        "step": "upload"  # upload, validate, enrich, cluster, results
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    st.markdown("## 📤 Step 1: Upload Keywords")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file with keywords",
            type=["csv", "xlsx", "xls"],
            help="Any CSV/Excel file - you'll select the keyword column"
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
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if (st.session_state.csv_preview is None or
                st.session_state.get("file_key") != file_key):
            try:
                parser = CSVParser()
                content = BytesIO(uploaded_file.read())
                preview = parser.preview(
                    content,
                    filename=uploaded_file.name
                )
                st.session_state.csv_preview = preview
                st.session_state.csv_parser = parser
                st.session_state.file_key = file_key
                st.session_state.keywords_raw = None
                st.session_state.keywords_validated = None
                st.session_state.keywords_enriched = None
                st.session_state.metrics_fetched = False
                st.session_state.step = "upload"
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None
        
        preview = st.session_state.csv_preview
        parser = st.session_state.csv_parser
        
        st.success(
            f"📄 Loaded **{preview['row_count']:,}** rows "
            f"with **{len(preview['columns'])}** columns"
        )
        
        # Column selection UI
        st.markdown("### Select Columns")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
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
                    st.session_state.keywords_raw = keywords_data
                    st.session_state.step = "validate"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing keywords: {e}")
        else:
            st.warning("⚠️ Please select a keyword column")
    else:
        # Reset state when no file
        st.session_state.csv_preview = None
        st.session_state.csv_parser = None
        st.session_state.keywords_raw = None
        st.session_state.file_key = None


def render_validation_section():
    """Render validation and deduplication section."""
    st.markdown("## 🔍 Step 2: Validation")
    
    keywords_data = st.session_state.keywords_raw
    
    if keywords_data is None:
        st.info("Upload keywords first")
        return
    
    # Validate
    validator = KeywordValidator()
    valid, invalid = validator.validate_batch(keywords_data)
    
    if invalid:
        st.warning(f"⚠️ {len(invalid)} invalid keywords removed")
    
    # Deduplicate
    dedup = KeywordDeduplicator()
    unique = dedup.deduplicate(valid)
    
    if len(unique) < len(valid):
        st.info(f"🔄 Removed {len(valid) - len(unique)} duplicates")
    
    # Store validated keywords
    st.session_state.keywords_validated = unique
    
    # Show summary
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
    
    # Check if metrics already in file
    if has_vol == len(unique):
        st.success("✅ All keywords have search volume data from file")
        st.session_state.keywords_enriched = unique
        st.session_state.metrics_fetched = True
    
    st.markdown("---")


async def fetch_metrics_async(keywords_data: list, config: dict, progress_bar):
    """Fetch metrics from DataForSEO."""
    settings = Settings()
    
    supabase = SupabaseClient(settings.supabase_url, settings.supabase_key)
    cache = CacheManager(supabase)
    
    keywords_text = [kw["keyword"] for kw in keywords_data]
    
    # Check cache first
    cached_metrics = cache.get_keyword_metrics(keywords_text)
    uncached = [kw for kw in keywords_text if kw not in cached_metrics]
    
    progress_bar.progress(0.1, text=f"Found {len(cached_metrics)} cached")
    
    if uncached:
        dataforseo = DataForSEOClient(
            login=settings.dataforseo_login,
            password=settings.dataforseo_password
        )
        
        location_code = LOCATION_CODES.get(config["location"], 2840)
        language_code = LANGUAGE_CODES.get(config["language"], "en")
        
        def update_progress(pct):
            progress_bar.progress(
                0.1 + pct * 0.8,
                text=f"Fetching metrics: {int(pct*100)}%"
            )
        
        new_metrics = await dataforseo.get_keyword_metrics(
            keywords=uncached,
            location_code=location_code,
            language_code=language_code,
            progress_callback=update_progress
        )
        
        # Cache new metrics
        cache.save_keyword_metrics(new_metrics)
        cached_metrics.update(new_metrics)
    
    progress_bar.progress(0.95, text="Merging data...")
    
    # Merge metrics into keyword data
    enriched = []
    for kw in keywords_data:
        kw_copy = kw.copy()
        if kw["keyword"] in cached_metrics:
            metrics = cached_metrics[kw["keyword"]]
            kw_copy["search_volume"] = metrics.get("search_volume", 0)
            kw_copy["keyword_difficulty"] = metrics.get(
                "keyword_difficulty", 0
            )
        enriched.append(kw_copy)
    
    progress_bar.progress(1.0, text="Complete!")
    
    return enriched


def render_enrichment_section(config: dict):
    """Render metrics enrichment section."""
    st.markdown("## 📊 Step 3: Fetch Metrics (Optional)")
    
    keywords = st.session_state.keywords_validated
    
    if keywords is None:
        st.info("Validate keywords first")
        return
    
    # Check if already enriched
    if st.session_state.metrics_fetched:
        st.success("✅ Metrics already fetched!")
        
        # Show enriched data summary
        enriched = st.session_state.keywords_enriched or keywords
        total_vol = sum(kw.get("search_volume", 0) for kw in enriched)
        avg_kd = (
            sum(kw.get("keyword_difficulty", 0) for kw in enriched) /
            len(enriched) if enriched else 0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Search Volume", f"{total_vol:,}")
        with col2:
            st.metric("Average KD", f"{avg_kd:.1f}")
        
        # Show top keywords by volume
        with st.expander("📊 Top Keywords by Volume"):
            sorted_kw = sorted(
                enriched,
                key=lambda x: x.get("search_volume", 0),
                reverse=True
            )[:20]
            
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Keyword": kw["keyword"],
                    "Volume": kw.get("search_volume", 0),
                    "KD": kw.get("keyword_difficulty", 0)
                }
                for kw in sorted_kw
            ])
            st.dataframe(df, use_container_width=True)
    else:
        st.info(
            "📡 Fetch search volume and keyword difficulty from DataForSEO. "
            "This step is optional if your file already contains this data."
        )
        
        # Estimate cost
        uncached_estimate = len(keywords)  # Assume all uncached for estimate
        cost_estimate = (uncached_estimate / 1000) * 0.05  # $0.05 per 1000
        
        st.caption(
            f"Estimated cost: ~${cost_estimate:.2f} for {len(keywords):,} keywords"
        )
        
        if st.button("📊 Fetch Metrics from DataForSEO", type="secondary"):
            st.session_state.fetching_metrics = True
            progress_bar = st.progress(0, text="Starting...")
            
            try:
                enriched = asyncio.run(
                    fetch_metrics_async(keywords, config, progress_bar)
                )
                st.session_state.keywords_enriched = enriched
                st.session_state.metrics_fetched = True
                st.session_state.fetching_metrics = False
                st.rerun()
            except Exception as e:
                st.error(f"Error fetching metrics: {e}")
                st.session_state.fetching_metrics = False
        
        # Allow skipping
        if st.button("⏭️ Skip - Use Existing Data", type="secondary"):
            st.session_state.keywords_enriched = keywords
            st.session_state.metrics_fetched = True
            st.rerun()
    
    st.markdown("---")


async def run_clustering(
    keywords_data: list,
    config: dict,
    progress_container
):
    """
    Run the clustering pipeline (embeddings + clustering + labeling).
    
    Args:
        keywords_data: List of keyword dictionaries with metrics
        config: Processing configuration
        progress_container: Streamlit container for progress
    """
    settings = Settings()
    start_time = time.time()
    
    # Initialize components
    supabase = SupabaseClient(settings.supabase_url, settings.supabase_key)
    cache = CacheManager(supabase)
    
    keywords_text = [kw["keyword"] for kw in keywords_data]
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    st.session_state.job_id = job_id
    
    # Simple progress tracker
    progress_bar = progress_container.progress(0, text="Starting...")
    status_text = progress_container.empty()
    
    try:
        # Stage 1: Generate embeddings
        progress_bar.progress(0.05, text="Generating embeddings...")
        status_text.markdown("🧠 **Embedding keywords...**")
        
        embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
        batch_processor = BatchEmbeddingProcessor(embedder, supabase)
        
        def embedding_progress(p):
            pct = 0.05 + p.progress * 0.35
            progress_bar.progress(pct, text=f"Embedding: {int(p.progress*100)}%")
        
        embeddings_dict = batch_processor.process_keywords(
            keywords=keywords_text,
            job_id=job_id,
            progress_callback=embedding_progress
        )
        
        # Convert dict to ordered numpy array
        import numpy as np
        embeddings = np.array([
            embeddings_dict[kw] for kw in keywords_text
        ])
        
        # Stage 2: Clustering
        progress_bar.progress(0.45, text="Clustering keywords...")
        status_text.markdown("🎯 **Clustering keywords...**")
        
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
        
        kw_data = {kw["keyword"]: kw for kw in keywords_data}
        
        def cluster_progress(stage, pct, msg):
            base = 0.45
            progress_bar.progress(
                base + pct * 0.25,
                text=f"Clustering: {msg}"
            )
        
        cluster_result = pipeline.run(
            embeddings=embeddings,
            keywords=keywords_text,
            keyword_data=kw_data,
            progress_callback=cluster_progress,
            skip_serp_validation=not config["serp_validation"]
        )
        
        # Stage 3: Label generation
        progress_bar.progress(0.75, text="Generating labels...")
        status_text.markdown("🏷️ **Generating cluster labels...**")
        
        llm = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            default_model=config.get("llm_model", "anthropic/claude-sonnet-4")
        )
        intent_classifier = IntentClassifier()
        label_gen = LabelGenerator(llm, intent_classifier)
        
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
        
        def label_progress(stage, pct, msg):
            progress_bar.progress(
                0.75 + pct * 0.2,
                text=f"Labeling: {msg}"
            )
        
        label_result = await label_gen.generate_labels(
            clusters=cluster_data,
            progress_callback=label_progress
        )
        
        # Stage 4: Aggregate and build results
        progress_bar.progress(0.95, text="Finalizing...")
        status_text.markdown("📊 **Building results...**")
        
        aggregator = MetricsAggregator()
        
        final_clusters = []
        for cid, info in cluster_result.clusters.items():
            label = label_result.labels.get(cid)
            
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
        
        progress_bar.progress(1.0, text="Complete!")
        status_text.markdown("✅ **Clustering complete!**")
        
        return results
        
    except Exception as e:
        status_text.markdown(f"❌ **Error:** {e}")
        raise


def render_clustering_section(config: dict):
    """Render the clustering section."""
    st.markdown("## 🎯 Step 4: Cluster Keywords")
    
    keywords = st.session_state.keywords_enriched
    
    if keywords is None:
        st.info("Complete previous steps first")
        return
    
    st.info(
        f"Ready to cluster **{len(keywords):,}** keywords. "
        "This will generate embeddings, cluster by semantic similarity, "
        "and create consistent labels using AI."
    )
    
    # Estimate time/cost
    embed_cost = (len(keywords) / 1000) * 0.00013 * 3072  # $0.00013 per 1K tokens
    llm_cost = 0.50  # Rough estimate for labeling
    
    st.caption(
        f"Estimated cost: ~${embed_cost + llm_cost:.2f} | "
        f"Time: ~{len(keywords) // 100 + 1} minutes"
    )
    
    if st.button("🚀 Start Clustering", type="primary"):
        st.session_state.processing = True
        st.session_state.error = None
        
        progress_container = st.container()
        
        try:
            results = asyncio.run(
                run_clustering(keywords, config, progress_container)
            )
            st.session_state.results = results
            st.session_state.processing = False
            st.session_state.step = "results"
            st.rerun()
            
        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.processing = False
            st.error(f"Error: {e}")


def render_results(results: dict):
    """Render clustering results."""
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
        
        # Distribution chart
        st.markdown("#### Cluster Size Distribution")
        sizes = {c["id"]: c["size"] for c in results["clusters"]}
        names = {c["id"]: c["label"] for c in results["clusters"]}
        fig = create_cluster_distribution(sizes, names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume treemap
        st.markdown("#### Volume Distribution")
        fig = create_volume_treemap(
            [{"cluster_id": c["id"], "volume": c["total_volume"],
              "size": c["size"]}
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
    
    # Show workflow steps
    if st.session_state.results:
        render_results(st.session_state.results)
        
        # Reset button
        if st.button("🔄 Start New Analysis"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        # Step 1: Upload
        render_upload_section()
        
        # Step 2: Validation (if we have raw keywords)
        if st.session_state.keywords_raw:
            render_validation_section()
            
            # Step 3: Enrichment (if validated)
            if st.session_state.keywords_validated:
                render_enrichment_section(config)
                
                # Step 4: Clustering (if enriched)
                if st.session_state.metrics_fetched:
                    render_clustering_section(config)
    
    # Show error if any
    if st.session_state.error:
        st.error(f"Processing failed: {st.session_state.error}")


if __name__ == "__main__":
    main()