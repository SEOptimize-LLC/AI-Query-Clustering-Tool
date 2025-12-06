# AI Keyword Clustering

An AI-powered keyword clustering application that groups keywords by semantic
similarity and search intent using embeddings, machine learning clustering,
and LLM-based labeling.

## Features

- **Semantic Clustering**: Uses OpenAI embeddings to group keywords by meaning
- **Multi-Stage Clustering**: 
  - Stage 1: Coarse clustering with MiniBatch K-Means
  - Stage 2: Fine-grained sub-clustering with HDBSCAN
  - Stage 3: SERP validation for cluster coherence
  - Stage 4: Outlier reassignment for high cluster rate
- **Consistent Labels**: Two-phase LLM labeling ensures non-overlapping labels
- **Keyword Metrics**: DataForSEO integration for search volume and difficulty
- **Interactive Visualizations**: Plotly charts for cluster analysis
- **Export Options**: CSV and Excel export with formatting

## Quick Start

### 1. Clone and Install

```bash
cd "Marketing/Roger SEO/Scripts/AI Keyword Clustering"
pip install -r requirements.txt
```

### 2. Configure API Keys

For local development, copy the secrets template:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your API keys:

```toml
[supabase]
url = "your-supabase-url"
key = "your-supabase-anon-key"

[openai]
api_key = "sk-..."

[openrouter]
api_key = "sk-or-..."

[dataforseo]
login = "your-login"
password = "your-password"

[serper]
api_key = "your-serper-key"
```

### 3. Set Up Database

Run the SQL schema in your Supabase project:

```sql
-- Enable pgvector extension
create extension if not exists vector;

-- Jobs table
create table jobs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  status text default 'pending',
  total_keywords integer,
  config jsonb,
  results jsonb
);

-- Keywords table with embeddings
create table keywords (
  id uuid primary key default gen_random_uuid(),
  job_id uuid references jobs(id) on delete cascade,
  keyword text not null,
  embedding vector(3072),
  search_volume integer default 0,
  keyword_difficulty real default 0,
  cluster_id integer,
  created_at timestamptz default now()
);

-- Clusters table
create table clusters (
  id uuid primary key default gen_random_uuid(),
  job_id uuid references jobs(id) on delete cascade,
  cluster_id integer not null,
  label text,
  intent text,
  total_volume integer default 0,
  avg_difficulty real default 0,
  quality_score real default 0,
  created_at timestamptz default now()
);

-- Cache tables
create table metrics_cache (
  keyword text primary key,
  search_volume integer,
  keyword_difficulty real,
  cached_at timestamptz default now()
);

create table serp_cache (
  keyword text primary key,
  results jsonb,
  cached_at timestamptz default now()
);

-- Indexes
create index on keywords(job_id);
create index on keywords(cluster_id);
create index on clusters(job_id);
create index on metrics_cache(cached_at);
create index on serp_cache(cached_at);
```

### 4. Run Locally

```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Connect repo in [Streamlit Cloud](https://share.streamlit.io/)
3. Add secrets in the Streamlit Cloud dashboard under Settings > Secrets
4. Deploy!

## Architecture

```
AI Keyword Clustering/
├── app.py                 # Main Streamlit application
├── config/
│   ├── settings.py        # Configuration from secrets
│   └── api_config.py      # API endpoints and constants
├── core/
│   ├── exceptions.py      # Custom exception classes
│   ├── job_manager.py     # Job state management
│   └── progress_tracker.py # Progress UI components
├── storage/
│   ├── supabase_client.py # Database operations
│   └── cache_manager.py   # Caching layer
├── ingestion/
│   ├── csv_parser.py      # CSV file parsing
│   ├── validator.py       # Keyword validation
│   └── deduplicator.py    # Duplicate removal
├── enrichment/
│   ├── dataforseo_client.py # Keyword metrics API
│   ├── serper_client.py   # SERP data API
│   └── metrics_aggregator.py # Cluster metrics
├── embedding/
│   ├── openai_embedder.py # OpenAI embedding client
│   └── batch_processor.py # Large-scale processing
├── clustering/
│   ├── pipeline.py        # Main orchestrator
│   ├── coarse_clusterer.py # MiniBatch K-Means
│   ├── fine_clusterer.py  # HDBSCAN
│   ├── serp_validator.py  # SERP overlap validation
│   └── outlier_handler.py # Outlier reassignment
├── labeling/
│   ├── prompts.py         # LLM prompt templates
│   ├── intent_classifier.py # Intent detection
│   ├── label_generator.py # Two-phase labeling
│   └── llm_client.py      # OpenRouter client
├── visualization/
│   ├── charts.py          # Plotly visualizations
│   └── metrics_display.py # Streamlit components
└── export/
    ├── csv_exporter.py    # CSV export
    └── excel_exporter.py  # Excel with formatting
```

## Key Algorithms

### Two-Phase Consistent Labeling

Solves the problem of inconsistent cluster labels:

1. **Phase 1**: Generate summaries for each cluster independently
2. **Phase 2**: Generate ALL labels in a single LLM call with global context

This ensures labels are:
- Mutually exclusive (no semantic overlap)
- Consistently formatted (same grammatical structure)
- Descriptive and specific

### Multi-Stage Clustering

Addresses the problem of semantically related keywords in different clusters:

1. **Coarse Clustering** (MiniBatch K-Means): Fast initial grouping
2. **Fine Clustering** (HDBSCAN): Density-based sub-clustering
3. **SERP Validation**: Validates clusters share ranking URLs
4. **Outlier Reassignment**: Uses cosine similarity to recover outliers

Target: >95% cluster rate (vs 50% with basic approaches)

### SERP Validation

Ensures clusters represent actual search intent groups:

- Select top 10 keywords per cluster by volume
- Fetch SERP results via Serper.dev
- Calculate URL overlap score
- Clusters with <40% overlap may be split

## API Costs

Estimated costs per 10,000 keywords:

| API | Cost |
|-----|------|
| OpenAI Embeddings | ~$0.13 |
| DataForSEO Metrics | ~$2.00 |
| Serper SERP Data | ~$1.00 |
| OpenRouter LLM | ~$0.50 |
| **Total** | **~$3.63** |

## Configuration Options

### Clustering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 5 | Minimum keywords per cluster |
| `similarity_threshold` | 0.7 | Cosine similarity for outlier reassignment |
| `serp_validation` | true | Enable SERP overlap validation |

### API Settings

| Setting | Description |
|---------|-------------|
| `location` | Target location (US, UK, etc.) |
| `language` | Target language code |

## Troubleshooting

### "Supabase connection failed"
- Check your Supabase URL and anon key in secrets
- Ensure the database schema is created

### "OpenAI rate limit"
- The embedder has built-in retry logic
- For very large files (>50K), processing is chunked

### "Low cluster rate"
- Try reducing `similarity_threshold`
- Ensure keywords are related (not random topics)
- Check for data quality issues

## License

MIT License - feel free to use and modify.

## Support

For issues or questions, please open a GitHub issue.