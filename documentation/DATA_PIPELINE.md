# Data Pipeline

This document describes the data processing pipeline, from raw IMDB data to search-ready indexes.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Raw IMDB Data                                │
│  title.basics.tsv, title.ratings.tsv, title.principals.tsv, etc.   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    cleaning.ipynb                                   │
│  • Filter to US movies only                                        │
│  • Convert placeholders to nulls                                    │
│  • Correct data types                                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                      imdb_us_movies_cleaned.parquet
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Merging.ipynb                                    │
│  • Join titles with ratings                                        │
│  • Parse principals into cast/crew structures                      │
│  • Extract actor/director/writer names                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                      imdb_us_movies_merged.parquet
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  preprocessing_pipeline.ipynb                       │
│  • Generate search corpus (searchable_text)                        │
│  • Create ranking features (log votes, scaled ratings)             │
│  • Feature binning (decades, runtime bins)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                      imdb_us_movies_processed.parquet
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  create_unified_data.py                             │
│  • Merge processed with metadata                                   │
│  • Deduplicate by tconst                                           │
│  • Extract flat name columns                                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                      imdb_us_movies_unified.parquet
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│        build_bm25.py          │   │       build_sbert.py          │
│  • Tokenize searchable_text   │   │  • Encode with SBERT model    │
│  • Build BM25Okapi index      │   │  • Save embedding vectors     │
└───────────────────────────────┘   └───────────────────────────────┘
          │                                   │
    bm25_model.joblib                sbert_embeddings.npy
```

## Data Cleaning (cleaning.ipynb)

The cleaning notebook processes raw IMDB TSV files into a clean parquet format.

### Input Files

| File | Description |
|------|-------------|
| `title.basics.tsv` | Title ID, type, title, year, runtime, genres |
| `title.ratings.tsv` | Average rating, number of votes |

### Processing Steps

1. **Filter by Region**: Keep only US movies (`isAdult = 0`, `titleType = 'movie'`)

2. **Placeholder Conversion**: Replace IMDB placeholders with nulls:
   - `\\N` → `None`
   - `-1.0` → `NaN` (for numeric columns)
   - `"missing"` → `None` (for categorical columns)

3. **Type Correction**:
   - `startYear` → nullable integer
   - `runtimeMinutes` → float
   - `genres` → string

4. **Column Prefixing**: Add type prefixes for downstream processing:
   - `num__` for numeric columns
   - `cat__` for categorical columns
   - `remainder__` for complex types

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `cat__tconst` | string | IMDB title identifier |
| `cat__title` | string | Primary title |
| `num__startYear` | Int64 | Release year |
| `num__runtimeMinutes` | float | Runtime in minutes |
| `cat__genres` | string | Comma-separated genres |
| `num__averageRating` | float | IMDB rating (0-10) |
| `num__numVotes` | float | Number of votes |

---

## Data Merging (Merging.ipynb)

The merging notebook joins title data with cast/crew information.

### Input Files

| File | Description |
|------|-------------|
| `imdb_us_movies_cleaned.parquet` | Cleaned title data |
| `title.principals.tsv` | Cast and crew per title |
| `name.basics.tsv` | Person information |

### Processing Steps

1. **Parse Principals**: Extract cast and crew from the principals table:
   - Filter by category: `actor`, `actress`, `director`, `writer`
   - Group by title ID

2. **Structure Creation**: Create nested structures for each role:
   ```python
   {
       'nconst': 'nm0000001',
       'primaryName': 'Tom Hanks',
       'category': 'actor'
   }
   ```

3. **Join with Titles**: Merge cast/crew with title metadata

### Output Schema Additions

| Column | Type | Description |
|--------|------|-------------|
| `cast` | List[Dict] | List of cast member structs |
| `directors` | List[Dict] | List of director structs |
| `writers` | List[Dict] | List of writer structs |

---

## Feature Engineering (preprocessing_pipeline.ipynb)

The preprocessing notebook generates features for search and ranking.

### Custom Transformers

The pipeline uses scikit-learn compatible transformers from `src/data/custom_transformers.py`:

#### PlaceholderToNullTransformer

Converts remaining placeholders to proper nulls.

```python
transformer = PlaceholderToNullTransformer(
    numeric_cols=['num__runtimeMinutes'],
    categorical_cols=['cat__genres']
)
```

#### TypeCorrector

Ensures correct data types after transformations.

```python
transformer = TypeCorrector(
    int_cols=['startYear'],
    float_cols=['averageRating'],
    obj_cols=['genres']
)
```

#### SearchCorpusGenerator

Generates the searchable text corpus by concatenating:
- Title
- Genres (comma → space)
- Cast names
- Director names
- Writer names

The generator also:
- Normalizes text (lowercase, remove punctuation)
- Applies Porter stemming

```python
generator = SearchCorpusGenerator()
result = generator.transform(df)
# Output columns: searchable_text, normalized_title, cat__title
```

#### NumericalRankingTransformer

Prepares numeric features for ranking:
- Log transform for skewed features (`numVotes`)
- Outlier clipping at 99th percentile
- Standard scaling

```python
transformer = NumericalRankingTransformer(
    log_cols=['num__numVotes'],
    clip_cols=['num__runtimeMinutes'],
    clip_quantile=0.99
)
```

#### FeatureBinner

Creates categorical bins for user-facing filters:

| Feature | Bins |
|---------|------|
| `startYear` | Decades (1950s, 1960s, ..., 2020s) |
| `runtimeMinutes` | <60 min, 60-90 min, 90-120 min, 120+ min |
| `averageRating` | Low (0-4), Medium (4-6), High (6-8), Excellent (8+) |

#### GenresBinarizer

Converts comma-separated genres to multi-label binary format.

```python
binarizer = GenresBinarizer()
# Input: "Action,Drama,Thriller"
# Output: [1, 0, 1, 0, 0, 1, 0, ...]  # One column per genre
```

### Output Schema Additions

| Column | Type | Description |
|--------|------|-------------|
| `searchable_text` | string | Combined search corpus |
| `normalized_title` | string | Stemmed, lowercased title |
| `numVotes_log` | float | Log-transformed vote count |
| `averageRating_scaled` | float | Standardized rating |

---

## Unified Data Creation (create_unified_data.py)

This script merges processed and merged data into a single optimized file.

### Processing Steps

1. **Load Data**: Read both parquet files
2. **Deduplicate**: Remove duplicate `tconst` entries (keep first)
3. **Extract Names**: Flatten nested structures to comma-separated strings:
   - `cast` → `actor_name`
   - `directors` → `director_name`
   - `writers` → `writer_name`
4. **Merge**: Join on `tconst`
5. **Save**: Write to parquet format

### Final Schema

| Column | Type | Description |
|--------|------|-------------|
| `tconst` | string | IMDB identifier |
| `primaryTitle` | string | Movie title |
| `startYear` | int | Release year |
| `genres` | string | Comma-separated genres |
| `actor_name` | string | Comma-separated actors |
| `director_name` | string | Comma-separated directors |
| `writer_name` | string | Comma-separated writers |
| `numVotes` | int | Raw vote count |
| `averageRating` | float | IMDB rating |
| `runtimeMinutes` | int | Runtime |
| `searchable_text` | string | Combined search text |
| `normalized_title` | string | Stemmed title |
| `numVotes_log` | float | Log votes |
| `averageRating_scaled` | float | Scaled rating |

---

## Index Building

### BM25 Index (build_bm25.py)

Builds a BM25Okapi index for keyword search.

**Process:**

1. Load unified data
2. Tokenize `searchable_text` using `SearchCorpusGenerator._normalize_text()`
3. Build `BM25Okapi` index from token lists
4. Serialize with joblib

**Output:** `data/bm25_model.joblib` (~100 MB)

### SBERT Embeddings (build_sbert.py)

Generates dense embeddings for semantic search.

**Process:**

1. Load unified data
2. Load SBERT model (`all-MiniLM-L6-v2`)
3. Encode `searchable_text` in batches
4. Save embeddings as NumPy array

**Output:** `data/sbert_embeddings.npy` (~500 MB)

**Embedding Dimensions:** 384 (for all-MiniLM-L6-v2)

---

## Data Statistics

Typical dataset size after processing:

| Metric | Value |
|--------|-------|
| Total movies | ~338,000 |
| Unique actors | ~500,000 |
| Unique directors | ~100,000 |
| Unique genres | 28 |
| Year range | 1894–2024 |

---

## Rebuilding Indexes

After modifying the data or preprocessing:

```bash
# 1. Regenerate unified data
python src/data/create_unified_data.py

# 2. Rebuild BM25 index
python scripts/build_bm25.py

# 3. Rebuild SBERT embeddings (takes 5-15 minutes)
python scripts/build_sbert.py
```

The QueryParser also caches lookup tables. To force rebuild:

```python
from src.search.query_parser import QueryParser
parser = QueryParser(force_rebuild=True)
```
