# Architecture Overview

This document describes the system architecture of the IMDB Search Engine, covering its major components and how they interact during query processing.

## High-Level Architecture

The search engine follows a multi-stage retrieval and ranking pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Query                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QueryParser                                       │
│  • Extract actors, directors, writers (exact + fuzzy matching)              │
│  • Extract genres (with aliases like "scifi" → "sci-fi")                    │
│  • Extract year filters (exact years, decades, ranges)                      │
│  • Produce corrected search term for downstream search                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HybridSearcher                                      │
│  • BM25 keyword retrieval (top-k candidates)                                │
│  • SBERT semantic retrieval (top-k candidates)                              │
│  • Reciprocal Rank Fusion (RRF) to merge both rankings                      │
│  • Title match boost using context-aware Jaccard similarity                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MovieSearchEngine                                    │
│  • Compute weighted scores for candidates                                   │
│  • Apply filter boosts (actor, director, writer, genre, year)               │
│  • Combine hybrid_score + filter_boost + popularity + rating                │
│  • Return final ranked results                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Search Results                                     │
│  • Ranked movies with scores and metadata                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### QueryParser (`src/search/query_parser.py`)

The QueryParser is responsible for understanding user intent by analyzing the raw query string and extracting structured filters.

**Key Responsibilities:**

1. **Person Extraction**: Identifies actor, director, and writer names using:
   - Trie-based prefix matching for exact name lookup
   - Fuzzy matching via RapidFuzz for typo tolerance
   - Contextual keywords ("directed by", "starring", "written by")

2. **Genre Extraction**: Matches genre terms using:
   - Canonical genre matching from the dataset
   - Genre aliases (e.g., "scifi" → "sci-fi", "romcom" → "romance")
   - Fuzzy genre matching for typos

3. **Year Extraction**: Parses various year formats:
   - Exact years ("1999")
   - Decades ("90s", "80s")
   - Year ranges ("after 2000", "before 1980")

4. **Role Disambiguation**: Handles cases where a person may have multiple roles (actor/director) by using popularity weighting.

**Data Structures:**

- `Trie`: Used for efficient prefix-based name lookup
- Role counts (`actor_counts`, `director_counts`, `writer_counts`): Track person appearances per role
- `all_names_set`: Set of all known person names for quick lookup
- `popular_titles`: Top N movie titles for fuzzy title matching

### HybridSearcher (`src/search/hybrid_search.py`)

The HybridSearcher combines two retrieval strategies to capture both lexical and semantic relevance.

**Retrieval Strategies:**

1. **BM25 (Keyword Search)**
   - Uses the `rank_bm25` library with Okapi BM25 algorithm
   - Searches against normalized `searchable_text` (stemmed, lowercased, no punctuation)
   - Returns top-k candidates with BM25 scores

2. **SBERT (Semantic Search)**
   - Uses Sentence-BERT (`all-MiniLM-L6-v2`) embeddings
   - Computes dot product similarity between query and document embeddings
   - Returns top-k candidates based on semantic similarity

**Score Fusion:**

The system uses Reciprocal Rank Fusion (RRF) to combine rankings:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where `rank_i(d)` is the rank of document `d` in ranking `i`, and `k` is a constant (default: 60).

**Title Boost:**

After RRF, titles are boosted using context-aware Jaccard similarity:
- Calculates token overlap between query and title
- Excludes detected filter tokens (genres, actor names) from similarity calculation
- Applies configurable boosts for exact, partial, and prefix matches

### MovieSearchEngine (`src/search/engine.py`)

The MovieSearchEngine orchestrates the entire search process and applies the final ranking formula.

**Weighted Scoring Formula:**

```
final_score = W_hybrid × hybrid_score
            + W_popularity × popularity_score
            + W_rating × rating_score
            + W_actor × actor_match
            + W_director × director_match
            + W_writer × writer_match
            + W_genre × genre_match
            + W_year × year_match
```

**Filter Match Computation:**

Each filter match function returns a score between 0.0 and 1.0:

- `_compute_actor_match`: Checks if detected actors appear in the movie's cast
- `_compute_director_match`: Checks if detected directors match movie's directors
- `_compute_writer_match`: Checks if detected writers match movie's writers
- `_compute_genre_match`: Computes overlap between detected genres and movie's genres
- `_compute_year_match`: Returns 1.0 for exact match, 0.5 for within tolerance, 0.0 otherwise

**Score Normalization:**

- Popularity score is derived from `numVotes_log` (pre-computed log transform)
- Rating score comes from `averageRating_scaled` (standardized)
- Both are normalized using precomputed min/max statistics

## Data Layer

### Unified Data File (`data/imdb_us_movies_unified.parquet`)

The unified data file merges processed features with raw metadata:

| Column | Description |
|--------|-------------|
| `tconst` | IMDB title identifier |
| `primaryTitle` | Movie title |
| `startYear` | Release year |
| `genres` | Comma-separated genre list |
| `actor_name` | Comma-separated actor names |
| `director_name` | Comma-separated director names |
| `writer_name` | Comma-separated writer names |
| `searchable_text` | Combined text for BM25/SBERT |
| `normalized_title` | Stemmed, lowercased title |
| `numVotes` | Vote count (popularity) |
| `averageRating` | IMDB rating |
| `numVotes_log` | Log-transformed vote count |
| `averageRating_scaled` | Standardized rating |

### Search Indexes

1. **BM25 Model** (`data/bm25_model.joblib`)
   - Serialized `BM25Okapi` instance from `rank_bm25`
   - Built from tokenized, normalized `searchable_text`

2. **SBERT Embeddings** (`data/sbert_embeddings.npy`)
   - NumPy array of shape `(n_documents, 384)`
   - Generated using `all-MiniLM-L6-v2` model

## Configuration

All tunable parameters are externalized in `src/config/search_config.ini`:

- **Weights**: Control the contribution of each scoring component
- **Query Parser**: Name matching thresholds and token requirements
- **Hybrid Search**: Candidate retrieval parameters and RRF settings
- **Fuzzy Matching**: Similarity thresholds for typo tolerance
