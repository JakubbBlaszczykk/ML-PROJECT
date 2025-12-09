# API Reference

This document provides detailed API documentation for the IMDB Search Engine classes and functions.

## MovieSearchEngine

The main entry point for search operations. Located in `src/search/engine.py`.

### Class: `MovieSearchEngine`

```python
from src.search.engine import MovieSearchEngine
```

#### Constructor

```python
MovieSearchEngine(
    bm25_path: str = None,
    sbert_embeddings_path: str = None,
    data_path: str = None,
    sbert_model_name: str = 'all-MiniLM-L6-v2'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bm25_path` | `str` | `None` | Path to BM25 model file. If `None`, uses `data/bm25_model.joblib` |
| `sbert_embeddings_path` | `str` | `None` | Path to SBERT embeddings. If `None`, uses `data/sbert_embeddings.npy` |
| `data_path` | `str` | `None` | Path to unified data file. If `None`, uses `data/imdb_us_movies_unified.parquet` |
| `sbert_model_name` | `str` | `'all-MiniLM-L6-v2'` | Sentence-BERT model name |


#### Method: `search`

```python
search(
    query: str,
    k: int = 20,
    weights: Optional[Dict] = None
) -> pd.DataFrame
```

Performs a search query and returns ranked results.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | The search query string |
| `k` | `int` | `20` | Number of results to return |
| `weights` | `Dict` | `None` | Custom weight overrides. If `None`, uses config defaults |

**Returns:**

`pd.DataFrame` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `title` | `str` | Movie title |
| `startYear` | `int` | Release year |
| `genres` | `str` | Comma-separated genres |
| `actor_name` | `str` | Comma-separated actor names |
| `director_name` | `str` | Comma-separated director names |
| `averageRating` | `float` | IMDB rating |
| `numVotes` | `int` | Number of votes |
| `final_score` | `float` | Combined ranking score |
| `hybrid_score` | `float` | BM25 + SBERT score |
| `filter_boost` | `float` | Boost from filter matches |
| `actor_match_score` | `float` | Actor match contribution |
| `director_match_score` | `float` | Director match contribution |
| `genre_match_score` | `float` | Genre match contribution |
| `year_match_score` | `float` | Year match contribution |


---

## QueryParser

Parses user queries to extract structured filters. Located in `src/search/query_parser.py`.

### Class: `QueryParser`

```python
from src.search.query_parser import QueryParser
```

#### Constructor

```python
QueryParser(
    data_path: str = None,
    force_rebuild: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | `None` | Path to parquet file with movie data |
| `force_rebuild` | `bool` | `False` | If `True`, rebuilds lookup sets even if cached |

#### Method: `parse_query`

```python
parse_query(query: str) -> Dict
```

Parses a query string and extracts filters.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Raw user query |

**Returns:**

`Dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `original_query` | `str` | The original query string |
| `cleaned_query` | `str` | Query with filter tokens removed |
| `corrected_search_term` | `str` | Query with fuzzy corrections applied |
| `actor_filter` | `List[str]` | Detected actor names |
| `director_filter` | `List[str]` | Detected director names |
| `writer_filter` | `List[str]` | Detected writer names |
| `genre_filter` | `List[str]` | Detected genres |
| `year_filter` | `Dict` | Year constraints with `min` and/or `max` keys |
| `actor_or_director` | `List[str]` | Ambiguous person names (could be actor or director) |
| `actor_or_writer` | `List[str]` | Ambiguous person names (could be actor or writer) |
| `director_or_writer` | `List[str]` | Ambiguous person names (could be director or writer) |
| `fuzzy_corrections` | `Dict` | Map of original tokens to corrected versions |

---

## HybridSearcher

Combines BM25 and SBERT for retrieval. Located in `src/search/hybrid_search.py`.

### Class: `HybridSearcher`

```python
from src.search.hybrid_search import HybridSearcher
```

#### Constructor

```python
HybridSearcher(
    bm25_path: str = None,
    sbert_embeddings_path: str = None,
    data_path: str = None,
    sbert_model_name: str = 'all-MiniLM-L6-v2'
)
```

**Parameters:**

Same as `MovieSearchEngine` constructor.

#### Method: `search`

```python
search(
    query: str,
    k: int = 60,
    rrf_k: int = 60,
    parsed_query: Dict = None
) -> pd.DataFrame
```

Performs hybrid retrieval using BM25 and SBERT.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `k` | `int` | `60` | Number of candidates to retrieve |
| `rrf_k` | `int` | `60` | RRF constant for score fusion |
| `parsed_query` | `Dict` | `None` | Pre-parsed query from `QueryParser` |

**Returns:**

`pd.DataFrame` with movie data and `rrf_score` column.

---

## SearchCorpusGenerator

Generates searchable text from movie data. Located in `src/data/custom_transformers.py`.

### Class: `SearchCorpusGenerator`

A scikit-learn compatible transformer for generating search corpus.

```python
from src.data.custom_transformers import SearchCorpusGenerator
```

#### Methods

##### `fit(X, y=None)`

No-op fit method for sklearn compatibility.

##### `transform(X)`

Transforms a DataFrame to generate searchable text.

**Input Columns:**

- `remainder__cast`: List of cast member dicts
- `remainder__directors`: List of director dicts
- `remainder__writers`: List of writer dicts
- `cat__genres`: Comma-separated genres
- `cat__title`: Movie title

**Output Columns:**

- `searchable_text`: Combined text for search indexing
- `normalized_title`: Stemmed, lowercased title
- `cat__title`: Original title

##### `_normalize_text(text)`

Normalizes text by lowercasing, removing punctuation, and stemming.

```python
generator = SearchCorpusGenerator()
normalized = generator._normalize_text("The Dark Knight")
# Returns: "dark knight"
```

---

## Configuration Functions

Configuration utilities in `src/config/config.py`.

### Function: `get_config`

```python
from src.config.config import get_config

config = get_config()
```

Returns the global `SearchConfig` instance. Loads from `search_config.ini` on first call.

**Returns:**

`SearchConfig` instance with the following attributes:

- `weights`: Ranking weight parameters
- `query_parser`: Query parsing settings
- `hybrid_search`: Retrieval parameters
- `title_matching`: Title boost settings
- `year_matching`: Year filter settings
- `fuzzy_matching`: Typo tolerance settings
- `results`: Output settings
- `metadata`: Display preferences

### Function: `reload_config`

```python
from src.config.config import reload_config

config = reload_config()
```

Reloads configuration from file, useful for picking up changes without restarting.

---

## Utility Scripts

### `scripts/build_bm25.py`

Builds the BM25 index from unified data.

```bash
python scripts/build_bm25.py
```

**Output:** `data/bm25_model.joblib`

### `scripts/build_sbert.py`

Generates SBERT embeddings for all documents.

```bash
python scripts/build_sbert.py
```

**Output:** `data/sbert_embeddings.npy`

### `scripts/run_pipeline.py`

Runs the full preprocessing pipeline on raw data.

```bash
python scripts/run_pipeline.py
```

---

## Data Types

### Year Filter Structure

```python
{
    'min': int,  # Minimum year (inclusive)
    'max': int,  # Maximum year (inclusive)
}
```

Examples:
- `{'min': 1990, 'max': 1999}` – 1990s decade
- `{'min': 2000}` – After 2000
- `{'max': 1980}` – Before 1980
- `{'min': 1999, 'max': 1999}` – Exact year

### Parsed Query Structure

```python
{
    'original_query': str,
    'cleaned_query': str,
    'corrected_search_term': str,
    'actor_filter': List[str],
    'director_filter': List[str],
    'writer_filter': List[str],
    'genre_filter': List[str],
    'year_filter': Dict[str, int],
    'actor_or_director': List[str],
    'actor_or_writer': List[str],
    'director_or_writer': List[str],
    'fuzzy_corrections': Dict[str, str],
}
```
