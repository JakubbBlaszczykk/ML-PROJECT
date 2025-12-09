# Configuration Guide

This document describes all configurable parameters in the search engine. Configuration is stored in `src/config/search_config.ini` and loaded at runtime.

## Configuration File Location

```
src/config/search_config.ini
```

## Loading Configuration

```python
from src.config.config import get_config, reload_config

# Get configuration (loads once, then cached)
config = get_config()

# Reload configuration from file
config = reload_config()
```

## Configuration Sections

### Weights

Controls how different scoring components contribute to the final ranking score.

```ini
[weights]
W_hybrid = 1.0
W_popularity = 0.7
W_rating = 0.1
W_actor = 0.9
W_director = 0.9
W_writer = 0.5
W_genre = 0.7
W_year = 0.7
```

**Scoring Formula:**

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W_hybrid` | `1.0` | Weight for BM25+SBERT hybrid score. Keep at 1.0 as anchor. |
| `W_popularity` | `0.7` | Weight for movie popularity (based on vote count). |
| `W_rating` | `0.1` | Weight for IMDB rating. |
| `W_actor` | `0.9` | Boost when detected actor is in movie cast. |
| `W_director` | `0.9` | Boost when detected director matches movie. |
| `W_writer` | `0.5` | Boost when detected writer matches movie. |
| `W_genre` | `0.7` | Boost when detected genre matches movie genres. |
| `W_year` | `0.7` | Boost when year filter matches movie release year. |

**Tuning Guidance:**

- Higher `W_hybrid` emphasizes text relevance (exact title matches)
- Higher `W_actor`/`W_director` prioritizes filter matches over text relevance
- Keep filter weights moderate (0.3–0.9) to avoid overpowering exact title matches

---

### Query Parser Settings

Controls how the query parser identifies and extracts filters.

```ini
[query_parser]
min_name_words = 2
min_single_word_length = 7
strict_token_matching = True
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_name_words` | `2` | Minimum words for a multi-word person name match. Prevents false positives like "Tom" alone matching. |
| `min_single_word_length` | `7` | Minimum character length for single-word person names (e.g., "Hitchcock" at 10 chars matches). |
| `strict_token_matching` | `True` | If set, tokens already matched are excluded from further matching. |

**Example:**

With `min_name_words = 2`:
- "Tom Hanks" (2 words) → matches
- "Tom" (1 word, 3 chars) → does not match

---

### Hybrid Search Settings

Controls the candidate retrieval phase.

```ini
[hybrid_search]
candidate_multiplier = 5
min_candidates = 1000
rrf_k = 60
rank_decay_factor = 0.08
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `candidate_multiplier` | `5` | Multiplier for requested results to get candidates. If asking for 20 results, retrieves 100 candidates. |
| `min_candidates` | `1000` | Minimum number of candidates to retrieve regardless of k. |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant. Higher values smooth out rank differences. |
| `rank_decay_factor` | `0.08` | Deprecated. Now using RRF instead of rank decay. |

**RRF Formula:**

```
RRF_score(d) = Σ 1/(rrf_k + rank(d))
```

---

### Title Matching Settings

Controls title similarity boosting using Jaccard similarity.

```ini
[title_matching]
jaccard_threshold = 0.6
exact_title_boost = 0.9
partial_title_boost = 0.5
prefix_title_boost = 0.25
generic_movie_terms = movie,movies,film,films,cinema,picture,flick
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jaccard_threshold` | `0.6` | Minimum Jaccard similarity for partial title boost. |
| `exact_title_boost` | `0.9` | Boost when query tokens exactly match title tokens (Jaccard = 1.0). |
| `partial_title_boost` | `0.5` | Boost when Jaccard similarity exceeds threshold. |
| `prefix_title_boost` | `0.25` | Boost when title starts with query (e.g., "Avatar" matches "Avatar: The Way of Water"). |
| `generic_movie_terms` | see above | Comma-separated terms excluded from Jaccard calculation. |

**Jaccard Similarity:**

```
Jaccard = |query_tokens ∩ title_tokens| / |query_tokens ∪ title_tokens|
```

The calculation excludes:
- Generic movie terms (movie, film, cinema, etc.)
- Detected filter tokens (actor names, genres, etc.)

---

### Year Matching Settings

Controls year filter behavior.

```ini
[year_matching]
year_tolerance = 5
exact_match_score = 1.0
near_match_score = 0.5
no_match_score = 0.0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `year_tolerance` | `5` | Movies within N years of target still get partial score. |
| `exact_match_score` | `1.0` | Score when year exactly matches or is within range. |
| `near_match_score` | `0.5` | Score when year is within tolerance. |
| `no_match_score` | `0.0` | Score when year is outside filter range. |

**Example:**

Query: "movies from 1995"
- Movie from 1995 → score: 1.0
- Movie from 1997 (within 5 years) → score: 0.5
- Movie from 2010 → score: 0.0

---

### Multi-Role Settings

Controls scoring for ambiguous person matches.

```ini
[multi_role]
full_match_score = 1.0
partial_match_score = 0.8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `full_match_score` | `1.0` | Score when person matches the exact detected role. |
| `partial_match_score` | `0.8` | Score when person matches one of their possible roles. |

Used when a person could be actor or director (e.g., Clint Eastwood).

---

### Fuzzy Matching Settings

Controls typo tolerance using RapidFuzz.

```ini
[fuzzy_matching]
enabled = True
people_min_similarity = 0.85
people_min_chars = 4
people_max_candidates = 3
genre_min_similarity = 0.83
genre_min_chars = 3
title_min_similarity = 0.85
title_min_chars = 5
return_all_above_threshold = True
use_popularity_weighting = True
popularity_weight = 0.3
title_fuzzy_enabled = True
title_fuzzy_min_similarity = 0.85
title_fuzzy_min_chars = 3
title_popularity_limit = 200000
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable fuzzy matching globally. |
| `people_min_similarity` | `0.85` | Minimum similarity (0.0–1.0) for person name matches. |
| `people_min_chars` | `4` | Names shorter than this require exact match. |
| `people_max_candidates` | `3` | Maximum fuzzy candidates per token. |
| `genre_min_similarity` | `0.83` | Minimum similarity for genre fuzzy matching. |
| `genre_min_chars` | `3` | Genre terms shorter than this require exact match. |
| `title_min_similarity` | `0.85` | Minimum similarity for title fuzzy matching. |
| `title_min_chars` | `5` | Title terms shorter than this require exact match. |
| `return_all_above_threshold` | `True` | Return all candidates above threshold (vs. just highest). |
| `use_popularity_weighting` | `True` | Prefer popular names when disambiguating fuzzy matches. |
| `popularity_weight` | `0.3` | Weight of popularity in combined fuzzy score. |
| `title_fuzzy_enabled` | `True` | Enable fuzzy matching for movie titles. |
| `title_fuzzy_min_similarity` | `0.85` | Minimum similarity for title corrections. |
| `title_fuzzy_min_chars` | `3` | Minimum chars for title fuzzy matching. |
| `title_popularity_limit` | `200000` | Number of popular titles to load for matching. |

**Combined Score Formula (when `use_popularity_weighting` is True):**

```
combined_score = (1 - popularity_weight) × fuzzy_score + popularity_weight × popularity_score
```

---

### Result Settings

Controls output behavior.

```ini
[results]
default_k = 20
max_k = 100
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_k` | `20` | Default number of results when not specified. |
| `max_k` | `100` | Maximum results allowed per query. |

---

### Metadata Settings

Controls display preferences.

```ini
[metadata]
preferred_title_field = primaryTitle
fallback_title_field = title
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preferred_title_field` | `primaryTitle` | Primary title column to use for display. |
| `fallback_title_field` | `title` | Fallback if preferred field is missing. |

---

## Programmatic Access

Configuration values are accessible through the `SearchConfig` instance:

```python
from src.config.config import get_config

config = get_config()

# Access weight values
hybrid_weight = config.weights['W_hybrid']
actor_weight = config.weights['W_actor']

# Access parser settings
min_name_words = config.query_parser['min_name_words']

# Access fuzzy matching settings
fuzzy_enabled = config.fuzzy_matching['enabled']
people_threshold = config.fuzzy_matching['people_min_similarity']
```

## Modifying Configuration

1. Edit `src/config/search_config.ini`
2. Either restart the application or call `reload_config()`

```python
from src.config.config import reload_config

# After modifying search_config.ini
config = reload_config()
```

Changes take effect immediately after reload. No need to rebuild indexes.
