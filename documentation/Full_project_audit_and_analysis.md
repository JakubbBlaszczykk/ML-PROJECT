## Full Project Audit & Analysis

Files reviewed
- cleaning.ipynb
- preprocessing_pipeline.ipynb
- PoC.ipynb
- Merging.ipynb
- Exploratory_Data_Analysis.ipynb
- src/custom_transformers.py
- documentation/preprocessing-recommendations.md

Summary verdict
- Preprocessing design is well documented and aims to produce ~64 engineered features via `full_preprocessor`.
- PoC.ipynb uses only a single text field (`search_corpus__searchable_text`) and ignores the engineered features and SBERT/ranking logic.
- The main architectural blocker is SearchCorpusGenerator: it concatenates all text into one blob, destroying field-level signals needed for filters, field-weighted embeddings, and accurate hybrid reranking.

1) Numeric-column data flow (example: numVotes)
- Intended pipeline (from notebooks):
  1. Raw placeholders (e.g., -1, "missing") → canonical missing (NaN) in cleaning.
  2. Extract missingness indicators (is_missing_*).
  3. Impute numeric missing values (SimpleImputer used in pipeline; currently fill_value=-1).
  4. Create ranking_numeric features, then apply transforms (log1p, scaling).
- Verified issues:
  - The pipeline often uses the -1 sentinel in downstream transforms (pd.cut, log1p, StandardScaler), so sentinel values contaminate bins, scaling, and log transforms (log1p(-1) -> -inf).
  - FeatureBinner (src/custom_transformers.py) uses pd.cut without first converting sentinels to NaN, causing missing to become real bin categories or `'nan'` strings in one-hot encodings.
- Conclusion: conceptual flow is correct, but implementation permits sentinel leakage into numeric transforms — this is a practical bug that must be fixed.

2) Concrete problems found
- Sentinels used as imputed values (-1) are passed into pd.cut, log1p, and scaling steps.
- FeatureBinner does not explicitly map sentinel → NaN → 'missing' before binning.
- Binning edge semantics (bin boundaries and labels) are not clearly documented; right/left inclusion could lead to unexpected bucket assignment.
- PoC uses only `searchable_text` vector; it does not incorporate SBERT, field vectors, or numeric reranking.

3) PoC vs. full_preprocessor gap
- full_preprocessor produces many features: ranking_numeric__*, ui_binned__*, filter_onehot_*, filter_multilabel_*, is_missing_*.
- PoC only uses the single `searchable_text` vector. Thus PoC is a limited text-only baseline and does not reflect retrieval/ranking strategy planned for production.

4) Key bottleneck — SearchCorpusGenerator
- It concatenates title, genres, cast, crew, etc. into one field.
- Consequences:
  - Loss of field-level semantics; prevents field-specific filters and efficient actor/director filtering.
  - Reranking and hybrid scoring accuracy suffers (cannot weight title vs. cast vs. plot differently).
  - Prevents precomputing per-field SBERT vectors and efficient field-weighted similarity.
- Recommendation: replace with multi-field corpus generator (title, cast, crew, genres, description) and compute/store field-specific representations.

5) Other recommended improvements
- Stop using -1 as impute sentinel that enters transforms. Workflow:
  1. Mark is_missing flags.
  2. Replace sentinel with np.nan.
  3. Impute with median/neutral value appropriate to variable.
  4. Apply log1p/scaling safely.
- Update FeatureBinner to treat NaN as explicit 'missing' category before pd.cut or use an explicit 'missing' bin.
- Ensure log1p is applied only to non-negative values; pre-check or shift where needed.
- Flatten nested lists (cast/directors) to separate index tables to enable efficient exact-match filters.
- Add unit tests checking: no -1 enters log1p, missing flags correctness, binning behavior at edges.

6) Small fixes (priority, low-effort)
- Convert -1 → np.nan before pd.cut / log1p / StandardScaler in preprocessing pipeline.
- Create missingness indicators before imputation and ensure imputer fill value doesn't propagate into transformations.
- Make FeatureBinner map NaNs to a 'missing' label (explicit category) instead of letting pd.cut create 'nan' strings.

7) Major design recommendation (priority)
- Implement multi-field SearchCorpusGenerator:
  - Output per-field text columns: title_text, cast_text, crew_text, genres_text, description_text.
  - Vectorize each with appropriate model (SBERT for semantic fields, TF-IDF for short categorical fields).
  - Compute field-weighted similarity at query time and merge with numeric reranking using ranking_numeric__* features.
- Precompute and store per-field SBERT vectors for scalability.

8) Next actionable steps
Short-term
- Patch preprocessing to ensure sentinel → NaN before any bin/log/scale.
- Update FeatureBinner to handle NaN explicitly.
- Add unit tests for sentinel leakage and binning correctness.

Sprint-level (Tasks 2–13 alignment)

## Sprint Tasks (2–13) — concise mapping, goals, deps, acceptance


2. Parse `principals` Column (The #1 Fix)
- Purpose: reliably extract cast/crew (actor, director, writer) into structured rows/columns and normalized actor/director lists.
- Inputs: raw principals column (Merging.ipynb / cleaning.ipynb).
- Outputs: flat actor/director/writer index columns and auxiliary lookup tables (actor -> tconst).
- Dependencies: Merging.ipynb, cleaning routines, Search corpus replacement.
- Acceptance: exact-match filters for actors/directors work; actor filters run via index (no substring search); unit tests for edge cases.

3. Normalize and Stem Titles for Search
- Purpose: create normalized title variants for exact/approximate matching and query parsing (lowercase, strip punctuation, remove stopwords, light stemming).
- Inputs: title column, title aliases.
- Outputs: `search_title_norm`, `search_title_shingles` etc.
- Dependencies: SearchCorpusGenerator replacement, PoC changes.
- Acceptance: normalized title queries return expected top results for test cases.

4. Calculate & Save SBERT Vectors (One-Time Task)
- Purpose: precompute and persist SBERT embeddings per field (title, cast, crew, genres, description).
- Inputs: per-field text outputs from new SearchCorpusGenerator.
- Outputs: stored vectors (joblib/npz) + mapping tconst->vectors.
- Dependencies: SBERT model, multi-field corpus.
- Acceptance: vector lookup for any tconst is faster than recompute; vectors reproducible and loaded by PoC.

5. R&D: 'Similarity Showdown' (Dot vs Cosine)
- Purpose: empirically choose similarity metric (cosine vs dot) and aggregation strategy for SBERT vectors and TF-IDF.
- Inputs: SBERT vectors, TF-IDF vectors, holdout queries with relevance labels.
- Outputs: recommended metric + small report.
- Dependencies: saved vectors, evaluation harness.
- Acceptance: clear choice with measured deltas on top-K retrieval metrics.

6. Build BM25 Keyword Model
- Purpose: implement BM25 or use an existing library to support keyword-first retrieval and lexicon scoring.
- Inputs: tokenized per-field TF corpora.
- Outputs: BM25 index/service and per-document BM25 scores per query.
- Dependencies: per-field text, tokenizer.
- Acceptance: BM25 baseline runs; improves recall for keyword-heavy queries.

7. Build `get_hybrid_relevance` Function
- Purpose: combine lexical (BM25/TF-IDF) and semantic (SBERT) scores plus numeric rerank into a single relevance signal.
- Inputs: per-field SBERT similarities, BM25 scores, ranking_numeric__* features.
- Outputs: unified relevance score per doc per query.
- Dependencies: steps 4–6, ranking numeric features from preprocessor.
- Acceptance: hybrid rerank reproducible; ablation shows hybrid > text-only.

8. Build `decade` Feature
- Purpose: derive decade buckets (e.g., 1990s) for filtering and boosting; add to ranking features.
- Inputs: startYear (cleaned).
- Outputs: `decade` categorical and one-hot/binned features.
- Dependencies: preprocessing fixes (sentinel -> NaN).
- Acceptance: decade filters work; included in LTR-Lite weighting and improves targeted queries.

9. Build `get_similar_films` Function ("More Like This")
- Purpose: produce "more like this" recommendations using multi-field SBERT + numeric similarity + optional genre overlap.
- Inputs: target tconst or vector, per-field vectors, numeric features.
- Outputs: ranked similar films list.
- Dependencies: steps 4,7.
- Acceptance: sanity checks: similar films for sample seeds match human expectations.

10. Build Smart Query Parser
- Purpose: parse user queries into structured components (title, person, genre, decade, free-text) and produce field-weighted subqueries.
- Inputs: normalized titles, actor/director index, genre list, numeric parsers.
- Outputs: parsed query object for retrieval pipeline.
- Dependencies: tasks 2,3,8.
- Acceptance: parser extracts fields for test queries; reduces false positives in filters.

11. Build `LTR-Lite` (Weighted Rank) Function
- Purpose: simple learnable weights or hand-tuned weights to combine signals (title match, SBERT per-field, BM25, numeric boosts).
- Inputs: features from get_hybrid_relevance, parsed query signals.
- Outputs: final ranked list; config file for weights.
- Dependencies: tasks 5–10.
- Acceptance: configurable weight set; baseline weights improve over PoC; unit tests verify deterministic ranking.

12. Build Main `search()` Function
- Purpose: end-to-end search API combining parsing, retrieval, hybrid scoring, LTR-Lite rerank, filters, pagination.
- Inputs: raw user query + filter params.
- Outputs: ranked results with explanations (which field matched, top contributing features).
- Dependencies: tasks 2–11.
- Acceptance: passes integration tests (sample queries produce expected outputs and respects filters).

13. Tune: LTR-Lite Weights
- Purpose: tune weights (manual grid or small optimization) using human-labeled or heuristic validation set.
- Inputs: small labeled query relevance set, LTR-Lite.
- Outputs: final weight config and evaluation report.
- Dependencies: tasks 11–12 and evaluation harness.
- Acceptance: improved evaluation metrics vs baseline and documented final weights.


### Prioritization & Notes
- Highest priority: tasks 2 (principals parsing), 4 (SBERT vectors), and 10 (query parser) — these unblock field-aware search.
- Immediate bugfixes (preprocessor) must run in parallel (sentinel → NaN, FeatureBinner fixes) before task 8/11.
- Deliverables per task: brief unit/integration tests, clear input/output contracts, and small example notebooks demonstrating behavior.


9) Final assessment
- The cleaning and preprocessing design is conceptually robust, but the implementation has sentinel-handling bugs that can corrupt numeric transforms and binning.
- The major functional limitation is PoC's reliance on a single concatenated text field (SearchCorpusGenerator). Fixing this and integrating SBERT + numeric reranking (the planned sprint tasks) is the correct path to production-readiness.
- The rest of the sprint (Tasks 2–13) is the correct path forward. It addresses the two critical failure modes we found (sentinel/missingness contamination of numeric transforms and the single-field search corpus), and it implements the hybrid, field-aware indexing and reranking required for production-grade search.
