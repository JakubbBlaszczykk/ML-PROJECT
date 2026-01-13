"""
MovieSearchEngine - Main search function for the movie search engine
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.search.hybrid_search import HybridSearcher
from src.search.query_parser import QueryParser
from src.config.config import get_config


class MovieSearchEngine:
    def __init__(
        self,
        bm25_path=None,
        sbert_embeddings_path=None,
        data_path=None,
        sbert_model_name="all-MiniLM-L6-v2",
    ):
        # Set paths relative to project root
        if bm25_path is None:
            bm25_path = os.path.join(project_root, "data/bm25_model.joblib")
        if sbert_embeddings_path is None:
            sbert_embeddings_path = os.path.join(project_root, "data/sbert_embeddings.npy")
        if data_path is None:
            data_path = os.path.join(project_root, "data/imdb_us_movies_unified.parquet")

        print("=" * 80)
        print("Initializing MovieSearchEngine...")
        print("=" * 80)

        # 1. Hybrid Searcher
        print("\n[1/2] Loading Hybrid Search components...")
        self.hybrid_searcher = HybridSearcher(
            bm25_path=bm25_path,
            sbert_embeddings_path=sbert_embeddings_path,
            data_path=data_path,
            sbert_model_name=sbert_model_name,
        )

        # 2. Query Parser
        print("\n[2/2] Loading Query Parser...")
        qp_data = os.path.join(project_root, "data/imdb_us_movies_merged.parquet")
        self.query_parser = QueryParser(data_path=qp_data)

        # 3. Config & Stats
        self.config = get_config()
        self._precompute_stats()

        print("\n" + "=" * 80)
        print("MovieSearchEngine ready!")
        print("=" * 80 + "\n")

    def _precompute_stats(self):
        """Calculate max log votes for normalization"""
        df = self.hybrid_searcher.df
        if "numVotes_log" in df.columns:
            self.max_log_votes = df["numVotes_log"].max()
        else:
            votes = df["numVotes"].dropna()
            self.max_log_votes = np.log1p(votes.max()) if len(votes) else 1.0

        print(f"Loaded {len(df):,} movies")
        print(f"  Max log(numVotes): {self.max_log_votes:.2f}")

    def search(self, query: str, k: int = 20, weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Main execution flow: Parse -> Retrieve Candidates -> Rank
        """
        if weights is None:
            weights = self.config.weights

        # 1. Parse
        parsed_query = self.query_parser.parse_query(query)

        # 2. Retrieve
        candidate_k = max(
            self.config.hybrid_search["min_candidates"],
            k * self.config.hybrid_search["candidate_multiplier"],
        )
        
        candidates = self.hybrid_searcher.search(
            query, k=candidate_k, parsed_query=parsed_query
        )

        if candidates.empty:
            return pd.DataFrame()

        # 3. Rank (Compute Scores)
        scored = self._compute_weighted_scores(candidates, parsed_query, weights)
        scored = self._add_metadata_columns(scored)

        # 4. Return Top K
        return scored.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)

    # =========================
    # SCORING LOGIC
    # =========================

    def _compute_weighted_scores(self, candidates, parsed_query, weights):
        results = candidates.copy()

        # Initialize columns
        score_cols = [
            "hybrid_score", "popularity_score", "rating_score", 
            "actor_match_score", "director_match_score", "writer_match_score", 
            "genre_match_score", "year_match_score", "filter_boost", "final_score"
        ]
        for col in score_cols:
            results[col] = 0.0

        # Normalize Hybrid Score (RRF)
        if "rrf_score" in results.columns:
            rrf = results["rrf_score"]
            if rrf.max() > rrf.min():
                results["hybrid_score"] = (rrf - rrf.min()) / (rrf.max() - rrf.min())
            else:
                results["hybrid_score"] = 1.0

        # Iterate and Calculate
        for idx, row in results.iterrows():
            
            # --- 1. Popularity Score ---
            num_votes = row.get("numVotes", 0)
            if pd.isna(num_votes): num_votes = 0
            
            if row.get("numVotes_log", 0) > 0:
                popularity = row["numVotes_log"] / self.max_log_votes
            else:
                popularity = np.log1p(num_votes) / self.max_log_votes if num_votes > 0 else 0.0

            # --- 2. Rating Score ---
            rating = row.get("averageRating", 0)
            if pd.isna(rating): rating = 0
            rating_score = rating / 10.0

            # --- 3. Filter Matching ---
            actor = self._compute_actor_match(row, parsed_query)
            director = self._compute_director_match(row, parsed_query)
            writer = self._compute_writer_match(row, parsed_query)
            genre = self._compute_genre_match(row, parsed_query)
            year = self._compute_year_match(row, parsed_query)

            filter_boost = (
                weights["W_actor"] * actor
                + weights["W_director"] * director
                + weights["W_writer"] * writer
                + weights["W_genre"] * genre
                + weights["W_year"] * year
            )

            # --- 4. Final Calculation ---
            final_score = (
                weights["W_hybrid"] * row["hybrid_score"]
                + weights["W_popularity"] * popularity
                + weights["W_rating"] * rating_score
                + filter_boost
            )

            # If a movie has fewer than 200 votes, subtract 5.0 points.
            if num_votes < 200:
                final_score -= 5.0 

            # Save scores
            results.at[idx, "popularity_score"] = popularity
            results.at[idx, "rating_score"] = rating_score
            results.at[idx, "actor_match_score"] = actor
            results.at[idx, "director_match_score"] = director
            results.at[idx, "writer_match_score"] = writer
            results.at[idx, "genre_match_score"] = genre
            results.at[idx, "year_match_score"] = year
            results.at[idx, "filter_boost"] = filter_boost
            results.at[idx, "final_score"] = final_score

        return results

    # =========================
    # MATCH HELPERS
    # =========================

    def _compute_actor_match(self, row, pq):
        # Helper: Check if query actor exists in movie's actor list
        names = str(row.get("actor_name", "")).lower().split(",")
        filters = pq.get("actor_filter", []) + pq.get("actor_or_director", [])
        if not filters: return 0.0
        return float(any(f in names for f in filters))

    def _compute_director_match(self, row, pq):
        names = str(row.get("director_name", "")).lower().split(",")
        filters = pq.get("director_filter", []) + pq.get("actor_or_director", [])
        if not filters: return 0.0
        return float(any(f in names for f in filters))

    def _compute_writer_match(self, row, pq):
        names = str(row.get("writer_name", "")).lower().split(",")
        filters = pq.get("writer_filter", [])
        if not filters: return 0.0
        return float(any(f in names for f in filters))

    def _compute_genre_match(self, row, pq):
        genres = str(row.get("genres", "")).lower().split(",")
        gf = pq.get("genre_filter", [])
        if not gf: return 0.0
        # Calculate percentage of matched genres
        return sum(g in genres for g in gf) / len(gf)

    def _compute_year_match(self, row, pq):
        yf = pq.get("year_filter")
        if not yf: return 0.0
        
        movie_year = row.get("startYear")
        if pd.isna(movie_year): return 0.0
        
        if "exact" in yf:
            return 1.0 if movie_year == yf["exact"] else 0.0
        return 0.0

    def _add_metadata_columns(self, df):
        # Ensure we have a standard 'title' column
        if "primaryTitle" in df.columns and "title" not in df.columns:
            df["title"] = df["primaryTitle"]
        return df