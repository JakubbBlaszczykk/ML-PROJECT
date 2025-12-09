"""
MovieSearchEngine - Main search function for the movie search engine

Combines query parsing, hybrid search, and weighted ranking with soft filter boosting
to create a bulletproof search experience.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Add parent directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.search.hybrid_search import HybridSearcher
from src.search.query_parser import QueryParser
from src.config.config import get_config




class MovieSearchEngine:
    """
    Main search engine that orchestrates query parsing, hybrid search, and weighted ranking.
    
    Bulletproof Design:
    - Handles ambiguous queries (e.g., "action movie" = genre OR title)
    - Exact title matches always rank above filter-only matches
    - Soft filter boosting (no hard filtering)
    - Weighted formula balances relevance and filter matches
    """
    
    def __init__(self, 
                 bm25_path=None,
                 sbert_embeddings_path=None,
                 data_path=None,
                 sbert_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the search engine.
        
        Args:
            bm25_path: Path to BM25 model
            sbert_embeddings_path: Path to SBERT embeddings
            data_path: Path to unified data file
            sbert_model_name: SBERT model name
        """
        # Use absolute paths based on project_root defined at module level
        if bm25_path is None:
            bm25_path = os.path.join(project_root, "data/bm25_model.joblib")
        if sbert_embeddings_path is None:
            sbert_embeddings_path = os.path.join(project_root, "data/sbert_embeddings.npy")
        if data_path is None:
            data_path = os.path.join(project_root, "data/imdb_us_movies_unified.parquet")
        """
        Initialize the search engine.
        
        Args:
            bm25_path: Path to BM25 model
            sbert_embeddings_path: Path to SBERT embeddings
            data_path: Path to unified data file
            sbert_model_name: SBERT model name
        """
        print("=" * 80)
        print("Initializing MovieSearchEngine...")
        print("=" * 80)
        
        # Initialize hybrid searcher
        print("\n[1/2] Loading Hybrid Search components...")
        self.hybrid_searcher = HybridSearcher(
            bm25_path=bm25_path,
            sbert_embeddings_path=sbert_embeddings_path,
            data_path=data_path,
            sbert_model_name=sbert_model_name
        )
        
        # Initialize query parser (still needs merged data for lookup sets)
        # TODO: Could optimize by extracting unique names from unified file instead
        print("\n[2/2] Loading Query Parser...")
        query_parser_data_path = os.path.join(project_root, "data/imdb_us_movies_merged.parquet")
        self.query_parser = QueryParser(data_path=query_parser_data_path)
        
        # Load configuration
        self.config = get_config()
        
        # Precompute normalization constants from hybrid_searcher's data
        self._precompute_stats()
        
        print("\n" + "=" * 80)
        print("MovieSearchEngine ready!")
        print("=" * 80 + "\n")
    
    def _precompute_stats(self):
        """Precompute statistics for normalization."""
        # Get numVotes stats from unified data
        df = self.hybrid_searcher.df
        if 'numVotes_log' in df.columns:
            # Use pre-computed log values
            self.max_log_votes = df['numVotes_log'].max()
        else:
            # Fallback to computing from numVotes
            votes = df['numVotes'].dropna()
            if len(votes) > 0:
                self.max_log_votes = np.log1p(votes.max())
            else:
                self.max_log_votes = 1.0
        
        print(f"Loaded {len(df):,} movies")
        print(f"  Max log(numVotes): {self.max_log_votes:.2f}")
    
    def search(self, query: str, k: int = 20, weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Main search function - the "brain" of the search engine.
        
        Bulletproof design handles both filter interpretation AND exact title matching:
        - Query "action movie" finds both genre:action movies AND movie titled "Action Movie"
        - Exact title matches rank highest (high hybrid score)
        - Filter matches get moderate boosts
        
        Args:
            query: User's search query
            k: Number of results to return
            weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)
            
        Returns:
            DataFrame with top k results, including:
            - All movie metadata
            - final_score: Overall weighted score
            - hybrid_score: Base BM25+SBERT score
            - filter_boost: Total boost from filters
            - Score breakdown columns
        """
        if weights is None:
            # Use weights from configuration
            weights = self.config.weights
        
        print(f"\nSearching for: '{query}'")
        print("-" * 80)
        
        # Step 1: Parse query to extract filters
        parsed_query = self.query_parser.parse_query(query)
        print(f"Parsed query:")
        for key, value in parsed_query.items():
            if value and key != 'search_term':
                print(f"  {key}: {value}")
        
        # Step 2: Get candidate results from hybrid search
        # Use larger k for candidates to ensure good coverage
        candidate_k = max(
            self.config.hybrid_search['min_candidates'],
            k * self.config.hybrid_search['candidate_multiplier']
        )
        print(f"\nGetting {candidate_k} candidates from hybrid search...")
        candidates = self.hybrid_searcher.search(
            query, 
            k=candidate_k,
            parsed_query=parsed_query  # Pass parsed query for context-aware title matching
        )
        
        if len(candidates) == 0:
            print("No results found!")
            return pd.DataFrame()
        
        # Step 3: Compute weighted scores with filter boosting
        print(f"Computing weighted scores for {len(candidates)} candidates...")
        scored_results = self._compute_weighted_scores(
            candidates=candidates,
            parsed_query=parsed_query,
            weights=weights
        )
        
        # Step 4: Add metadata columns for better display
        scored_results = self._add_metadata_columns(scored_results)
        
        # Step 5: Sort by final score and return top k
        scored_results = scored_results.sort_values('final_score', ascending=False)
        top_results = scored_results.head(k).reset_index(drop=True)
        
        print(f"\nReturning top {len(top_results)} results")
        print("=" * 80)
        
        return top_results
    
    def _compute_weighted_scores(self, 
                                  candidates: pd.DataFrame,
                                  parsed_query: Dict,
                                  weights: Dict) -> pd.DataFrame:
        """
        Compute weighted scores for candidate results.
        
        Formula:
        final_score = W_hybrid * hybrid_score +
                     W_popularity * popularity_score +
                     W_rating * rating_score +
                     W_actor * actor_match_score +
                     W_director * director_match_score +
                     W_writer * writer_match_score +
                     W_genre * genre_match_score +
                     W_year * year_match_score
        
        Args:
            candidates: Candidate results from hybrid search
            parsed_query: Parsed query with filters
            weights: Weight configuration
            
        Returns:
            DataFrame with score columns added
        """
        results = candidates.copy()
        
        # Unified file has 'tconst' column directly (no prefix)
        if 'tconst' not in results.columns:
            raise ValueError("Cannot find tconst column in unified data")
        
        # No need to look up in separate file - all data is in the candidates DataFrame
        
        # Initialize score columns
        results['hybrid_score'] = 0.0
        results['popularity_score'] = 0.0
        results['rating_score'] = 0.0
        results['actor_match_score'] = 0.0
        results['director_match_score'] = 0.0
        results['writer_match_score'] = 0.0
        results['genre_match_score'] = 0.0
        results['year_match_score'] = 0.0
        results['filter_boost'] = 0.0
        results['final_score'] = 0.0
        
        # Normalize RRF scores to 0-1 range for hybrid_score
        # RRF scores are already in results from HybridSearcher
        if 'rrf_score' in results.columns:
            max_rrf = results['rrf_score'].max()
            min_rrf = results['rrf_score'].min()
            
            if max_rrf > min_rrf:
                # Normalize to 0-1 range
                results['hybrid_score'] = (results['rrf_score'] - min_rrf) / (max_rrf - min_rrf)
            else:
                # All scores are the same, give them all max score
                results['hybrid_score'] = 1.0
        else:
            # Fallback: use position-based scoring if rrf_score is missing
            results['hybrid_score'] = 0.0
        
        # Compute scores for each result
        for rank, (idx, row) in enumerate(results.iterrows()):
            # movie_data is just the row itself from unified data
            
            # 1. Hybrid score - already computed above from RRF scores
            # (No need to recalculate, it's already normalized)
            
            # 2. Popularity score (from pre-computed numVotes_log in unified file)
            if 'numVotes_log' in row and row['numVotes_log'] > 0:
                popularity_score = row['numVotes_log'] / self.max_log_votes
            else:
                # Fallback to computing from numVotes
                num_votes = row.get('numVotes', 0)
                if num_votes and num_votes > 0:
                    popularity_score = np.log1p(num_votes) / self.max_log_votes
                else:
                    popularity_score = 0.0
            results.at[idx, 'popularity_score'] = popularity_score
            
            # 3. Rating score (from averageRating in unified file)
            rating = row.get('averageRating')
            if rating and rating > 0:
                rating_score = rating / 10.0
            else:
                rating_score = 0.0
            results.at[idx, 'rating_score'] = rating_score
            
            # 4-8. Filter match scores (using row data directly)
            actor_match = self._compute_actor_match(row, parsed_query)
            results.at[idx, 'actor_match_score'] = actor_match
            
            director_match = self._compute_director_match(row, parsed_query)
            results.at[idx, 'director_match_score'] = director_match
            
            writer_match = self._compute_writer_match(row, parsed_query)
            results.at[idx, 'writer_match_score'] = writer_match
            
            genre_match = self._compute_genre_match(row, parsed_query)
            results.at[idx, 'genre_match_score'] = genre_match
            
            year_match = self._compute_year_match(row, parsed_query)
            results.at[idx, 'year_match_score'] = year_match
            
            # Compute filter boost (sum of filter matches weighted)
            filter_boost = (
                weights['W_actor'] * actor_match +
                weights['W_director'] * director_match +
                weights['W_writer'] * writer_match +
                weights['W_genre'] * genre_match +
                weights['W_year'] * year_match
            )
            results.at[idx, 'filter_boost'] = filter_boost
            
            # Compute final score (additive formula for bulletproof design)
            final_score = (
                weights['W_hybrid'] * row['hybrid_score'] +
                weights['W_popularity'] * popularity_score +
                weights['W_rating'] * rating_score +
                filter_boost
            )
            results.at[idx, 'final_score'] = final_score
        
        return results
    
    def _compute_actor_match(self, movie_data: Dict, parsed_query: Dict) -> float:
        """Compute actor match score (0.0 to 1.0)."""
        actor_filters = parsed_query.get('actor_filter', [])
        actor_or_director = parsed_query.get('actor_or_director', [])
        
        if not actor_filters and not actor_or_director:
            return 0.0
        
        # Get movie's actors from comma-separated string
        actor_names_str = movie_data.get('actor_name', '')
        if not actor_names_str or pd.isna(actor_names_str):
            return 0.0
        
        # Parse comma-separated names
        movie_actors = set(name.strip().lower() for name in actor_names_str.split(',') if name.strip())
        
        # Check for matches
        matched = 0
        total = len(actor_filters) + len(actor_or_director)
        
        for actor in actor_filters:
            if actor in movie_actors:
                matched += 1.0  # Full match
        
        for actor in actor_or_director:
            if actor in movie_actors:
                matched += 0.5  # Partial match (ambiguous)
        
        return matched / total if total > 0 else 0.0
    
    def _compute_director_match(self, movie_data: Dict, parsed_query: Dict) -> float:
        """Compute director match score (0.0 to 1.0)."""
        director_filters = parsed_query.get('director_filter', [])
        actor_or_director = parsed_query.get('actor_or_director', [])
        director_or_writer = parsed_query.get('director_or_writer', [])
        
        if not director_filters and not actor_or_director and not director_or_writer:
            return 0.0
        
        # Get movie's directors from comma-separated string
        director_names_str = movie_data.get('director_name', '')
        if not director_names_str or pd.isna(director_names_str):
            return 0.0
        
        # Parse comma-separated names
        movie_directors = set(name.strip().lower() for name in director_names_str.split(',') if name.strip())
        
        # Check for matches
        matched = 0
        total = len(director_filters) + len(actor_or_director) + len(director_or_writer)
        
        for director in director_filters:
            if director in movie_directors:
                matched += 1.0  # Full match
        
        for director in actor_or_director:
            if director in movie_directors:
                matched += 0.5  # Partial match (ambiguous)
        
        for director in director_or_writer:
            if director in movie_directors:
                matched += 0.5  # Partial match (ambiguous)
        
        return matched / total if total > 0 else 0.0
    
    def _compute_writer_match(self, movie_data: Dict, parsed_query: Dict) -> float:
        """Compute writer match score (0.0 to 1.0)."""
        writer_filters = parsed_query.get('writer_filter', [])
        actor_or_writer = parsed_query.get('actor_or_writer', [])
        director_or_writer = parsed_query.get('director_or_writer', [])
        
        if not writer_filters and not actor_or_writer and not director_or_writer:
            return 0.0
        
        # Get movie's writers from comma-separated string
        writer_names_str = movie_data.get('writer_name', '')
        if not writer_names_str or pd.isna(writer_names_str):
            return 0.0
        
        # Parse comma-separated names
        movie_writers = set(name.strip().lower() for name in writer_names_str.split(',') if name.strip())
        
        # Check for matches
        matched = 0
        total = len(writer_filters) + len(actor_or_writer) + len(director_or_writer)
        
        for writer in writer_filters:
            if writer in movie_writers:
                matched += 1.0  # Full match
        
        for writer in actor_or_writer:
            if writer in movie_writers:
                matched += 0.5  # Partial match (ambiguous)
        
        for writer in director_or_writer:
            if writer in movie_writers:
                matched += 0.5  # Partial match (ambiguous)
        
        return matched / total if total > 0 else 0.0
    
    def _compute_genre_match(self, movie_data: Dict, parsed_query: Dict) -> float:
        """Compute genre match score (0.0 to 1.0)."""
        genre_filters = parsed_query.get('genre_filter', [])
        
        if not genre_filters:
            return 0.0
        
        # Get movie's genres
        genres_str = movie_data.get('genres', '')
        if not genres_str:
            return 0.0
        
        movie_genres = set(g.strip().lower() for g in genres_str.split(',') if g.strip())
        
        # Count matches
        matched = sum(1 for genre in genre_filters if genre in movie_genres)
        
        return matched / len(genre_filters)
    
    def _compute_year_match(self, movie_data: Dict, parsed_query: Dict) -> float:
        """
        Compute year match score (0.0 to 1.0).
        
        - 1.0 if year matches exactly or in range
        - 0.5 if within 5 years
        - 0.0 otherwise
        """
        year_filter = parsed_query.get('year_filter')
        
        if not year_filter:
            return 0.0
        
        movie_year = movie_data.get('startYear')
        if not movie_year:
            return 0.0
        
        # Get config settings
        year_tolerance = self.config.year_matching['year_tolerance']
        exact_score = self.config.year_matching['exact_match_score']
        near_score = self.config.year_matching['near_match_score']
        no_match = self.config.year_matching['no_match_score']
        
        # Check exact year
        if 'exact' in year_filter:
            target_year = year_filter['exact']
            diff = abs(movie_year - target_year)
            if diff == 0:
                return exact_score
            elif diff <= year_tolerance:
                return near_score
            else:
                return no_match
        
        # Check year range
        min_year = year_filter.get('min')
        max_year = year_filter.get('max')
        
        # Within range
        if min_year and max_year:
            if min_year <= movie_year <= max_year:
                return exact_score
            elif abs(movie_year - min_year) <= year_tolerance or abs(movie_year - max_year) <= year_tolerance:
                return near_score
            else:
                return no_match
        elif min_year:
            if movie_year >= min_year:
                return exact_score
            elif abs(movie_year - min_year) <= year_tolerance:
                return near_score
            else:
                return no_match
        elif max_year:
            if movie_year <= max_year:
                return exact_score
            elif abs(movie_year - max_year) <= year_tolerance:
                return near_score
            else:
                return no_match
        
        return no_match
    
    def _add_metadata_columns(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Unified file already has all metadata - just rename/select columns.
        
        Args:
            results: Results DataFrame from unified file
            
        Returns:
            DataFrame with standardized column names
        """
        # Unified file already has: primaryTitle, startYear, genres, actor_name, etc.
        # Just ensure we have a 'title' column for backwards compatibility
        if 'primaryTitle' in results.columns and 'title' not in results.columns:
            results['title'] = results['primaryTitle']
        
        return results



if __name__ == "__main__":
    # Simple test
    print("Testing MovieSearchEngine...")
    engine = MovieSearchEngine()
    
    # Test query
    results = engine.search("top gun maverick", k=10)
    
    print("\nTop 10 Results:")
    print(results[['title', 'final_score', 'hybrid_score', 'filter_boost']].to_string())
