import joblib
import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
import sys
import os

import sys
import os

# Add parent directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.custom_transformers import SearchCorpusGenerator
from src.config.config import get_config

class HybridSearcher:
    def __init__(self, 
                 bm25_path=None, 
                 sbert_embeddings_path=None, 
                 data_path=None,
                 sbert_model_name='all-MiniLM-L6-v2'):
        
        # Use absolute paths based on project_root defined at module level
        if bm25_path is None:
            bm25_path = os.path.join(project_root, "data/bm25_model.joblib")
        if sbert_embeddings_path is None:
            sbert_embeddings_path = os.path.join(project_root, "data/sbert_embeddings.npy")
        if data_path is None:
            data_path = os.path.join(project_root, "data/imdb_us_movies_unified.parquet")
        
        print("Loading BM25 model...")
        # --- POCZĄTEK WKLEJANIA ---
        if os.path.exists(bm25_path):
            self.bm25 = joblib.load(bm25_path)
        else:
            print("⚠️ Index file not found. Initializing empty for build...")
            self.bm25 = None
        # --- KONIEC WKLEJANIA ---
        
        print("Loading SBERT embeddings...")
        # --- Zastąp tamtą linię tym blokiem: ---
        if os.path.exists(sbert_embeddings_path):
            self.sbert_embeddings = np.load(sbert_embeddings_path)
        else:
            print("⚠️ SBERT embeddings not found. Will need to build them...")
            self.sbert_embeddings = None
        # ---------------------------------------
        
        print("Loading unified data...")
        # Note: BM25/SBERT were built on processed data (390,980 rows)
        # Unified data has 338,374 rows (deduplicated)
        # We need to rebuild BM25/SBERT to match unified data
        self.df = pl.read_parquet(data_path).to_pandas()
        
        print("Loading SBERT model...")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
        self.corpus_generator = SearchCorpusGenerator()
        
    def search(self, query, k=60, rrf_k=60, parsed_query=None):
        # Use corrected query if available (with fuzzy-matched names)
        # This ensures BM25/SBERT search for "tom hamks" uses "tom hanks"
        search_query = query
        if parsed_query and 'corrected_search_term' in parsed_query:
            search_query = parsed_query['corrected_search_term']
        
        # 1. BM25 Search
        normalized_query = self.corpus_generator._normalize_text(search_query).split()
        bm25_scores = self.bm25.get_scores(normalized_query)
        
        # Get top k BM25 indices
        # We want the indices of the top scores. 
        # argsort returns indices that sort the array. [::-1] reverses it (descending).
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        
        # 2. SBERT Search
        query_embedding = self.sbert_model.encode(search_query)
        # Dot product
        sbert_scores = np.dot(self.sbert_embeddings, query_embedding)
        
        # Get top k SBERT indices
        top_sbert_indices = np.argsort(sbert_scores)[::-1][:k]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Create a map of index -> score
        rrf_scores = {}
        
        # Process BM25 ranks
        for rank, idx in enumerate(top_bm25_indices):
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
            
        # Process SBERT ranks
        for rank, idx in enumerate(top_sbert_indices):
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
            
        # --- Title Match Boost (Context-Aware Jaccard Similarity) ---
        # Boost movies where title matches query using Jaccard similarity
        # Excludes detected filters and generic movie terms to prevent false matches
        config = get_config()
        
        jaccard_threshold = config.title_matching['jaccard_threshold']
        exact_boost = config.title_matching['exact_title_boost']
        partial_boost = config.title_matching['partial_title_boost']
        prefix_boost = config.title_matching['prefix_title_boost']
        generic_terms = config.title_matching['generic_movie_terms']
        
        # Build exclusion set from detected filters and generic terms
        exclude_tokens = set(generic_terms)  # Start with generic terms
        
        if parsed_query:
            # Add detected genres
            if 'genre_filter' in parsed_query and parsed_query['genre_filter']:
                exclude_tokens.update(parsed_query['genre_filter'])
            
            # Add detected actors (split multi-word names into tokens)
            if 'actor_filter' in parsed_query and parsed_query['actor_filter']:
                for actor in parsed_query['actor_filter']:
                    exclude_tokens.update(actor.lower().split())
            
            # Add detected directors
            if 'director_filter' in parsed_query and parsed_query['director_filter']:
                for director in parsed_query['director_filter']:
                    exclude_tokens.update(director.lower().split())
            
            # Add detected writers
            if 'writer_filter' in parsed_query and parsed_query['writer_filter']:
                for writer in parsed_query['writer_filter']:
                    exclude_tokens.update(writer.lower().split())
            
            # Add ambiguous role detections
            for key in ['actor_or_director', 'actor_or_writer', 'director_or_writer']:
                if key in parsed_query and parsed_query[key]:
                    for person in parsed_query[key]:
                        exclude_tokens.update(person.lower().split())
        
        normalized_query_str = " ".join(normalized_query)
        query_tokens = set(normalized_query)
        
        # Filter query tokens
        clean_query_tokens = query_tokens - exclude_tokens
        
        for idx in rrf_scores:
            title = self.df.iloc[idx]['primaryTitle']
            # Normalize title using same logic
            normalized_title = self.corpus_generator._normalize_text(title)
            title_tokens = set(normalized_title.split())
            
            # Filter title tokens with same exclusions
            clean_title_tokens = title_tokens - exclude_tokens
            
            # Calculate Jaccard similarity on cleaned tokens
            if len(clean_query_tokens) > 0 and len(clean_title_tokens) > 0:
                intersection = clean_query_tokens & clean_title_tokens
                union = clean_query_tokens | clean_title_tokens
                jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0
                
                # Apply boosts based on Jaccard similarity
                if jaccard_sim == 1.0:
                    # Perfect match (all tokens match)
                    rrf_scores[idx] += exact_boost
                elif jaccard_sim >= jaccard_threshold:
                    # High similarity (above threshold)
                    rrf_scores[idx] += partial_boost
                elif normalized_title.startswith(normalized_query_str + " "):
                    # Prefix match (e.g. "Avatar" matches "Avatar: The Way of Water")
                    rrf_scores[idx] += prefix_boost
            
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return top results with RRF scores
        results = []
        for idx in sorted_indices[:k]:
            row = self.df.iloc[idx].to_dict()
            # Add RRF score to the row
            row['rrf_score'] = rrf_scores[idx]
            results.append(row)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Simple test if run directly
    searcher = HybridSearcher()
    results = searcher.search("top gun maverick")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(results.head())
