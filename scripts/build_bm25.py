import pandas as pd
import polars as pl
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from src.data.custom_transformers import SearchCorpusGenerator
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.custom_transformers import SearchCorpusGenerator

def build_bm25():
    print("Loading unified data...")
    input_file = "data/imdb_us_movies_unified.parquet"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        print("Please run create_unified_data.py first to generate the unified file.")
        return

    df = pl.read_parquet(input_file).to_pandas()
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    target_col = 'searchable_text'
    if target_col not in df.columns:
        print(f"Error: {target_col} not found in columns.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print("Tokenizing corpus...")
    corpus_generator = SearchCorpusGenerator()
    
    # Apply normalization to each document
    # This removes punctuation, lowercases, etc.
    tokenized_corpus = df[target_col].fillna('').apply(corpus_generator._normalize_text).str.split().tolist()
    
    print(f"Fitting BM25Okapi on {len(tokenized_corpus)} documents...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the model
    output_file = "data/bm25_model.joblib"
    print(f"Saving model to {output_file}...")
    joblib.dump(bm25, output_file)
    
    print("Model saved successfully.")
    
    # Test the model
    # We need to normalize the query using the same logic
    corpus_generator = SearchCorpusGenerator()
    query = "top gun maverick"
    normalized_query = corpus_generator._normalize_text(query).split()
    print(f"Testing with query: '{query}' -> {normalized_query}")
    
    scores = bm25.get_scores(normalized_query)
    top_n = np.argsort(scores)[::-1][:5]
    
    print("Top 5 matches:")
    for idx in top_n:
        # Show the searchable text to verify it contains all features
        searchable_text = df.iloc[idx][target_col]
        # Truncate for readability
        display_text = searchable_text[:100] + "..." if len(searchable_text) > 100 else searchable_text
        print(f"Score: {scores[idx]:.4f} | Text: {display_text}")

if __name__ == "__main__":
    build_bm25()
