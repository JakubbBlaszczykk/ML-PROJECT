"""
Build SBERT Embeddings for Unified Data

This script creates semantic embeddings for the searchable_text column
using Sentence-BERT (all-MiniLM-L6-v2 model).
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def build_sbert_embeddings():
    print("="*80)
    print("Building SBERT Embeddings for Unified Data")
    print("="*80)
    
    # Load unified data
    input_file = "data/imdb_us_movies_unified.parquet"
    print(f"\n[1/3] Loading {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        print("Please run create_unified_data.py first.")
        return
    
    df = pd.read_parquet(input_file)
    print(f"  Loaded {len(df):,} rows")
    
    # Check for searchable_text column
    if 'searchable_text' not in df.columns:
        print(f"Error: 'searchable_text' column not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Load SBERT model
    print(f"\n[2/3] Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")
    
    # Encode all texts
    print(f"\n[3/3] Encoding {len(df):,} documents...")
    print(f"  This may take 5-15 minutes depending on your hardware...")
    
    texts = df['searchable_text'].fillna('').tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,  # Adjust based on available memory
        convert_to_numpy=True
    )
    
    print(f"\n  ✓ Encoded {len(embeddings):,} documents")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Save embeddings
    output_file = "data/sbert_embeddings.npy"
    print(f"\n  Saving embeddings to {output_file}...")
    np.save(output_file, embeddings)
    
    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✓ Saved embeddings ({file_size_mb:.1f} MB)")
    
    print(f"\n{'='*80}")
    print(f"SUCCESS")
    print(f"{'='*80}")
    print(f"\nOutput: {output_file}")
    print(f"  - {len(embeddings):,} embeddings")
    print(f"  - {embeddings.shape[1]} dimensions")
    print(f"  - {file_size_mb:.1f} MB file size")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    build_sbert_embeddings()
