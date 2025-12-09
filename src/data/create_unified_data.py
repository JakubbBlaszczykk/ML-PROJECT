"""
Create Unified Data File

This script merges imdb_us_movies_processed.parquet and imdb_us_movies_merged.parquet
into a single optimized file with only necessary columns and deduplication by tconst.

Output: imdb_us_movies_unified.parquet
"""

import pandas as pd
import numpy as np

def extract_names_from_array(arr):
    """Extract primaryName values from structured arrays."""
    if arr is None or (isinstance(arr, float) and np.isnan(arr)):
        return ""
    if not isinstance(arr, (list, np.ndarray)):
        return ""
    
    names = []
    for item in arr:
        if isinstance(item, dict) and 'primaryName' in item:
            name = item['primaryName']
            if name is not None:  # Filter out None values
                names.append(name)
    
    return ','.join(names) if names else ""

def extract_actor_names(cast_arr):
    """Extract actor names (category='actor' or 'actress') from cast array."""
    if cast_arr is None or (isinstance(cast_arr, float) and np.isnan(cast_arr)):
        return ""
    if not isinstance(cast_arr, (list, np.ndarray)):
        return ""
    
    names = []
    for item in cast_arr:
        if isinstance(item, dict):
            category = item.get('category', '').lower()
            if category in ['actor', 'actress'] and 'primaryName' in item:
                name = item['primaryName']
                if name is not None:  # Filter out None values
                    names.append(name)
    
    return ','.join(names) if names else ""

def create_unified_data():
    print("="*80)
    print("Creating Unified Movie Data File")
    print("="*80)
    
    # Load processed data
    print("\n[1/5] Loading processed data...")
    processed_df = pd.read_parquet("data/imdb_us_movies_processed.parquet")
    print(f"  Loaded {len(processed_df):,} rows")
    print(f"  Columns: {len(processed_df.columns)} total")
    
    # Load merged data
    print("\n[2/5] Loading merged data...")
    merged_df = pd.read_parquet("data/imdb_us_movies_merged.parquet")
    print(f"  Loaded {len(merged_df):,} rows")
    print(f"  Columns: {len(merged_df.columns)} total")
    
    # Check for duplicates in merged data
    print(f"\n[3/5] Deduplicating by tconst...")
    duplicate_tconsts = merged_df[merged_df.duplicated(subset=['tconst'], keep=False)]
    if len(duplicate_tconsts) > 0:
        print(f"  Found {len(duplicate_tconsts):,} duplicate rows across {duplicate_tconsts['tconst'].nunique():,} unique tconsts")
    
    # Deduplicate merged data - keep first occurrence per tconst
    merged_df_dedup = merged_df.drop_duplicates(subset=['tconst'], keep='first')
    removed_count = len(merged_df) - len(merged_df_dedup)
    if removed_count > 0:
        print(f"  ✓ Removed {removed_count:,} duplicate rows")
    print(f"  After deduplication: {len(merged_df_dedup):,} unique tconsts")
    
    # Extract names from structured arrays
    print(f"\n[4/5] Extracting actor/director/writer names...")
    merged_df_dedup['actor_name'] = merged_df_dedup['cast'].apply(extract_actor_names)
    merged_df_dedup['director_name'] = merged_df_dedup['directors'].apply(extract_names_from_array)
    merged_df_dedup['writer_name'] = merged_df_dedup['writers'].apply(extract_names_from_array)
    print(f"  ✓ Extracted names from structured arrays")
    
    # Create unified dataframe
    print(f"\n[5/5] Creating unified dataframe...")
    
    # Extract tconst from processed data (it's prefixed)
    processed_df['tconst'] = processed_df['pass_tconst__cat__tconst']
    
    # Select columns from processed data
    processed_cols = {
        'tconst': 'tconst',
        'search_corpus__searchable_text': 'searchable_text',
        'search_corpus__normalized_title': 'normalized_title',
        'ranking_numeric__num__numVotes_log': 'numVotes_log',
        'ranking_numeric__num__averageRating': 'averageRating_scaled',
        'ranking_numeric__num__startYear': 'startYear_scaled',
        'ranking_numeric__num__runtimeMinutes': 'runtimeMinutes_scaled'
    }
    
    processed_subset = processed_df[list(processed_cols.keys())].rename(columns=processed_cols)
    
    # Select columns from merged data  
    merged_cols = [
        'tconst', 'primaryTitle', 'startYear', 'genres', 
        'actor_name', 'director_name', 'writer_name',
        'numVotes', 'averageRating', 'runtimeMinutes'
    ]
    
    merged_subset = merged_df_dedup[merged_cols]
    
    # Merge on tconst
    print(f"  Merging datasets on tconst...")
    unified_df = processed_subset.merge(merged_subset, on='tconst', how='inner')
    
    print(f"  ✓ Unified dataframe created: {len(unified_df):,} rows × {len(unified_df.columns)} columns")
    
    # Verify no duplicates
    duplicate_check = unified_df[unified_df.duplicated(subset=['tconst'], keep=False)]
    if len(duplicate_check) > 0:
        print(f"  ⚠ WARNING: Still found {len(duplicate_check):,} duplicates after merge!")
        print(f"  Removing remaining duplicates...")
        unified_df = unified_df.drop_duplicates(subset=['tconst'], keep='first')
        print(f"  ✓ After final deduplication: {len(unified_df):,} rows")
    else:
        print(f"  ✓ No duplicates found - 1 row per tconst")
    
    # Save unified file
    output_file = "data/imdb_us_movies_unified.parquet"
    print(f"\n  Saving to {output_file}...")
    unified_df.to_parquet(output_file, index=False)
    
    # Get file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✓ Saved {len(unified_df):,} rows to {output_file} ({file_size_mb:.1f} MB)")
    
    # Show summary
    print(f"\n{'='*80}")
    print(f"SUCCESS")
    print(f"{'='*80}")
    print(f"\nOutput file: {output_file}")
    print(f"  - {len(unified_df):,} unique movies (1 row per tconst)")
    print(f"  - {len(unified_df.columns)} columns")
    print(f"  - {file_size_mb:.1f} MB file size")

if __name__ == "__main__":
    create_unified_data()
