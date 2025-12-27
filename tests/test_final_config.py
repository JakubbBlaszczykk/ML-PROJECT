import sys
import os
import time
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.engine import MovieSearchEngine

# Configure pandas for clean output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

def test_query(engine, query, expected_top_title=None, description=""):
    print(f"\n🧪 TEST: {description}")
    print(f"🔎 Query: '{query}'")
    
    start = time.time()
    results = engine.search(query, k=5)
    duration = time.time() - start
    
    if results.empty:
        print("❌ No results found.")
        return

    # Define columns to display
    cols = ['title', 'startYear', 'final_score', 'hybrid_score', 'numVotes', 'averageRating']
    # Add match flags if they exist
    for col in ['actor_match_score', 'director_match_score']:
        if col in results.columns:
            cols.append(col)

    # Print top 5 results
    print("-" * 100)
    print(results[cols].to_string(index=False))
    print("-" * 100)
    
    # Validation Logic
    top_movie = results.iloc[0]
    top_title = top_movie['title']
    
    if expected_top_title:
        # Check if expected string is in the top title (case insensitive)
        if expected_top_title.lower() in top_title.lower():
            print(f"✅ PASS: Top result is '{top_title}'")
        else:
            print(f"⚠️ CHECK: Top result is '{top_title}' (Expected something like '{expected_top_title}')")
            
    print(f"⏱️ Time: {duration:.3f}s")

def main():
    print("🚀 Initializing Engine with NEW Config...")
    engine = MovieSearchEngine()
    print("✅ Engine Loaded.\n")

    # --- TEST CASE 1: The "Rise by Sin" Killer ---
    # Goal: Popularity (1.2) should help Iron Man crush the obscure movie
    test_query(
        engine, 
        "ironman", 
        expected_top_title="Iron Man", 
        description="Popularity Check: Does Iron Man (2008) beat Rise by Sin?"
    )

    # --- TEST CASE 2: The "Relevance" Defender ---
    # Goal: Hybrid weight (5.0) should keep Star Wars on top despite popularity
    test_query(
        engine, 
        "starwars", 
        expected_top_title="Star Wars", 
        description="Relevance Check: Does exact text match win?"
    )

    # --- TEST CASE 3: The "Adam Sandler" Logic ---
    # Goal: Actor Boost (10.0) should beat Title Match (1.0)
    test_query(
        engine, 
        "comedy with adam sandler", 
        expected_top_title=None, # Accepts any Sandler movie
        description="Entity Check: Does Actor match beat 'Stranded with Comedy'?"
    )
    
    # --- TEST CASE 4: Director Power ---
    test_query(
        engine,
        "movies directed by tarantino",
        expected_top_title=None,
        description="Director Check: Pulp Fiction / Reservoir Dogs"
    )

if __name__ == "__main__":
    main()