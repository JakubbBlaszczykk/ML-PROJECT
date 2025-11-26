import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.hybrid_search import HybridSearcher
import pandas as pd

def test_hybrid_search():
    print("Initializing HybridSearcher...")
    searcher = HybridSearcher()
    
    query = "top gun maverick"
    print(f"Searching for: '{query}'")
    
    results = searcher.search(query, k=10)
    
    print("\nTop 10 Results:")
    print(results[['title', 'score', 'bm25_score', 'sbert_score']])
    
    # Verification checks
    top_title = results.iloc[0]['title']
    print(f"\nTop result: {top_title}")
    
    if "Top Gun" in top_title:
        print("SUCCESS: Top result contains 'Top Gun'")
    else:
        print("FAILURE: Top result does not contain 'Top Gun'")

if __name__ == "__main__":
    test_hybrid_search()
