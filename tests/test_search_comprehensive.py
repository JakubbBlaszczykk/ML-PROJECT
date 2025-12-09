import sys
import os
import time
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.engine import MovieSearchEngine

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 50)

class SearchTester:
    def __init__(self):
        print("Initializing MovieSearchEngine...")
        start_time = time.time()
        self.engine = MovieSearchEngine()
        print(f"Engine initialized in {time.time() - start_time:.2f}s")
        print("=" * 100)

    def run_test(self, query: str, description: str, expected_top: str = None, k: int = 5):
        print(f"\nTEST: {description}")
        print(f"Query: '{query}'")
        
        start_time = time.time()
        try:
            results = self.engine.search(query, k=k)
            duration = time.time() - start_time
            
            if results.empty:
                print("No results found!")
                return
            
            # Display columns
            cols = ['title', 'startYear', 'final_score', 'hybrid_score', 'filter_boost']
            # Add match score columns if they exist and are non-zero
            for col in ['actor_match_score', 'director_match_score', 'genre_match_score']:
                if col in results.columns and results[col].sum() > 0:
                    cols.append(col)
            
            print(results[cols].head(k).to_string())
            print(f"\nSearch time: {duration:.4f}s")
            
            # Validation
            top_title = results.iloc[0]['title']
            if expected_top:
                if expected_top.lower() in top_title.lower():
                    print(f"PASSED: Top result '{top_title}' matches expected '{expected_top}'")
                else:
                    print(f"CHECK: Top result '{top_title}' != expected '{expected_top}'")
            else:
                print(f"Top result: {top_title}")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    tester = SearchTester()
    
    test_cases = [
        # --- 1. Exact Title Matches ---
        ("The Matrix", "Exact title match", "The Matrix"),
        ("Inception", "Exact title match", "Inception"),
        ("The Godfather", "Exact title match", "The Godfather"),
        
        # --- 2. Fuzzy Title Matches (Typos) ---
        ("avtar", "Title typo: avtar -> Avatar", "Avatar"),
        ("the matrx", "Title typo: matrx -> Matrix", "The Matrix"),
        ("godfther", "Title typo: godfther -> Godfather", "The Godfather"),
        ("fight club", "Title typo: club -> Club", "Fight Club"),
        
        # --- 3. Exact Person Matches ---
        ("Tom Hanks", "Actor search", "Larry Crowne"), # Or any Tom Hanks movie
        ("Christopher Nolan", "Director search", "Inception"),
        ("Quentin Tarantino", "Director search", "Pulp Fiction"),
        
        # --- 4. Fuzzy Person Matches ---
        ("tom hamks", "Actor typo: hamks -> Hanks", None),
        ("chris nolan", "Director nickname/typo", None),
        ("tarantino", "Director last name only", "Pulp Fiction"),
        ("brad pit", "Actor typo: pit -> Pitt", None),
        
        # --- 5. Person + Title ---
        ("tom cruise top gun", "Actor + Title", "Top Gun"),
        ("dicaprio inception", "Actor + Title", "Inception"),
        ("harrison ford indiana jones", "Actor + Title", "Indiana Jones"),
        
        # --- 6. Genre Filters ---
        ("action movies", "Genre filter: action", None),
        ("horror films", "Genre filter: horror", None),
        ("best comedies", "Genre filter: comedy + quality", None),
        
        # --- 7. Fuzzy Genre Filters ---
        ("horrer movies", "Genre typo: horrer -> horror", None),
        ("comdy films", "Genre typo: comdy -> comedy", None),
        ("sci-fi movies", "Genre alias: sci-fi -> sci-fi", None),
        
        # --- 8. Year Filters ---
        ("movies from 1999", "Specific year", None),
        ("films from the 90s", "Decade filter", None),
        ("movies after 2020", "Year range >", None),
        
        # --- 9. Complex Queries ---
        ("action movies with tom cruise", "Genre + Actor", None),
        ("sci-fi directed by nolan", "Genre + Director", "Inception"),
        ("90s comedy movies", "Decade + Genre", None),
        ("horror movies from 2022", "Genre + Year", None),
        
        # --- 10. Edge Cases ---
        ("", "Empty query", None),
        ("   ", "Whitespace query", None),
        ("!@#$%", "Special characters", None),
        ("asdfghjkl", "Nonsense query", None),
        ("the", "Stop word only", None),
        ("tOm HaNkS", "Mixed case", None),
        ("a" * 100, "Very long query", None),
    ]
    
    print(f"Running {len(test_cases)} test cases...")
    
    for query, desc, expected in test_cases:
        tester.run_test(query, desc, expected)
        
    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETED")
    print("=" * 100)

if __name__ == "__main__":
    main()
