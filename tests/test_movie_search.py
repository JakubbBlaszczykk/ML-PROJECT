"""
Batch validation test for movie search engine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.engine import MovieSearchEngine
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def run_validation_tests(test_queries=None, k=5):

    print("=" * 120)
    print("MOVIE SEARCH ENGINE — VALIDATION SUITE")
    print("=" * 120)

    engine = MovieSearchEngine()

    if test_queries is None:
        test_queries = [
            "avtar", "Tom crus top gun", "tom hamks", "actin movie",
            # Actors / directors misspelled 
            "tomm hanks", "jonny depp", "stevn spielbrg", "morgan freemn",
            "leonrdo dicaprio", "brad pit club", "emm watson fantasy",
            # Movie titles wrong or partial
            "incepton", "lord of te ring", "pirtes carbean", "harry poter 2",
            "marvl avngers", "batmn dark knight", "matrix reloed",
            "star warss new hope",
            # Genre / vague descriptors
            "space war film", "romantic comedy 2010s", "crime drama classic",
            "sc-fi robot movie", "magic wizard film series",
            "True story survival movie",
            # Multi-token queries
            "space crew trapped", "time travel paradox", "plane crash survival film",
            "haunted house horror", "spy thriller cold war",
            # Actor + genre
            "tom hanks war movie", "tom cruise mission impossible",
            "robert downey jr iron man", "keanu reeves action",
            # Sequel/grouping
            "matrix trilogy", "avengers infinity", "x men first", "toy story pixar",
            # Year-based
            "movies 1999", "action 1980s", "classic movies 1975",
            "sci fi 2020", "oscar winners 2014",
            # Hard noisy queries
            "vilot purlpe girl kidnap magic", "robot boy future save world",
            "guy wakes no memory cyberpunk", "animated lion savannah animal",
            # Very short queries (edge cases)
            "war", "love", "future", "space", "magic", "spy",
            # Mixed bag
            "older tom cruise jets movie", "hacker simulation reality",
            "movie with blue aliens", "guy stuck on mars potatoes",
            "ship iceberg sink romance"
        ]

    engine = MovieSearchEngine()

    for i, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 120)
        print(f" Test {i}: '{query}' ")
        print("=" * 120)

        try:
            results = engine.search(query, k=k)
            if results is None or len(results) == 0:
                print("⚠ No results returned")
                continue

            cols = ['title', 'final_score', 'hybrid_score', 'filter_boost', 'startYear']
            print(results[cols].to_string(index=False))
            print(f" → Top result: {results.iloc[0]['title']} (score={results.iloc[0]['final_score']:.4f})")

        except Exception as e:
            print(f"❌ Error while processing query '{query}': {e}")

    print("\n" + "=" * 120)
    print("VALIDATION SUITE COMPLETE")
    print("=" * 120)



# Run tests when executed directly
if __name__ == "__main__":
    run_validation_tests()