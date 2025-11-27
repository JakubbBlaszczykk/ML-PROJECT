"""
Test script for fuzzy matching in query parser.

Tests typo tolerance for:
- People names (actors/directors/writers)
- Genres
- Multi-candidate scenarios
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.search.query_parser import QueryParser

def test_fuzzy_matching():
    print("=" * 80)
    print("Testing Fuzzy Matching in Query Parser")
    print("=" * 80)
    
    parser = QueryParser()
    
    test_cases = [
        # People name typos
        {
            "query": "Tom Crus action movie",
            "expected_actor": "tom cruise",
            "description": "Single typo in last name"
        },
        {
            "query": "Christopher Nolan thriller",
            "expected_director": "christopher nolan",
            "description": "Exact match (should still work)"
        },
        {
            "query": "Kristofer Nolan",
            "expected_director": "christopher nolan",
            "description": "Multiple typos in first name"
        },
        {
            "query": "chris action movies",
            "description": "Ambiguous short name (should return multiple candidates)"
        },
        {
            "query": "tom hanks comedy",
            "expected_actor": "tom hanks",
            "description": "Multi-word exact match"
        },
        
        # Genre typos
        {
            "query": "horrer movie",
            "expected_genre": "horror",
            "description": "Genre typo"
        },
        {
            "query": "comady film",
            "expected_genre": "comedy",
            "description": "Another genre typo"
        },
        {
            "query": "action thriller",
            "expected_genre": "action",
            "description": "Multiple genres, exact match"
        },
        
        # Too short (should not fuzzy match)
        {
            "query": "tom",
            "description": "Too short - should not fuzzy match (< min_chars)"
        },
        
        # Combined cases
        {
            "query": "tom crus maverick",
            "expected_actor": "tom cruise",
            "description": "Name typo + title"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['description']}")
        print(f"Query: '{test['query']}'")
        print("-" * 80)
        
        result = parser.parse_query(test['query'])
        
        # Print results
        print(f"Parsed result:")
        for key, value in result.items():
            if value and key != 'search_term':
                print(f"  {key}: {value}")
        
        # Check expectations
        passed = True
        
        if 'expected_actor' in test:
            if test['expected_actor'] not in result.get('actor_filter', []):
                print(f"FAIL: Expected actor '{test['expected_actor']}' not found")
                passed = False
        
        if 'expected_director' in test:
            if test['expected_director'] not in result.get('director_filter', []):
                print(f"FAIL: Expected director '{test['expected_director']}' not found")
                passed = False
        
        if 'expected_genre' in test:
            if test['expected_genre'] not in result.get('genre_filter', []):
                print(f"FAIL: Expected genre '{test['expected_genre']}' not found")
                passed = False
        
        if passed:
            print("PASS")

if __name__ == "__main__":
    test_fuzzy_matching()
