"""
Enhanced test script for the Smart Query Parser with contextual keywords and genre aliases
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.query_parser import QueryParser
import json

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def test_enhanced_parser():
    """Test the enhanced query parser with contextual keywords and genre aliases."""
    
    print_section("ENHANCED SMART QUERY PARSER - TEST SUITE")
    
    # Initialize parser
    print("Initializing QueryParser...")
    parser = QueryParser()
    print()
    
    # Test 1: Contextual Keywords
    print_section("TEST 1: Contextual Keywords")
    
    contextual_tests = [
        {
            "query": "movies directed by Christopher Nolan",
            "description": "Using 'directed by' keyword"
        },
        {
            "query": "films starring Tom Hanks",
            "description": "Using 'starring' keyword"
        },
        {
            "query": "thriller directed by Quentin Tarantino",
            "description": "Director context + genre"
        },
        {
            "query": "action starring Tom Cruise",
            "description": "Actor context + genre"
        },
        {
            "query": "written by Aaron Sorkin",
            "description": "Using 'written by' keyword"
        },
    ]
    
    for test in contextual_tests:
        print(f"Query: '{test['query']}'")
        print(f"Description: {test['description']}")
        print("-" * 80)
        result = parser.parse_query(test['query'])
        
        print(f"Contextual role detected: {result.get('contextual_role_detected', False)}")
        for key in ['actor_filter', 'director_filter', 'writer_filter', 'genre_filter']:
            if result.get(key):
                print(f"  {key}: {result[key]}")
        print()
    
    # Test 2: Genre Aliases
    print_section("TEST 2: Genre Aliases")
    
    alias_tests = [
        {
            "query": "romantic comedy",
            "description": "Testing 'romantic' → 'romance' alias"
        },
        {
            "query": "scifi movies from 2010",
            "description": "Testing 'scifi' → 'sci-fi' alias"
        },
        {
            "query": "science fiction thriller",
            "description": "Testing 'science fiction' → 'sci-fi' alias"
        },
        {
            "query": "funny movie",
            "description": "Testing 'funny' → 'comedy' alias"
        },
        {
            "query": "scary movies",
            "description": "Testing 'scary' → 'horror' alias"
        },
        {
            "query": "animated film",
            "description": "Testing 'animated' → 'animation' alias"
        },
        {
            "query": "biographical drama",
            "description": "Testing 'biographical' → 'biography' alias"
        },
    ]
    
    for test in alias_tests:
        print(f"Query: '{test['query']}'")
        print(f"Description: {test['description']}")
        print("-" * 80)
        result = parser.parse_query(test['query'])
        
        if result.get('genre_filter'):
            print(f"  Genres detected: {result['genre_filter']}")
        else:
            print(f"  ❌ No genres detected")
        print()
    
    # Test 3: Multi-Role Handling
    print_section("TEST 3: Multi-Role Handling (People in Multiple Roles)")
    
    multi_role_tests = [
        {
            "query": "Quentin Tarantino films",
            "description": "No context - should show actor_or_director"
        },
        {
            "query": "directed by Quentin Tarantino",
            "description": "With context - should only be in director_filter"
        },
        {
            "query": "starring Quentin Tarantino",
            "description": "With context - should only be in actor_filter"
        },
        {
            "query": "Ben Affleck",
            "description": "No context - likely actor_or_director"
        },
        {
            "query": "Tom Hanks",
            "description": "No context - check if multi-role"
        },
    ]
    
    for test in multi_role_tests:
        print(f"Query: '{test['query']}'")
        print(f"Description: {test['description']}")
        print("-" * 80)
        result = parser.parse_query(test['query'])
        
        print(f"Contextual role detected: {result.get('contextual_role_detected', False)}")
        for key in ['actor_filter', 'director_filter', 'actor_or_director', 
                    'actor_or_writer', 'director_or_writer']:
            if result.get(key):
                print(f"  {key}: {result[key]}")
        print()
    
    # Test 4: Combined Features
    print_section("TEST 4: Combined Features (Context + Aliases + Years)")
    
    combined_tests = [
        {
            "query": "romantic comedy starring Tom Hanks from 1990s",
            "description": "Alias + Context + Decade"
        },
        {
            "query": "scifi directed by Christopher Nolan after 2010",
            "description": "Alias + Context + Year range"
        },
        {
            "query": "scary movie with good acting",
            "description": "Alias (scary → horror)"
        },
        {
            "query": "thriller directed by Quentin Tarantino between 1990 and 2000",
            "description": "Genre + Context + Year range"
        },
    ]
    
    for test in combined_tests:
        print(f"Query: '{test['query']}'")
        print(f"Description: {test['description']}")
        print("-" * 80)
        result = parser.parse_query(test['query'])
        
        print(f"Contextual role detected: {result.get('contextual_role_detected', False)}")
        for key in ['actor_filter', 'director_filter', 'writer_filter', 
                    'genre_filter', 'year_filter', 'actor_or_director']:
            if result.get(key):
                print(f"  {key}: {result[key]}")
        print()
    
    # Test 5: Original Test Cases (Regression)
    print_section("TEST 5: Regression Tests (Original Functionality)")
    
    regression_tests = [
        "Tom Hanks comedy",
        "action movie",
        "Christopher Nolan thriller",
        "horror movies from 1990s",
    ]
    
    for query in regression_tests:
        print(f"Query: '{query}'")
        print("-" * 80)
        result = parser.parse_query(query)
        
        for key, value in result.items():
            if value and key != 'search_term':
                print(f"  {key}: {value}")
        print()
    
    print_section("TEST SUITE COMPLETE")


if __name__ == "__main__":
    test_enhanced_parser()
