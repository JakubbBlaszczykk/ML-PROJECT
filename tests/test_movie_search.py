"""
Quick validation test for bug fixes
"""


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.search.engine import MovieSearchEngine
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("QUICK VALIDATION TEST - Bug Fixes")
print("=" * 100)

engine = MovieSearchEngine()

# Test 1: Avatar should show correct title
print("\n" + "=" * 100)
print("Test 1: 'avatar'")
print("=" * 100)
results = engine.search("avatar", k=3)
print(results[['title', 'final_score', 'hybrid_score', 'filter_boost', 'startYear']].to_string())
print(f"\nTop result: {results.iloc[0]['title']}")
print(f"Scores positive? {results.iloc[0]['final_score'] > 0}")

# Test 2: Tom Hanks should not match "Tom" movies
print("\n" + "=" * 100)
print("Test 2: 'Tom Hanks'")
print("=" * 100)
results = engine.search("Tom Hanks", k=5)
print(results[['title', 'final_score', 'hybrid_score', 'filter_boost', 'startYear']].to_string())

# Test 3: Action should not detect as actor
print("\n" + "=" * 100)
print("Test 3: 'action movie'")
print("=" * 100)
results = engine.search("action movie", k=5)
print(results[['title', 'final_score', 'hybrid_score', 'filter_boost', 'startYear']].to_string())

print("\n" + "=" * 100)
print("VALIDATION COMPLETE")
print("=" * 100)
