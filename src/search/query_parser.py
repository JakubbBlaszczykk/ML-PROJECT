"""
Smart Query Parser for Movie Search Engine

This module provides intelligent query parsing that extracts filters (actors, directors, 
writers, genres, years) from user queries while preserving important search terms.
"""

import polars as pl
import re
import pickle
import os
from typing import Dict, List, Optional, Set
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.config import get_config
from pathlib import Path


# Contextual keywords for role detection
ACTOR_KEYWORDS = [
    # Primary keywords
    'starring', 'stars', 'star', 'acted by', 'acts in', 'acting',
    'with', 'featuring', 'features', 'played by', 'performance by',
    # Aliases
    'cast member', 'cast', 'actor', 'actress', 'lead', 'leading role',
]

DIRECTOR_KEYWORDS = [
    # Primary keywords
    'directed by', 'director', 'directs', 'direction by', 'directed',
    'helmed by', 'helm by', 'from director',
    # Aliases
    'filmmaker', 'film by', 'movie by', 'by director',
]

WRITER_KEYWORDS = [
    # Primary keywords
    'written by', 'writer', 'writes', 'screenplay by', 'script by',
    'penned by', 'story by', 'authored by',
    # Aliases
    'screenwriter', 'scribe', 'from writer',
]

# Genre aliases - map variations to canonical genre names
GENRE_ALIASES = {
    # Romance variations
    'romantic': 'romance',
    'love story': 'romance',
    'love': 'romance',
    
    # Sci-Fi variations
    'scifi': 'sci-fi',
    'science fiction': 'sci-fi',
    'science-fiction': 'sci-fi',
    'sf': 'sci-fi',
    
    # Action variations
    'actionmovie': 'action',
    
    # Comedy variations
    'comedic': 'comedy',
    'funny': 'comedy',
    'humorous': 'comedy',
    
    # Drama variations
    'dramatic': 'drama',
    
    # Horror variations
    'scary': 'horror',
    'frightening': 'horror',
    
    # Thriller variations
    'suspense': 'thriller',
    'suspenseful': 'thriller',
    
    # Crime variations
    'criminal': 'crime',
    
    # Documentary variations
    'doc': 'documentary',
    'doco': 'documentary',
    
    # Animation variations
    'animated': 'animation',
    'cartoon': 'animation',
    
    # Mystery variations
    'mysterious': 'mystery',
    'whodunit': 'mystery',
    
    # Adventure variations
    'adventurous': 'adventure',
    
    # Fantasy variations
    'fantastical': 'fantasy',
    
    # Western variations
    'cowboy': 'western',
    'wild west': 'western',
    
    # War variations
    'wartime': 'war',
    'war film': 'war',
    
    # Biography variations
    'biographical': 'biography',
    'biopic': 'biography',
    'bio': 'biography',
    
    # Musical variations
    'music': 'musical',
}
# Trie implementation for fast keyword matching
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word

    def search_in_text(self, text):
        """
        Find all occurrences of words in the trie within the text.
        Returns a set of found words.
        """
        found_words = set()
        n = len(text)
        
        # Iterate through each character in text as a potential start of a word
        # This is a simplified approach; Aho-Corasick would be faster but more complex to implement from scratch
        # Given the constraints, this is still much faster than regex for 100k patterns
        for i in range(n):
            node = self.root
            for j in range(i, n):
                char = text[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end_of_word:
                    # Verify word boundary to avoid partial matches inside other words
                    # Check previous char
                    if i > 0 and text[i-1].isalnum():
                        continue
                    # Check next char
                    if j < n - 1 and text[j+1].isalnum():
                        continue
                    found_words.add(node.word)
        
        return found_words

class QueryParser:
    """
    Intelligently parses search queries to extract filters and search terms.
    
    The parser creates lookup sets from movie data and matches query terms against
    known entities (actors, directors, writers, genres). It maintains partial retention
    of detected terms in the search query to handle cases like "action movie" which
    should match both genre:action and title containing "action".
    """
    
    def __init__(self, data_path: str = None, force_rebuild: bool = False):
        """
        Initialize the query parser by creating or loading lookup sets.
        
        Args:
            data_path: Path to the parquet file containing movie data
            force_rebuild: If True, rebuild lookup sets even if cache exists
        """
        # Use absolute paths based on project_root defined at module level
        if data_path is None:
            data_path = os.path.join(project_root, "data/imdb_us_movies_merged.parquet")
        print("Initializing QueryParser...")
        
        # Load configuration
        self.config = get_config()
        
        # Define cache path
        cache_dir = Path.home() / ".gemini" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "query_parser_lookups.pkl"
        
        # Check if we can use cached lookup sets
        use_cache = False
        if not force_rebuild and cache_path.exists():
            # Check if cache is newer than source data
            try:
                cache_time = os.path.getmtime(cache_path)
                data_time = os.path.getmtime(data_path)
                if cache_time > data_time:
                    use_cache = True
                    print(f"Loading lookup sets from cache: {cache_path}")
                else:
                    print("Cache is outdated, rebuilding lookup sets...")
            except Exception as e:
                print(f"Cache check failed: {e}, rebuilding...")
        
        if use_cache:
            # Load from cache
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.actors_set = cached_data['actors']
                self.directors_set = cached_data['directors']
                self.writers_set = cached_data['writers']
                self.primary_roles = cached_data['primary_roles']
                self.genres_set = cached_data['genres']
                
                # Load and normalize popularity scores
                popularity_scores = cached_data.get('popularity_scores', {})
                if popularity_scores:
                    max_popularity = max(popularity_scores.values())
                    self.person_popularity = {
                        name: count / max_popularity 
                        for name, count in popularity_scores.items()
                    }
                else:
                    self.person_popularity = {}
                
                # Always load popular titles even when using cache
                # This is needed for title fuzzy matching
                if self.config.fuzzy_matching.get('title_fuzzy_enabled', True):
                    print("Loading data for title fuzzy matching...")
                    # Need to load the dataframe to extract popular titles
                    self.df = pl.read_parquet(data_path) 
                    self._load_popular_titles()
                else:
                    self.popular_titles = set()
                    self.normalized_titles = {}
                
                print(f"✓ Loaded {len(self.actors_set)} unique actors from cache")
                print(f"✓ Loaded {len(self.directors_set)} unique directors from cache")
                print(f"✓ Loaded {len(self.writers_set)} unique writers from cache")
                print(f"✓ Loaded {len(self.directors_set)} unique directors from cache")
                print(f"✓ Loaded {len(self.writers_set)} unique writers from cache")
                print(f"✓ Loaded {len(self.genres_set)} unique genres from cache")
                if self.config.fuzzy_matching.get('title_fuzzy_enabled', True):
                    print(f"✓ Loaded {len(self.popular_titles)} popular titles for fuzzy matching")
                
                # Rebuild Trie (it's fast enough to not need pickling, or we could pickle it too)
                # Pickling recursive objects can be tricky/large, rebuilding is safer
                print("Building Trie for fast keyword matching...")
                self.trie = Trie()
                all_names = self.actors_set.union(self.directors_set).union(self.writers_set)
                
                min_words = self.config.query_parser['min_name_words']
                min_length = self.config.query_parser['min_single_word_length']
                
                count = 0
                for n in all_names:
                    word_count = len(n.split())
                    if word_count >= min_words or len(n) >= min_length:
                        self.trie.insert(n)
                        count += 1
                print(f"✓ Added {count} names to Trie")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print("Rebuilding lookup sets...")
                use_cache = False
        
        if not use_cache:
            # Build from scratch
            print(f"Loading data from: {data_path}")
            self.df = pl.read_parquet(data_path)
            
            print("Creating lookup sets...")
            people_roles = self._extract_people_with_roles(self.df)
            
            self.actors_set = people_roles['actors']
            self.directors_set = people_roles['directors']
            self.writers_set = people_roles['writers']
            self.primary_roles = people_roles['primary_roles']
            self.genres_set = self._extract_genres(self.df)
            
            # Store and normalize popularity scores for fuzzy matching
            popularity_scores = people_roles['popularity_scores']
            max_popularity = max(popularity_scores.values()) if popularity_scores else 1
            self.person_popularity = {
                name: count / max_popularity 
                for name, count in popularity_scores.items()
            }
            
            # Load popular titles for fuzzy matching
            if self.config.fuzzy_matching.get('title_fuzzy_enabled', True):
                self._load_popular_titles()
            else:
                self.popular_titles = set()
                self.normalized_titles = {}
            
            print(f"✓ Created {len(self.actors_set)} unique actors")
            print(f"✓ Created {len(self.directors_set)} unique directors")
            print(f"✓ Created {len(self.writers_set)} unique writers")
            print(f"✓ Created {len(self.genres_set)} unique genres")
            if self.config.fuzzy_matching.get('title_fuzzy_enabled', True):
                print(f"✓ Loaded {len(self.popular_titles)} popular titles for fuzzy matching")
            
            # Build Trie for fast searching
            print("Building Trie for fast keyword matching...")
            self.trie = Trie()
            all_names = self.actors_set.union(self.directors_set).union(self.writers_set)
            
            # Filter names to avoid false positives using config settings
            min_words = self.config.query_parser['min_name_words']
            min_length = self.config.query_parser['min_single_word_length']
            
            count = 0
            for n in all_names:
                word_count = len(n.split())
                if word_count >= min_words or len(n) >= min_length:
                    self.trie.insert(n)
                    count += 1
            print(f"✓ Added {count} names to Trie")
            
            # Save to cache
            try:
                # Get raw popularity scores before normalization for caching
                popularity_scores = people_roles['popularity_scores']
                cache_data = {
                    'actors': self.actors_set,
                    'directors': self.directors_set,
                    'writers': self.writers_set,
                    'primary_roles': self.primary_roles,
                    'genres': self.genres_set,
                    'popularity_scores': popularity_scores
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"✓ Saved lookup sets to cache: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
        
        print("QueryParser ready!")
    
    def _extract_people_with_roles(self, df: pl.DataFrame) -> Dict:
        """
        Extract people names and count their appearances in each role using optimized Polars operations.
        """
        # Helper to extract and count names for a role
        def get_role_counts(col_name, role_name):
            return (
                df.select(pl.col(col_name))
                .explode(col_name)
                .drop_nulls()
                .select(pl.col(col_name).struct.field("primaryName").alias("name"))
                .drop_nulls()
                .with_columns(pl.col("name").str.to_lowercase())
                .group_by("name")
                .count()
                .rename({"count": f"{role_name}_count"})
            )

        # Get counts for each role
        actor_counts = get_role_counts("cast", "actor")
        director_counts = get_role_counts("directors", "director")
        writer_counts = get_role_counts("writers", "writer")

        # Join all counts
        # We start with all unique names
        all_names = (
            actor_counts.select("name")
            .vstack(director_counts.select("name"))
            .vstack(writer_counts.select("name"))
            .unique()
        )

        # Join counts back
        combined = (
            all_names
            .join(actor_counts, on="name", how="left")
            .join(director_counts, on="name", how="left")
            .join(writer_counts, on="name", how="left")
            .fill_null(0)
        )

        # Convert to dictionary for fast lookup
        # This is much faster than iterating rows
        result_dicts = combined.to_dicts()
        
        actors_set = set()
        directors_set = set()
        writers_set = set()
        primary_roles = {}
        popularity_scores = {}
        
        for row in result_dicts:
            name = row['name']
            a_count = row['actor_count']
            d_count = row['director_count']
            w_count = row['writer_count']
            
            if a_count > 0: actors_set.add(name)
            if d_count > 0: directors_set.add(name)
            if w_count > 0: writers_set.add(name)
            
            # Determine primary role
            counts = {'actor': a_count, 'director': d_count, 'writer': w_count}
            primary_roles[name] = max(counts.items(), key=lambda x: x[1])[0]
            
            # Store total count for popularity scoring
            popularity_scores[name] = a_count + d_count + w_count
            
        return {
            'actors': actors_set,
            'directors': directors_set,
            'writers': writers_set,
            'primary_roles': primary_roles,
            'popularity_scores': popularity_scores
        }
    
    def _extract_genres(self, df: pl.DataFrame) -> Set[str]:
        """Extract unique genres from the dataframe."""
        genres = set()
        
        # Get unique genre strings (comma-separated)
        genre_strings = df.select('genres').unique().to_series().to_list()
        
        for genre_str in genre_strings:
            if genre_str and isinstance(genre_str, str):
                # Split comma-separated genres
                for genre in genre_str.split(','):
                    genre = genre.strip()
                    if genre:
                        genres.add(genre.lower())
        
        return genres
    
    def _load_popular_titles(self):
        """
        Load top N popular movie titles for fuzzy matching.
        Popularity is based on numVotes.
        """
        limit = self.config.fuzzy_matching.get('title_popularity_limit', 10000)
        
        # Get top N movies by numVotes
        top_movies = (
            self.df
            .select(['primaryTitle', 'numVotes'])
            .sort('numVotes', descending=True)
            .limit(limit)
        )
        
        # Extract titles and create normalized versions
        self.popular_titles = set(top_movies['primaryTitle'].to_list())
        
        # Store normalized versions for fuzzy matching
        # This allows "avtar" to match "Avatar" by comparing normalized forms
        from src.data.custom_transformers import SearchCorpusGenerator
        corpus_gen = SearchCorpusGenerator()
        
        self.normalized_titles = {}
        for title in self.popular_titles:
            normalized = corpus_gen._normalize_text(title)
            self.normalized_titles[normalized] = title
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse a user query to extract filters and search terms.
        
        The function intelligently detects:
        - Actor names (e.g., "Tom Hanks")
        - Director names (e.g., "Christopher Nolan")
        - Writer names
        - Genres (e.g., "comedy", "action")
        - Year patterns (e.g., "2020", "1990s", "before 2000")
        
        Important: Detected terms are kept in the search_term with partial retention.
        For example, "action movie" returns both genre_filter:action AND 
        search_term:"action movie" to handle cases where "action" might be in the title.
        
        Args:
            query: The raw user query string
            
        Returns:
            Dictionary with:
                - search_term: Cleaned search text (keeps some detected terms)
                - actor_filter: List of detected actors
                - director_filter: List of detected directors
                - writer_filter: List of detected writers
                - genre_filter: List of detected genres
                - year_filter: Dict with year constraints (e.g., {'min': 2000, 'max': 2020})
        """
        result = {
            'search_term': query.strip(),
            'actor_filter': [],
            'director_filter': [],
            'writer_filter': [],
            'genre_filter': [],
            'year_filter': None,
            # Multi-role fields for ambiguous classifications
            'actor_or_director': [],
            'actor_or_writer': [],
            'director_or_writer': [],
            'contextual_role_detected': False  # Flag if context keywords were used
        }
        
        query_lower = query.lower()
        
        # Track what we've found to avoid duplicates
        found_tokens = set()
        
        # 1. Extract year patterns
        year_info = self._extract_years(query)
        if year_info:
            result['year_filter'] = year_info
        
        # 2. Detect contextual keywords to determine role preferences
        contextual_roles = self._detect_contextual_roles(query)
        if contextual_roles:
            result['contextual_role_detected'] = True
        
        # 3. Check for genres FIRST (to mark these tokens as used)
        result['genre_filter'] = self._find_genres_in_query(query_lower, found_tokens)
        
        # 4. Check for multi-word people names
        # Find all people mentioned
        all_people = self._find_all_people_in_query(query_lower, found_tokens)
        
        # Build corrected_search_term for BM25/SBERT by replacing fuzzy-matched typos
        # This ensures "tom hamks" becomes "tom hanks" for semantic search
        corrected_query = query_lower
        replacements = []  # Track (original_span, corrected_name) for replacement
        
        # Collect all fuzzy matches that need replacement
        # We need to replace the original query tokens with the corrected names
        for person in all_people:
            # person is the corrected name (e.g., "tom cruise")
            # We need to find what query tokens matched to it
            # For now, if the person name is not in the original query, it was fuzzy matched
            if person not in query_lower:
                # This was a fuzzy match - we should replace the typo
                # Find the most likely tokens that got fuzzy matched
                # This is tricky - for simplicity, just add the corrected name
                corrected_query += f" {person}"
        
        # Add fuzzy-matched titles to corrected query
        # This allows "avtar" to match "Avatar" via BM25/SBERT
        query_tokens = query_lower.split()
        matched_titles = self._fuzzy_find_titles(query_tokens, found_tokens)
        for title in matched_titles:
            corrected_query += f" {title}"
       
        result['corrected_search_term'] = corrected_query
        
        # Classify based on contextual keywords OR primary role
        for person_name in all_people:
            # Check if context specifies the role
            contextual_role = contextual_roles.get(person_name) if contextual_roles else None
            
            if contextual_role:
                # Use contextual role (overrides primary role)
                if contextual_role == 'actor':
                    result['actor_filter'].append(person_name)
                elif contextual_role == 'director':
                    result['director_filter'].append(person_name)
                elif contextual_role == 'writer':
                    result['writer_filter'].append(person_name)
            else:
                # No context - check if person has multiple roles
                roles = self._get_person_roles(person_name)
                
                if len(roles) == 1:
                    # Single role - use it
                    role = roles[0]
                    if role == 'actor':
                        result['actor_filter'].append(person_name)
                    elif role == 'director':
                        result['director_filter'].append(person_name)
                    elif role == 'writer':
                        result['writer_filter'].append(person_name)
                else:
                    # Multiple roles - use primary role but also add to ambiguous field
                    primary_role = self.primary_roles.get(person_name, 'actor')
                    
                    # Add to primary role
                    if primary_role == 'actor':
                        result['actor_filter'].append(person_name)
                    elif primary_role == 'director':
                        result['director_filter'].append(person_name)
                    elif primary_role == 'writer':
                        result['writer_filter'].append(person_name)
                    
                    # Also add to ambiguous fields
                    if 'actor' in roles and 'director' in roles:
                        result['actor_or_director'].append(person_name)
                    if 'actor' in roles and 'writer' in roles:
                        result['actor_or_writer'].append(person_name)
                    if 'director' in roles and 'writer' in roles:
                        result['director_or_writer'].append(person_name)
        
        # 5. Keep search term unchanged for BM25/SBERT matching
        # Do NOT remove detected filters from query
        # The full query text is needed for searchable_text matching
        result['search_term'] = query.strip()
        
        return result
    
    def _detect_contextual_roles(self, query: str) -> Optional[Dict[str, str]]:
        """
        Detect contextual keywords that specify roles (e.g., "directed by", "starring").
        
        Args:
            query: The query string
            
        Returns:
            Dictionary mapping person names to their contextual roles, or None
        """
        query_lower = query.lower()
        contextual_roles = {}
        
        # Check for each role type
        role_keywords = [
            ('actor', ACTOR_KEYWORDS),
            ('director', DIRECTOR_KEYWORDS),
            ('writer', WRITER_KEYWORDS)
        ]
        
        for role, keywords in role_keywords:
            for keyword in keywords:
                # Look for pattern: "[keyword] [person name]"
                pattern = r'\b' + re.escape(keyword) + r'\s+([a-z\s]+?)(?:\s+(?:and|,|in|from|\d)|$)'
                matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                
                for match in matches:
                    potential_name = match.group(1).strip()
                    
                    # Check if this matches a known person
                    if potential_name in self.actors_set.union(self.directors_set).union(self.writers_set):
                        contextual_roles[potential_name] = role
        
        return contextual_roles if contextual_roles else None
    
    def _get_person_roles(self, person_name: str) -> List[str]:
        """
        Get all roles a person has in the database.
        
        Args:
            person_name: Person's name (lowercase)
            
        Returns:
            List of roles: ['actor', 'director', 'writer']
        """
        roles = []
        if person_name in self.actors_set:
            roles.append('actor')
        if person_name in self.directors_set:
            roles.append('director')
        if person_name in self.writers_set:
            roles.append('writer')
        return roles
    
    def _fuzzy_find_people(self, query_token: str, exclude_found: Set[str]) -> List[str]:
        """
        Find people names using fuzzy matching with RapidFuzz.
        
        Args:
            query_token: The token to match
            exclude_found: Set of already found names to exclude
            
        Returns:
            List of matched names (can be multiple if above threshold)
        """
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            # RapidFuzz not available, skip fuzzy matching
            return []
        
        # Check if fuzzy matching is enabled
        if not self.config.fuzzy_matching['enabled']:
            return []
        
        # Check minimum length
        if len(query_token) < self.config.fuzzy_matching['people_min_chars']:
            return []
        
        all_people = self.actors_set | self.directors_set | self.writers_set
        
        # Exclude already found names
        searchable = all_people - exclude_found
        
        if not searchable:
            return []
        
        # Use WRatio for better partial matching
        threshold = self.config.fuzzy_matching['people_min_similarity'] * 100
        
        # Get more candidates than needed for filtering
        max_candidates = self.config.fuzzy_matching['people_max_candidates']
        results = process.extract(
            query_token,
            searchable,
            scorer=fuzz.WRatio,
            limit=max_candidates * 3  # Get more, filter later
        )
        
        # Filter by threshold
        candidates = [(name, score/100) for name, score, _ in results if score >= threshold]
        
        if not candidates:
            return []
        
        # Apply popularity weighting if enabled
        if self.config.fuzzy_matching['use_popularity_weighting']:
            candidates = self._apply_popularity_weighting(candidates)
        
        # Return based on config
        if self.config.fuzzy_matching['return_all_above_threshold']:
            return [name for name, score in candidates]
        else:
            # Return top N
            return [name for name, score in candidates[:max_candidates]]
    
    def _apply_popularity_weighting(self, candidates: List[tuple]) -> List[tuple]:
        """
        Weight fuzzy match candidates by popularity (based on role counts).
        
        Args:
            candidates: List of (name, fuzzy_score) tuples
            
        Returns:
            List of (name, combined_score) tuples, sorted by combined score
        """
        weighted = []
        popularity_weight = self.config.fuzzy_matching['popularity_weight']
        
        for name, fuzzy_score in candidates:
            popularity = self.person_popularity.get(name, 0.0)
            # Weighted average
            combined_score = (1 - popularity_weight) * fuzzy_score + popularity_weight * popularity
            weighted.append((name, combined_score))
        
        return sorted(weighted, key=lambda x: x[1], reverse=True)
    
    def _fuzzy_find_genre(self, query_token: str) -> Optional[str]:
        """
        Fuzzy match genre with strict threshold.
        
        Args:
            query_token: The token to match
            
        Returns:
            Best matching genre name or None
        """
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return None
        
        # Check if fuzzy matching is enabled
        if not self.config.fuzzy_matching['enabled']:
            return None
        
        if len(query_token) < self.config.fuzzy_matching['genre_min_chars']:
            return None
        
        threshold = self.config.fuzzy_matching['genre_min_similarity'] * 100
        
        result = process.extractOne(
            query_token,
            self.genres_set,
            scorer=fuzz.ratio,  # Use exact ratio for short genre names
            score_cutoff=threshold
        )
        
        if result:
            return result[0]  # Return genre name
        return None
    
    def _fuzzy_find_titles(self, query_tokens: List[str], found_tokens: Set[str]) -> List[str]:
        """
        Fuzzy match query tokens against popular movie titles.
        
        Args:
            query_tokens: List of query tokens
            found_tokens: Set of tokens already matched to people/genres
            
        Returns:
            List of matched title strings
        """
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return []
        
        # Check if title fuzzy matching is enabled
        if not self.config.fuzzy_matching.get('title_fuzzy_enabled', True):
            return []
        
        matched_titles = []
        min_chars = self.config.fuzzy_matching.get('title_fuzzy_min_chars', 3)
        threshold = self.config.fuzzy_matching.get('title_fuzzy_min_similarity', 0.85) * 100
        
        # Try to match unmatched tokens against titles
        for token in query_tokens:
            # Skip if already used for people/genres
            if token in found_tokens:
                continue
            
            # Skip very short tokens
            if len(token) < min_chars:
                continue
            
            # Fuzzy match against normalized titles
            result = process.extractOne(
                token,
                self.normalized_titles.keys(),
                scorer=fuzz.ratio,
                score_cutoff=threshold
            )
            
            if result:
                normalized_title, score, _ = result
                original_title = self.normalized_titles[normalized_title]
                matched_titles.append(original_title.lower())
                # Mark token as used
                found_tokens.add(token)
        
        return matched_titles
    
    def _find_all_people_in_query(self, query_lower: str, found_tokens: Set[str]) -> List[str]:
        """
        Find all people names (actors, directors, writers) in the query string.
        Uses Trie for exact matching, then fuzzy matching ONLY for multi-word names (2+ words).
        Single-word fuzzy matching is disabled to prevent false positives.
        """
        matched_people = []
        
        # Step 1: Use Trie to find exact matches
        found_names = self.trie.search_in_text(query_lower)
        
        # Sort by length (longest first) to match multi-word names before single words
        sorted_names = sorted(list(found_names), key=len, reverse=True)
        
        for name in sorted_names:
            # Check if not already found (e.g. part of another name)
            name_words = set(name.split())
            if not name_words.intersection(found_tokens):
                matched_people.append(name)
                found_tokens.update(name_words)
        
        # Step 2: Try fuzzy matching ONLY on consecutive token pairs (multi-word names)
        # This prevents false positives from single-word matches like "o", "cru", "ham"
        query_tokens = query_lower.split()
        
        # Try matching consecutive token pairs (for "tom crus" → "tom cruise")
        for i in range(len(query_tokens) - 1):
            token1 = query_tokens[i]
            token2 = query_tokens[i + 1]
            
            # Skip if either token is already used
            if token1 in found_tokens or token2 in found_tokens:
                continue
            
            # Try matching the two-word combination
            two_word_query = f"{token1} {token2}"
            
            # Skip if too short
            if len(two_word_query) < self.config.fuzzy_matching['people_min_chars']:
                continue
            
            # Try fuzzy matching on the two-word combination
            fuzzy_matches = self._fuzzy_find_people(two_word_query, found_tokens)
            
            if fuzzy_matches:
                # Check all fuzzy matches and accept the first that passes validation
                # This is needed because top fuzzy match might not have correct token alignment
                for name in fuzzy_matches:
                    # Only accept multi-word results (filter out single-word names)
                    if len(name.split()) >= 2:
                        # Additional validation: ensure query tokens reasonably match name tokens
                        # This prevents "top gun" → "hot glue gun" but allows "tom crus" → "tom cruise"
                        name_tokens = name.split()
                        
                        # Simple heuristic: first chars of query tokens should match first chars of name tokens
                        # Use flexible matching to handle typos
                        try:
                            from rapidfuzz import fuzz
                            # Check if first query token fuzzy matches first name token
                            # and last query token fuzzy matches last name token
                            first_match = fuzz.partial_ratio(token1, name_tokens[0]) >= 70
                            last_match = fuzz.partial_ratio(token2, name_tokens[-1]) >= 70
                            
                            if first_match and last_match:
                                # Found a valid match!
                                matched_people.append(name)
                                found_tokens.update(name.split())
                                # Mark the query tokens as used
                                found_tokens.add(token1)
                                found_tokens.add(token2)
                                break  # Only take first valid match
                        except ImportError:
                            # Fallback if rapidfuzz not available - just accept the match
                            matched_people.append(name)
                            found_tokens.update(name.split())
                            found_tokens.add(token1)
                            found_tokens.add(token2)
                            break
        
        # NOTE: Single-token fuzzy matching is DISABLED to prevent false positives
        # like "o", "cru", "ham" being matched to random people
        
        return matched_people
    
    def _find_genres_in_query(self, query_lower: str, found_tokens: Set[str]) -> List[str]:
        """
        Find genre keywords in the query string, including aliases.
        Uses exact matching first, then fuzzy matching for unmatched tokens.
        
        Args:
            query_lower: Lowercase query string
            found_tokens: Set to track found tokens
            
        Returns:
            List of matched genres (canonical names)
        """
        matched_genres = set()  # Use set to avoid duplicates
        matched_tokens = set()  # Track which tokens were matched
        
        # Step 1: Check genre aliases (exact matching)
        for alias, canonical_genre in GENRE_ALIASES.items():
            # Use word boundaries for multi-word aliases
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, query_lower):
                # Check if not already found as part of a person's name
                if alias not in found_tokens:
                    # Check if canonical genre exists in our genre set
                    if canonical_genre in self.genres_set:
                        matched_genres.add(canonical_genre)
                        found_tokens.add(alias)
                        matched_tokens.add(alias)
        
        # Step 2: Check actual genre names (exact matching)
        for genre in self.genres_set:
            # Use word boundaries
            pattern = r'\b' + re.escape(genre) + r'\b'
            if re.search(pattern, query_lower):
                # Check if not already found as part of a person's name
                if genre not in found_tokens:
                    matched_genres.add(genre)
                    found_tokens.add(genre)
                    matched_tokens.add(genre)
        
        # Step 3: Try fuzzy matching for unmatched tokens
        query_tokens = query_lower.split()
        
        # Try consecutive token pairs first (for multi-word genre misspellings)
        for i in range(len(query_tokens) - 1):
            token1 = query_tokens[i]
            token2 = query_tokens[i + 1]
            
            # Skip if already matched or used
            if token1 in found_tokens or token2 in found_tokens:
                continue
            if token1 in matched_tokens or token2 in matched_tokens:
                continue
            
            # Try two-word combination
            two_word_query = f"{token1} {token2}"
            
            if len(two_word_query) >= self.config.fuzzy_matching['genre_min_chars']:
                fuzzy_genre = self._fuzzy_find_genre(two_word_query)
                
                if fuzzy_genre:
                    matched_genres.add(fuzzy_genre)
                    found_tokens.add(token1)
                    found_tokens.add(token2)
                    matched_tokens.add(token1)
                    matched_tokens.add(token2)
        
        # Then try single tokens
        for token in query_tokens:
            # Skip if token already matched or used for people
            if token in found_tokens or token in matched_tokens:
                continue
            
            # Skip very short tokens
            if len(token) < self.config.fuzzy_matching['genre_min_chars']:
                continue
            
            # Skip tokens that look like they might be part of a person's name
            # (i.e., capitalized or similar to known person words)
            # This is a heuristic to avoid matching "horrer" to people with "movie" in their name
            
            # Try fuzzy genre matching
            fuzzy_genre = self._fuzzy_find_genre(token)
            
            if fuzzy_genre:
                matched_genres.add(fuzzy_genre)
                found_tokens.add(token)
        
        return list(matched_genres)
    
    def _extract_years(self, query: str) -> Optional[Dict]:
        """
        Extract year constraints from the query.
        
        Patterns supported:
        - Single year: "2020", "1995"
        - Decade (full): "1990s", "2000s"
        - Decade (short): "90s", "80s", "70s" (assumes 19xx for 20-99, 20xx for 00-19)
        - Range: "2000-2010", "between 2000 and 2010"
        - Comparative: "before 2000", "after 1995", "since 2010"
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with 'min' and/or 'max' year constraints, or None
        """
        year_info = {}
        
        # Single year (e.g., "2020")
        single_year = re.search(r'\b(19\d{2}|20\d{2})\b', query)
        if single_year:
            year = int(single_year.group(1))
            year_info['exact'] = year
            return year_info
        
        # Short decade pattern (e.g., "90s", "80s", "00s")
        # Assumes 19xx for decades 20-99 and 20xx for decades 00-19
        short_decade = re.search(r'\b(\d{2})s\b', query)
        if short_decade:
            decade_suffix = int(short_decade.group(1))
            # Determine century based on decade
            if decade_suffix >= 20:
                # "90s" -> 1990s, "80s" -> 1980s, etc.
                decade_start = 1900 + decade_suffix
            else:
                # "00s" -> 2000s, "10s" -> 2010s
                decade_start = 2000 + decade_suffix
            year_info['min'] = decade_start
            year_info['max'] = decade_start + 9
            return year_info
        
        # Full decade (e.g., "1990s", "2000s")
        decade = re.search(r'\b(19\d0|20\d0)s\b', query)
        if decade:
            decade_start = int(decade.group(1))
            year_info['min'] = decade_start
            year_info['max'] = decade_start + 9
            return year_info
        
        # Range patterns (more flexible regex)
        range_pattern = re.search(r'\b(19\d{2}|20\d{2})\s*[-–—]\s*(19\d{2}|20\d{2})\b', query)
        if range_pattern:
            year_info['min'] = int(range_pattern.group(1))
            year_info['max'] = int(range_pattern.group(2))
            return year_info
        
        # "to" pattern
        to_pattern = re.search(r'\b(19\d{2}|20\d{2})\s+to\s+(19\d{2}|20\d{2})\b', query, re.IGNORECASE)
        if to_pattern:
            year_info['min'] = int(to_pattern.group(1))
            year_info['max'] = int(to_pattern.group(2))
            return year_info
        
        # "between X and Y"
        between_pattern = re.search(r'between\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2})', query, re.IGNORECASE)
        if between_pattern:
            year_info['min'] = int(between_pattern.group(1))
            year_info['max'] = int(between_pattern.group(2))
            return year_info
        
        # "before X" / "until X"
        before_pattern = re.search(r'(?:before|until)\s+(19\d{2}|20\d{2})', query, re.IGNORECASE)
        if before_pattern:
            year_info['max'] = int(before_pattern.group(1)) - 1  # Before means up to year-1
            return year_info
        
        # "after X" / "since X"
        after_pattern = re.search(r'(?:after|since)\s+(19\d{2}|20\d{2})', query, re.IGNORECASE)
        if after_pattern:
            year_info['min'] = int(after_pattern.group(1)) + 1  # After means from year+1
            return year_info
        
        return None if not year_info else year_info


# Module-level function for convenience
_parser_instance = None

def get_parser(data_path: str = None) -> QueryParser:
    """
    Get or create a singleton QueryParser instance.
    
    Args:
        data_path: Path to the parquet file
        
    Returns:
        QueryParser instance
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = QueryParser(data_path)
    return _parser_instance


def parse_query(query: str, data_path: str = None) -> Dict:
    """
    Convenience function to parse a query.
    
    Args:
        query: The query string to parse
        data_path: Path to the parquet file (only used for first call)
        
    Returns:
        Parsed query dictionary
    """
    parser = get_parser(data_path)
    return parser.parse_query(query)


if __name__ == "__main__":
    # Test the parser with sample queries
    print("=" * 80)
    print("Testing Smart Query Parser")
    print("=" * 80)
    
    test_queries = [
        "Tom Hanks comedy",
        "action movie",
        "Christopher Nolan thriller",
        "Tom Cruise action",
        "horror movies from 1990s",
        "comedy with Jim Carrey before 2000",
        "drama directed by Martin Scorsese"
    ]
    
    parser = QueryParser()
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        result = parser.parse_query(query)
        for key, value in result.items():
            if value:
                print(f"  {key}: {value}")
        print()
