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


class QueryParser:
    """
    Intelligently parses search queries to extract filters and search terms.
    
    The parser creates lookup sets from movie data and matches query terms against
    known entities (actors, directors, writers, genres). It maintains partial retention
    of detected terms in the search query to handle cases like "action movie" which
    should match both genre:action and title containing "action".
    """
    
    def __init__(self, data_path: str = "data/imdb_us_movies_merged.parquet", force_rebuild: bool = False):
        """
        Initialize the query parser by creating or loading lookup sets.
        
        Args:
            data_path: Path to the parquet file containing movie data
            force_rebuild: If True, rebuild lookup sets even if cache exists
        """
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
                print(f"✓ Loaded {len(self.actors_set)} unique actors from cache")
                print(f"✓ Loaded {len(self.directors_set)} unique directors from cache")
                print(f"✓ Loaded {len(self.writers_set)} unique writers from cache")
                print(f"✓ Loaded {len(self.genres_set)} unique genres from cache")
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
            
            print(f"✓ Created {len(self.actors_set)} unique actors")
            print(f"✓ Created {len(self.directors_set)} unique directors")
            print(f"✓ Created {len(self.writers_set)} unique writers")
            print(f"✓ Created {len(self.genres_set)} unique genres")
            
            # Save to cache
            try:
                cache_data = {
                    'actors': self.actors_set,
                    'directors': self.directors_set,
                    'writers': self.writers_set,
                    'primary_roles': self.primary_roles,
                    'genres': self.genres_set
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"✓ Saved lookup sets to cache: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
        
        print("QueryParser ready!")
    
    def _extract_people_with_roles(self, df: pl.DataFrame) -> Dict:
        """
        Extract people names and count their appearances in each role.
        
        Args:
            df: DataFrame containing movie data
            
        Returns:
            Dictionary with:
                - 'actors': Set of actor names
                - 'directors': Set of director names
                - 'writers': Set of writer names
                - 'primary_roles': Dict mapping name -> primary role
        """
        from collections import defaultdict
        
        role_counts = defaultdict(lambda: {'actor': 0, 'director': 0, 'writer': 0})
        
        # Count appearances in each role
        for row in df.select(['cast', 'directors', 'writers']).to_dicts():
            # Process cast
            cast_list = row.get('cast')
            if cast_list:
                for person in cast_list:
                    if person and isinstance(person, dict):
                        name = person.get('primaryName')
                        if name:
                            role_counts[name.lower()]['actor'] += 1
            
            # Process directors
            directors_list = row.get('directors')
            if directors_list:
                for person in directors_list:
                    if person and isinstance(person, dict):
                        name = person.get('primaryName')
                        if name:
                            role_counts[name.lower()]['director'] += 1
            
            # Process writers
            writers_list = row.get('writers')
            if writers_list:
                for person in writers_list:
                    if person and isinstance(person, dict):
                        name = person.get('primaryName')
                        if name:
                            role_counts[name.lower()]['writer'] += 1
        
        # Determine primary roles and create sets
        actors_set = set()
        directors_set = set()
        writers_set = set()
        primary_roles = {}
        
        for name, counts in role_counts.items():
            # Add to all relevant sets
            if counts['actor'] > 0:
                actors_set.add(name)
            if counts['director'] > 0:
                directors_set.add(name)
            if counts['writer'] > 0:
                writers_set.add(name)
            
            # Determine primary role (most appearances)
            primary_role = max(counts.items(), key=lambda x: x[1])[0]
            primary_roles[name] = primary_role
        
        return {
            'actors': actors_set,
            'directors': directors_set,
            'writers': writers_set,
            'primary_roles': primary_roles
        }
    
    def _extract_genres(self, df: pl.DataFrame) -> Set[str]:
        """
        Extract unique genres from the genres column.
        
        Args:
            df: DataFrame containing movie data
            
        Returns:
            Set of unique genres (lowercase for matching)
        """
        genres = set()
        
        # Get unique genres using polars - much more efficient
        genres_list = df.select('genres').unique().to_series().to_list()
        
        for row in genres_list:
            if row is not None and isinstance(row, str):
                # Genres are comma-separated
                for genre in row.split(','):
                    genre = genre.strip()
                    if genre:
                        genres.add(genre.lower())
        
        return genres
    
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
    
    def _find_all_people_in_query(self, query_lower: str, found_tokens: Set[str]) -> List[str]:
        """
        Find all people names (actors, directors, writers) in the query string.
        
        Args:
            query_lower: Lowercase query string
            found_tokens: Set to track found tokens
            
        Returns:
            List of matched people names
        """
        matched_people = []
        
        # Combine all people into one set
        all_people = self.actors_set.union(self.directors_set).union(self.writers_set)
        
        # Filter names to avoid false positives using config settings
        min_words = self.config.query_parser['min_name_words']
        min_length = self.config.query_parser['min_single_word_length']
        
        valid_names = []
        for n in all_people:
            word_count = len(n.split())
            if word_count >= min_words:  # Multi-word names always valid
                valid_names.append(n)
            elif len(n) >= min_length:  # Single word but long enough
                valid_names.append(n)
        
        # Sort by length (longest first) to match multi-word names before single words
        sorted_names = sorted(valid_names, key=len, reverse=True)
        
        for name in sorted_names:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, query_lower):
                # Avoid overlapping matches
                name_words = set(name.split())
                if not name_words.intersection(found_tokens):
                    matched_people.append(name)
                    found_tokens.update(name_words)
        
        return matched_people
    
    def _find_genres_in_query(self, query_lower: str, found_tokens: Set[str]) -> List[str]:
        """
        Find genre keywords in the query string, including aliases.
        
        Args:
            query_lower: Lowercase query string
            found_tokens: Set to track found tokens
            
        Returns:
            List of matched genres (canonical names)
        """
        matched_genres = set()  # Use set to avoid duplicates
        
        # First, check genre aliases
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
        
        # Then check actual genre names
        for genre in self.genres_set:
            # Use word boundaries
            pattern = r'\b' + re.escape(genre) + r'\b'
            if re.search(pattern, query_lower):
                # Check if not already found as part of a person's name
                if genre not in found_tokens:
                    matched_genres.add(genre)
                    found_tokens.add(genre)
        
        return list(matched_genres)
    
    def _extract_years(self, query: str) -> Optional[Dict]:
        """
        Extract year constraints from the query.
        
        Patterns supported:
        - Single year: "2020", "1995"
        - Decade: "1990s", "2000s"
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
        
        # Decade (e.g., "1990s", "2000s")
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

def get_parser(data_path: str = "data/imdb_us_movies_merged.parquet") -> QueryParser:
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


def parse_query(query: str, data_path: str = "data/imdb_us_movies_merged.parquet") -> Dict:
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
