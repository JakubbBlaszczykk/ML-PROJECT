"""
Configuration loader for search engine

Loads settings from search_config.ini
"""

import configparser
import os

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class SearchConfig:
    """Centralized configuration for the search engine."""
    
    def __init__(self, config_path=None):
        """
        Load configuration from INI file.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "src/config/search_config.ini")
            
        self.config = configparser.ConfigParser()
        
        # Load config file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config.read(config_path)
        
        # Parse and store settings
        self._load_weights()
        self._load_query_parser_settings()
        self._load_hybrid_search_settings()
        self._load_title_matching_settings()
        self._load_year_matching_settings()
        self._load_multi_role_settings()
        self._load_result_settings()
        self._load_metadata_settings()
        self._load_fuzzy_matching_settings()
    
    def _load_weights(self):
        """Load ranking formula weights."""
        weights_section = self.config['weights']
        self.weights = {
            'W_hybrid': weights_section.getfloat('W_hybrid'),
            'W_popularity': weights_section.getfloat('W_popularity'),
            'W_rating': weights_section.getfloat('W_rating'),
            'W_actor': weights_section.getfloat('W_actor'),
            'W_director': weights_section.getfloat('W_director'),
            'W_writer': weights_section.getfloat('W_writer'),
            'W_genre': weights_section.getfloat('W_genre'),
            'W_year': weights_section.getfloat('W_year'),
        }
    
    def _load_query_parser_settings(self):
        """Load query parser settings."""
        parser_section = self.config['query_parser']
        self.query_parser = {
            'min_name_words': parser_section.getint('min_name_words'),
            'min_single_word_length': parser_section.getint('min_single_word_length'),
            'strict_token_matching': parser_section.getboolean('strict_token_matching'),
        }
    
    def _load_hybrid_search_settings(self):
        """Load hybrid search settings."""
        search_section = self.config['hybrid_search']
        self.hybrid_search = {
            'candidate_multiplier': search_section.getint('candidate_multiplier'),
            'min_candidates': search_section.getint('min_candidates'),
            'rrf_k': search_section.getint('rrf_k'),
            'rank_decay_factor': search_section.getfloat('rank_decay_factor', fallback=0.2),
        }
    
    def _load_title_matching_settings(self):
        """Load title matching settings."""
        title_section = self.config['title_matching']
        
        # Parse generic movie terms (comma-separated string to set)
        generic_terms_str = title_section.get('generic_movie_terms', '')
        generic_terms = set(term.strip().lower() for term in generic_terms_str.split(',') if term.strip())
        
        self.title_matching = {
            'jaccard_threshold': title_section.getfloat('jaccard_threshold'),
            'exact_title_boost': title_section.getfloat('exact_title_boost'),
            'partial_title_boost': title_section.getfloat('partial_title_boost'),
            'prefix_title_boost': title_section.getfloat('prefix_title_boost'),
            'generic_movie_terms': generic_terms,
        }
    
    def _load_year_matching_settings(self):
        """Load year matching settings."""
        year_section = self.config['year_matching']
        self.year_matching = {
            'year_tolerance': year_section.getint('year_tolerance'),
            'exact_match_score': year_section.getfloat('exact_match_score'),
            'near_match_score': year_section.getfloat('near_match_score'),
            'no_match_score': year_section.getfloat('no_match_score'),
        }
    
    def _load_multi_role_settings(self):
        """Load multi-role matching settings."""
        role_section = self.config['multi_role']
        self.multi_role = {
            'full_match_score': role_section.getfloat('full_match_score'),
            'partial_match_score': role_section.getfloat('partial_match_score'),
        }
    
    def _load_result_settings(self):
        """Load result settings."""
        results_section = self.config['results']
        self.results = {
            'default_k': results_section.getint('default_k'),
            'max_k': results_section.getint('max_k'),
        }
    
    def _load_metadata_settings(self):
        """Load metadata settings."""
        metadata_section = self.config['metadata']
        self.metadata = {
            'preferred_title_field': metadata_section.get('preferred_title_field'),
            'fallback_title_field': metadata_section.get('fallback_title_field'),
        }
    
    def _load_fuzzy_matching_settings(self):
        """Load fuzzy matching settings."""
        fuzzy_section = self.config['fuzzy_matching']
        self.fuzzy_matching = {
            'enabled': fuzzy_section.getboolean('enabled'),
            'people_min_similarity': fuzzy_section.getfloat('people_min_similarity'),
            'people_min_chars': fuzzy_section.getint('people_min_chars'),
            'people_max_candidates': fuzzy_section.getint('people_max_candidates'),
            'genre_min_similarity': fuzzy_section.getfloat('genre_min_similarity'),
            'genre_min_chars': fuzzy_section.getint('genre_min_chars'),
            'title_min_similarity': fuzzy_section.getfloat('title_min_similarity'),
            'title_min_chars': fuzzy_section.getint('title_min_chars'),
            'return_all_above_threshold': fuzzy_section.getboolean('return_all_above_threshold'),
            'use_popularity_weighting': fuzzy_section.getboolean('use_popularity_weighting'),
            'popularity_weight': fuzzy_section.getfloat('popularity_weight'),
            # Title fuzzy matching
            'title_fuzzy_enabled': fuzzy_section.getboolean('title_fuzzy_enabled'),
            'title_fuzzy_min_similarity': fuzzy_section.getfloat('title_fuzzy_min_similarity'),
            'title_fuzzy_min_chars': fuzzy_section.getint('title_fuzzy_min_chars'),
            'title_popularity_limit': fuzzy_section.getint('title_popularity_limit'),
        }


# Global config instance
_config = None

def get_config(config_path=None):
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SearchConfig instance
    """
    global _config
    if _config is None:
        _config = SearchConfig(config_path)
    return _config


def reload_config(config_path=None):
    """
    Reload configuration from file.
    
    Useful for picking up changes without restarting.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SearchConfig instance
    """
    global _config
    _config = SearchConfig(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Loaded configuration:")
    print(f"\nWeights: {config.weights}")
    print(f"\nQuery Parser: {config.query_parser}")
    print(f"\nHybrid Search: {config.hybrid_search}")
    print(f"\nYear Matching: {config.year_matching}")
    print(f"\nMulti-Role: {config.multi_role}")
    print(f"\nResults: {config.results}")
    print(f"\nMetadata: {config.metadata}")
