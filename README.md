# IMDB Search Engine

A machine learning-based search engine for the IMDB dataset, combining keyword-based retrieval (BM25) with semantic search (SBERT) and intelligent query parsing to deliver relevant movie search results.

## Overview

This search engine provides a hybrid approach to movie search, leveraging both lexical and semantic similarity to handle diverse query types:

- **Exact title matching** for queries like "The Matrix" or "Inception"
- **Actor/director search** for "Tom Hanks movies" or "directed by Christopher Nolan"
- **Genre filtering** for "action movies" or "horror films"
- **Year filtering** for "movies from the 90s" or "films after 2020"
- **Fuzzy matching** for typos like "avtar" → "Avatar" or "tom hamks" → "Tom Hanks"
- **Complex queries** combining multiple filters like "90s action movies with Arnold Schwarzenegger"

## Features

- **Hybrid Search**: Combines BM25 keyword search with Sentence-BERT semantic embeddings using Reciprocal Rank Fusion (RRF).
- **Smart Query Parser**: Automatically extracts actors, directors, writers, genres, and year filters from natural language queries.
- **Fuzzy Matching**: Tolerates typos in movie titles, actor names, and genre terms using RapidFuzz.
- **Weighted Ranking**: Configurable scoring weights for relevance, popularity, rating, and filter matches.
- **Trie-based Name Lookup**: Efficient pattern matching for person names using a Trie data structure.

## Team Members

- Alicja Biernat (alicjabiernat)
- Adrianna Bartoszek (adriannabartoszek)
- Jakub Błaszczyk (JakubbBlaszczykk)
- Wojciech Jurewicz (wojciechjurewicz)
- Alan Makowski (alanmakowski16)
- Nikodem Kędzia (nikodemkedzia)
