class IMDBChatBot:
    def __init__(self, search_engine, data_df):
        self.search_engine = search_engine
        self.data_df = data_df 
        
        self.context = {
            'awaiting_confirmation': False,
            'candidates': [],       # List of found movies
            'current_idx': 0,       # Index of current movie candidate
            'detected_intent': None
        }

        self.intents = {
            'rating': ["What is the rating of", "What is the rating", "How many stars", "Is it good", "score", "IMDB rating", "rating"],
            'year': ["When was it released", "When came out", "What year", "release date", "How old is", "year"],
            'director': ["Who directed", "Who is the director of", "Who is the director", "filmmaker", "director", "who made"],
            'cast': ["Who acts in", "Who played in", "cast of", "actors in", "Who stars in", "actress", "cast", "actors"],
            'plot': ["What is the plot of", "What is the plot", "What is it about", "description", "summary", "plot"],
            'genre': ["What genre is", "What kind of movie is", "category", "genre"]
        }

    def _detect_intent(self, user_text):
        """
        Check if any phrase from self.intents exists in user text.
        """
        text = user_text.lower()
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                if phrase.lower() in text:
                    return intent
        return None

    def _clean_query(self, text, intent):
        """
        Removes the specific phrase that triggered the intent to leave just the title.
        """
        if not intent: return text
        
        clean_text = text.lower()
        # Sort phrases by length desc to remove specific long phrases first
        phrases = sorted(self.intents[intent], key=len, reverse=True)
        
        for phrase in phrases:
            if phrase.lower() in clean_text:
                clean_text = clean_text.replace(phrase.lower(), "")
                break # Remove only the best match
                
        return clean_text.replace("?", "").replace(".", "").strip()

    def _get_answer_from_row(self, row, intent):
        try:
            title = row.get('title', row.get('primaryTitle', 'Unknown Title'))
            
            if intent == 'rating':
                val = row.get('averageRating', 'N/A')
                votes = row.get('numVotes', 0)
                return f"The movie '{title}' has a rating of *{val}/10* (based on {votes} votes)."
            elif intent == 'year':
                val = row.get('startYear', 'N/A')
                return f"'{title}' was released in *{val}*."
            elif intent == 'director':
                val = row.get('director_name', 'Unknown')
                return f"Directed by: *{val}*."
            elif intent == 'cast':
                val = row.get('actor_name', 'Unknown')
                if isinstance(val, str) and len(val) > 100: val = val[:100] + "..."
                return f"Top cast: *{val}*."
            elif intent == 'genre':
                val = row.get('genres', 'Unknown')
                return f"Genres: *{val}*."
            elif intent == 'plot':
                val = row.get('overview', 'Plot summary not available.')
                return f"*Plot:* {val}"

            return f"I found '{title}', but I don't have information about {intent}."
        except Exception as e:
            return f"Error retrieving data: {str(e)}"

    def get_response(self, user_input):
        # 1. Handle confirmation loop (Yes/No)
        if self.context['awaiting_confirmation']:
            positives = ['yes', 'yeah', 'yep', 'correct', 'right', 'tak', 'sure']
            negatives = ['no', 'nah', 'nope', 'not', 'wrong', 'nie']
            
            user_input_lower = user_input.lower().strip()

            # A. User confirmed
            if any(t in user_input_lower for t in positives):
                current_movie = self.context['candidates'][self.context['current_idx']]
                answer = self._get_answer_from_row(current_movie, self.context['detected_intent'])
                
                self.context['awaiting_confirmation'] = False
                self.context['candidates'] = []
                return answer

            # B. User denied -> Try next candidate
            elif any(t in user_input_lower for t in negatives):
                self.context['current_idx'] += 1
                
                if self.context['current_idx'] < len(self.context['candidates']):
                    next_movie = self.context['candidates'][self.context['current_idx']]
                    title = next_movie.get('title', 'Unknown')
                    year = next_movie.get('startYear', 'N/A')
                    return f"Ah, okay. Did you mean *'{title}' ({year})*?"
                else:
                    self.context['awaiting_confirmation'] = False
                    return "I'm sorry, I couldn't find the movie you are looking for."
            
            else:
                return "Please answer 'yes' or 'no'."

        # 2. Main Logic
        intent = self._detect_intent(user_input)
        
        # Clean query using the dictionary phrases
        search_query = self._clean_query(user_input, intent)
        
        # Search for top 5 candidates
        results = self.search_engine.search(search_query, k=5)

        # Fallback if cleaning removed too much
        if results.empty:
            results = self.search_engine.search(user_input, k=5)
            if results.empty:
                return "I couldn't find any movie matching your request."

        candidates = results.to_dict('records')
        
        # Update context
        self.context['candidates'] = candidates
        self.context['current_idx'] = 0
        self.context['detected_intent'] = intent
        self.context['awaiting_confirmation'] = True

        first_movie = candidates[0]
        title = first_movie.get('title', 'Unknown')
        year = first_movie.get('startYear', 'N/A')

        if not intent:
             self.context['awaiting_confirmation'] = False
             return f"I found *'{title}' ({year})*. Ask me about its rating, director, or cast!"

        return f"Did you mean the movie *'{title}' ({year})*?"