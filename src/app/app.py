import streamlit as st
import pandas as pd
import time
import sys
import os

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.search.engine import MovieSearchEngine

# --- IMPORT YOUR CHATBOT HERE ---
try:
    from src.search.chatbot import IMDBChatBot
except ImportError:
    st.warning("⚠️ Could not import IMDBChatBot. Make sure src/search/chatbot.py exists.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie Mind 🎬",
    page_icon="🍿",
    layout="centered"
)

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    /* Card Container */
    .stContainer {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
   
    /* Tags */
    .match-tag {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
        border: 1px solid #d2e3fc;
    }

    /* Metadata Labels */
    .meta-label {
        font-weight: 600;
        color: #555;
        font-size: 0.9rem;
    }
    .meta-value {
        color: #333;
        font-size: 0.9rem;
    }
   
    /* Chat Message Styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE ENGINE & CHATBOT (CACHED) ---
@st.cache_resource
def load_engine():
    return MovieSearchEngine()

@st.cache_resource
def load_chatbot(_engine):
    # --- FIX: Dane są wewnątrz hybrid_searcher ---
    if hasattr(_engine, 'hybrid_searcher') and hasattr(_engine.hybrid_searcher, 'df'):
        dataframe = _engine.hybrid_searcher.df
    elif hasattr(_engine, 'df'):
        dataframe = _engine.df
    else:
        st.error("❌ Error: Could not find dataframe in Engine (checked engine.hybrid_searcher.df).")
        return None
       
    return IMDBChatBot(_engine, dataframe)

try:
    if 'engine' not in st.session_state:
        with st.spinner('🚀 Booting up the engine... (Takes ~30s)'):
            st.session_state.engine = load_engine()
           
    # Initialize Chatbot only once engine is ready
    if 'chatbot' not in st.session_state and st.session_state.engine:
        st.session_state.chatbot = load_chatbot(st.session_state.engine)
       
    engine = st.session_state.engine
    chatbot = st.session_state.get('chatbot')

except Exception as e:
    st.error(f"Failed to load engine: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
   
    # --- MODE SELECTION ---
    mode = st.radio("Select Mode:", ["🔎 Search Engine", "🤖 AI Assistant"], index=0)
    st.markdown("---")
   
    if mode == "🔎 Search Engine":
        num_results = st.slider("Number of results", 5, 50, 10)
        st.markdown("### 💡 Search Tips")
        st.info("**Natural Language:**\n'90s comedy movies'\n'Sci-fi directed by Nolan'")
    else:
        st.markdown("### 🤖 Chat Tips")
        st.info("Ask specifically about movies:\n'What is the rating of The Matrix?'\n'Who directed Inception?'\n'Cast of Titanic'")
        if st.button("Clear Chat History"):
            st.session_state.messages = []

# --- MAIN PAGE LOGIC ---

if mode == "🔎 Search Engine":
    # --- EXISTING FRIEND'S CODE ---
    st.title("🎬 Movie Mind Search")
    st.caption("Semantic Search Engine (Sprint 3)")

    query = st.text_input("", placeholder="🔍 Search for movies, people, genres, or years...", label_visibility="collapsed")

    if query:
        start_time = time.time()
       
        # Run Search
        try:
            results = engine.search(query, k=num_results)
            duration = time.time() - start_time
           
            st.markdown(f"Found **{len(results)}** movies in `{duration:.3f}s`")
            st.markdown("---")
           
            if results.empty:
                st.warning("No movies found. Try a different query.")
            else:
                # Display Results
                for index, row in results.iterrows():
                    with st.container(border=True):
                        cols = st.columns([5, 2])
                       
                        # --- LEFT COLUMN: Main Info ---
                        with cols[0]:
                            title = row.get('title', 'Unknown Title')
                            year = int(row['startYear']) if pd.notna(row['startYear']) else "N/A"
                           
                            st.markdown(f"### {title} <span style='color:gray; font-size:0.8em;'>({year})</span>", unsafe_allow_html=True)
                           
                            # Genres
                            genres = row.get('genres', 'N/A')
                            if pd.notna(genres) and genres:
                                st.markdown(f"<span class='meta-label'>🎭 Genres:</span> <span class='meta-value'>{genres.replace(',', ', ')}</span>", unsafe_allow_html=True)
                               
                            # Directors
                            directors = row.get('director_name', '')
                            if pd.notna(directors) and directors:
                                st.markdown(f"<span class='meta-label'>🎬 Director:</span> <span class='meta-value'>{directors.replace(',', ', ')}</span>", unsafe_allow_html=True)
                               
                            # Actors
                            actors = row.get('actor_name', '')
                            if pd.notna(actors) and actors:
                                actor_list = actors.split(',')
                                display_actors = ', '.join(actor_list[:5])
                                if len(actor_list) > 5: display_actors += "..."
                                st.markdown(f"<span class='meta-label'>👤 Cast:</span> <span class='meta-value'>{display_actors}</span>", unsafe_allow_html=True)
                           
                            # Match Tags
                            st.write("")
                            tags = []
                            if row.get('actor_match_score', 0) > 0: tags.append("👤 Actor Match")
                            if row.get('director_match_score', 0) > 0: tags.append("🎬 Director Match")
                            if row.get('genre_match_score', 0) > 0: tags.append("🎭 Genre Match")
                            if row.get('year_match_score', 0) > 0: tags.append("📅 Year Match")
                           
                            if tags:
                                st.markdown(" ".join([f"<span class='match-tag'>{t}</span>" for t in tags]), unsafe_allow_html=True)

                        # --- RIGHT COLUMN: Stats ---
                        with cols[1]:
                            imdb_rating = row.get('averageRating', 0)
                            votes = row.get('numVotes', 0)
                           
                            rating_display = f"⭐ {imdb_rating:.1f}" if imdb_rating > 0 else "⭐ N/A"
                            votes_display = f"🗳️ {int(votes):,}" if votes > 0 else "🗳️ N/A"
                           
                            stat_cols = st.columns(2)
                            with stat_cols[0]:
                                st.markdown(f"**{rating_display}**")
                                st.caption("Rating")
                            with stat_cols[1]:
                                st.markdown(f"**{votes_display}**")
                                st.caption("Votes")
                               
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    # --- NEW AI CHATBOT CODE ---
    st.title("🤖 Movie Assistant")
    st.caption("Ask specific questions about movies (Rating, Cast, Plot)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your movie expert. Ask me about any movie rating, director, or cast!"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me about a movie..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if chatbot:
                    response = chatbot.get_response(prompt)
                else:
                    response = "⚠️ Chatbot is not initialized properly."
               
                st.markdown(response)
       
        st.session_state.messages.append({"role": "assistant", "content": response})

#to run: conda activate movie-mind
#  streamlit run src/app/app.py