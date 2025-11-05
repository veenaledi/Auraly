import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Auraly - Mood-Based Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #b3b3b3;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Load the song dataset
        songs_df = pd.read_csv('spotify_mood_dataset.csv')
        
        # Load phrase dataset
        phrases_df = pd.read_csv('phrases.csv')
        
        # Load TF-IDF components
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        with open('tfidf_phrase_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        # Load TF-IDF phrases lookup
        tfidf_phrases = pd.read_csv('tfidf_phrases_lookup.csv')
        
        return songs_df, phrases_df, tfidf_vectorizer, tfidf_matrix, tfidf_phrases
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def get_mood_from_phrase(user_input, tfidf_vectorizer, tfidf_matrix, tfidf_phrases):
    """Convert user phrase to mood using enhanced keyword matching + TF-IDF similarity"""
    try:
        import re

        # Clean and lowercase user input
        user_input_clean = re.sub(r'[^\w\s]', '', user_input.lower().strip())

        # Enhanced keyword mapping for better accuracy
        mood_keywords = {
            'energetic': [
                'gym', 'workout', 'exercise', 'running', 'energy', 'energetic', 'pump',
                'intense', 'powerful', 'strong', 'adrenaline', 'hype', 'beast mode',
                'cardio', 'hiit', 'training', 'motivated', 'power', 'explosive'
            ],
            'happy': [
                'happy', 'joyful', 'cheerful', 'upbeat', 'positive', 'bright',
                'sunny', 'fun', 'party', 'celebrate', 'excited', 'good vibes',
                'mood boost', 'smile', 'dance', 'feel good'
            ],
            'calm': [
                'calm', 'relax', 'chill', 'peaceful', 'tranquil', 'soothing',
                'meditate', 'zen', 'quiet', 'serene', 'mellow', 'soft',
                'focus', 'study', 'concentrate', 'ambient', 'unwind', 'destress'
            ],
            'sad': [
                'sad', 'melancholy', 'melancholic', 'depressed', 'down', 'blue', 'heartbreak',
                'crying', 'tears', 'lonely', 'miss', 'grief', 'somber',
                'emotional', 'hurt', 'pain', 'breakup', 'reflection'
            ]
        }

        # Keyword-based scoring
        mood_scores = {
            mood: sum(1 for keyword in keywords if keyword in user_input_clean)
            for mood, keywords in mood_keywords.items()
        }
        mood_scores = {m: s for m, s in mood_scores.items() if s > 0}

        if mood_scores:
            best_mood = max(mood_scores, key=mood_scores.get)
            confidence = min(mood_scores[best_mood] * 0.3, 1.0)
            return best_mood, confidence

        # Fallback to TF-IDF similarity
        user_vector = tfidf_vectorizer.transform([user_input_clean])
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        best_match_idx = similarities.argmax()
        confidence = similarities[best_match_idx]

        # Use 'mood_label' column explicitly
        if 'mood_label' in tfidf_phrases.columns:
            mood = tfidf_phrases.iloc[best_match_idx]['mood_label']
        else:
            st.error("Column 'mood_label' not found in tfidf_phrases_lookup.csv")
            return None, confidence

        if confidence < 0.1:
            return None, confidence

        return mood, confidence

    except Exception as e:
        st.error(f"Error processing phrase: {str(e)}")
        return None, 0


def generate_playlist(songs_df, mood, num_songs=15):
    """Generate playlist based on mood"""
    try:
        # Filter songs by mood
        mood_songs = songs_df[songs_df['mood_label'].str.lower() == mood.lower()].copy()
        
        # Remove Duplicates
        mood_songs = mood_songs.drop_duplicates(subset=['track', 'artist'])

        if mood_songs.empty:
            return None
        
        # Mood-specific scoring weights
        if mood.lower() == 'happy':
            mood_songs['score'] = (
                mood_songs['valence'] * 0.4 + 
                mood_songs['energy'] * 0.3 + 
                mood_songs['danceability'] * 0.2 +
                mood_songs['tempo'] * 0.1
            )
        elif mood.lower() == 'sad':
            mood_songs['score'] = (
                (1 - mood_songs['valence']) * 0.4 + 
                mood_songs['acousticness'] * 0.3 + 
                (1 - mood_songs['energy']) * 0.2 +
                mood_songs['instrumentalness'] * 0.1
            )
        elif mood.lower() == 'energetic':
            mood_songs['score'] = (
                mood_songs['energy'] * 0.4 + 
                mood_songs['tempo'] * 0.3 + 
                mood_songs['danceability'] * 0.2 +
                mood_songs['loudness'] * 0.1
            )
        elif mood.lower() == 'calm':
            mood_songs['score'] = (
                (1 - mood_songs['energy']) * 0.4 + 
                mood_songs['acousticness'] * 0.3 + 
                (1 - mood_songs['tempo']) * 0.2 +
                mood_songs['instrumentalness'] * 0.1
            )
        else:
            mood_songs['score'] = mood_songs['valence']
        
        # Sort by score and get top songs
        top_songs = mood_songs.nlargest(num_songs, 'score')
        
        return top_songs
    except Exception as e:
        st.error(f"Error generating playlist: {str(e)}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Auraly</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Mood-Based Playlist Generator</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading music library...'):
        songs_df, phrases_df, tfidf_vectorizer, tfidf_matrix, tfidf_phrases = load_data()
    
    if songs_df is None:
        st.error("Failed to load data. Please check your files.")
        return
    
    st.success(f"Loaded {len(songs_df):,} songs across 4 moods")
    
    # Create two columns for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "How are you feeling?",
            placeholder="e.g., 'need calm focus music', 'upbeat gym vibes', or just 'happy'",
            help="Describe your mood or simply enter a mood keyword (happy, sad, energetic, calm)"
        )
    
    with col2:
        num_songs = st.slider("Playlist size", 5, 30, 15)
    
    # Generate playlist button
    if st.button("Generate Playlist", type="primary", use_container_width=True):
        if user_input:
            with st.spinner('Creating your perfect playlist...'):
                # Get mood from phrase
                mood, confidence = get_mood_from_phrase(
                    user_input, 
                    tfidf_vectorizer, 
                    tfidf_matrix, 
                    tfidf_phrases
                )
                
                if mood:
                    st.info(f"Detected Mood: **{mood.title()}** (Confidence: {confidence:.2%})")
                    
                    # Generate playlist
                    playlist = generate_playlist(songs_df, mood, num_songs)
                    
                    if playlist is not None and len(playlist) > 0:
                        st.success(f"Generated {len(playlist)} songs for you!")
                        
                        # Display playlist
                        st.markdown("### Your Playlist")
                        
                        for idx, row in enumerate(playlist.itertuples(), 1):
                            col_a, col_b, col_c = st.columns([0.5, 3, 1])
                            
                            with col_a:
                                st.markdown(f"**{idx}.**")
                            
                            with col_b:
                                st.markdown(f"**{row.track}**")
                                st.caption(f"by {row.artist}")
                            
                            with col_c:
                                if hasattr(row, 'spotify_uri') and pd.notna(row.spotify_uri):
                                    spotify_url = f"https://open.spotify.com/track/{row.spotify_uri.split(':')[-1]}"
                                    st.markdown(f"[▶️ Play]({spotify_url})")
                            
                            st.divider()
                        
                        # Mood statistics
                        st.markdown("### Playlist Characteristics")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.metric("Avg Energy", f"{playlist['energy'].mean():.2f}")
                        with stat_col2:
                            st.metric("Avg Valence", f"{playlist['valence'].mean():.2f}")
                        with stat_col3:
                            st.metric("Avg Tempo", f"{playlist['tempo'].mean():.0f} BPM")
                        with stat_col4:
                            st.metric("Avg Danceability", f"{playlist['danceability'].mean():.2f}")
                    else:
                        st.warning("No songs found for this mood. Try a different phrase!")
                else:
                    st.error("Could not detect mood. Please try again with a different phrase.")
        else:
            st.warning("Please enter a mood or phrase!")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## About Auraly")
        st.markdown("""
        Auraly uses machine learning to understand your mood and recommend the perfect playlist.
        
        **How it works:**
        1. Enter how you're feeling
        2. AI analyzes your phrase
        3. Get a curated playlist matching your mood
        
        **Supported Moods:**
        - Happy
        - Sad
        - Energetic
        - Calm
        """)
        
        st.markdown("---")
        st.markdown("### Example Phrases")
        examples = [
            "need calm focus music",
            "upbeat gym vibes",
            "feeling melancholic",
            "party mood",
            "relaxing evening"
        ]
        for example in examples:
            if st.button(f"{example}", key=example):
                st.session_state.example_input = example
        
        st.markdown("---")
        st.caption(f"🎵 {len(songs_df):,} songs in library")

if __name__ == "__main__":
    main()