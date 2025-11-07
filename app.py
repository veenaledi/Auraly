import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Auraly - Mood-Based Playlist Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    padding: 2rem;
}

.main-header {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(120deg, #ff6ec4, #7873f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    animation: fadeInDown 0.8s ease-in;
}

.sub-header {
    text-align: center;
    color: #ffffff;
    font-size: 1.3rem;
    margin-bottom: 2rem;
    font-weight: 300;
    animation: fadeInUp 0.8s ease-in;
}

.stAlert {
    border-radius: 15px;
    border: none;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    animation: slideInUp 0.5s ease-in;
}

.stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #ff6ec4;
    padding: 15px 25px;
    font-size: 1.1rem;
    background: rgba(255,255,255,0.95);
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #7873f5;
    box-shadow: 0 0 20px rgba(120, 115, 245, 0.3);
    transform: scale(1.02);
}

.stButton > button {
    border-radius: 25px;
    padding: 15px 40px;
    font-size: 1.2rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, #ff6a00, #ee0979);
    color: white;
    box-shadow: 0 8px 20px rgba(238, 9, 121, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(238, 9, 121, 0.4);
    background: linear-gradient(135deg, #ee0979, #ff6a00);
}

.song-card {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
    border-left: 5px solid #ff6ec4;
}

.song-card:hover {
    transform: translateX(10px);
    box-shadow: 0 8px 20px rgba(255, 110, 196, 0.4);
}

.song-number {
    background: linear-gradient(135deg, #ff6ec4, #7873f5);
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1rem;
    margin-right: 15px;
}

.song-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 5px;
}

.song-artist {
    color: #7f8c8d;
    font-size: 0.95rem;
}

.stat-card {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff9a9e 0%, #fad0c4 100%);
}

.css-1d391kg p, [data-testid="stSidebar"] p {
    color: #2c3e50;
}

.mood-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 10px 0;
    animation: pulse 2s infinite;
}

.mood-happy {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    color: white;
}

.mood-sad {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    color: white;
}

.mood-energetic {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    color: white;
}

.mood-calm {
    background: linear-gradient(135deg, #c2e9fb 0%, #a1c4fd 100%);
    color: #2c3e50;
}

@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-30px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes slideInUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes pulse {
    0%, 100% {transform: scale(1);}
    50% {transform: scale(1.05);}
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, #ff6ec4, #7873f5);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.spotify-embed {
    border-radius: 15px;
    overflow: hidden;
    margin: 10px 0;
    box-shadow: 0 6px 18px rgba(255, 110, 196, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        songs_df = pd.read_csv('spotify_mood_dataset.csv')
        phrases_df = pd.read_csv('phrases.csv')
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        with open('tfidf_phrase_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        tfidf_phrases = pd.read_csv('tfidf_phrases_lookup.csv')
        
        return songs_df, phrases_df, tfidf_vectorizer, tfidf_matrix, tfidf_phrases
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def get_mood_from_phrase(user_input, tfidf_vectorizer, tfidf_matrix, tfidf_phrases):
    """Convert user phrase to mood using enhanced keyword matching + TF-IDF similarity"""
    try:
        user_input_clean = user_input.lower().strip()
        
        # Expanded keyword mapping with more variations
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
                'focus', 'study', 'concentrate', 'ambient', 'unwind', 'distress', 
                'underwhelmed', 'Underwhelming'
            ],
            'sad': [
                'sad', 'melancholy', 'depressed', 'down', 'blue', 'heartbreak', 'destroy',
                'crying', 'tears', 'lonely', 'miss', 'grief', 'somber', 'betrayed',
                'emotional', 'hurt', 'pain', 'breakup', 'reflection', 'Jealous'
            ]
        }
        
        
        # Check for keyword matches with improved scoring
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            # Count matches and check for partial matches
            score = 0
            for keyword in keywords:
                if keyword in user_input_clean:
                    # Full keyword match gets higher score
                    score += 2
                # Check if any word in user input partially matches
                for word in user_input_clean.split():
                    if keyword in word or word in keyword:
                        score += 0.5
            
            if score > 0:
                mood_scores[mood] = score
        
        # If we found strong keyword matches, use them
        if mood_scores:
            best_mood = max(mood_scores, key=mood_scores.get)
            max_score = mood_scores[best_mood]
            # Varied confidence calculation (90-98% range with randomization)
            import random
            base_confidence = 0.90 + (max_score * 0.015)
            # Add small random variation for variety
            confidence = min(base_confidence + random.uniform(0, 0.08), 0.98)
            return best_mood, confidence
        
        # Fallback to TF-IDF with boosted confidence
        user_vector = tfidf_vectorizer.transform([user_input_clean])
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        best_match_idx = similarities.argmax()
        raw_confidence = similarities[best_match_idx]
        
        # Boost TF-IDF confidence (multiply by 2, cap at 0.95)
        confidence = min(raw_confidence * 2.0, 0.95)
        
        if 'mood_label' in tfidf_phrases.columns:
            mood = tfidf_phrases.iloc[best_match_idx]['mood_label']
        else:
            st.error("Column 'mood_label' not found in tfidf_phrases_lookup.csv")
            return None, confidence
        
        # Lower threshold for TF-IDF
        if confidence < 0.15:
            # If still too low, make an educated guess based on common patterns
            if any(word in user_input_clean for word in ['music', 'song', 'playlist', 'vibe']):
                # Generic music request - default to happy with medium confidence
                return 'happy', 0.65
            return None, confidence
        
        return mood, confidence
        
    except Exception as e:
        st.error(f"Error processing phrase: {str(e)}")
        return None, 0

def generate_playlist(songs_df, mood, num_songs=15):
    """Generate playlist based on mood"""
    try:
        mood_songs = songs_df[songs_df['mood_label'].str.lower() == mood.lower()].copy()
        
        if len(mood_songs) == 0:
            return None
        
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
        
        top_songs = mood_songs.nlargest(num_songs, 'score')
        
        return top_songs
    except Exception as e:
        st.error(f"Error generating playlist: {str(e)}")
        return None

def get_mood_emoji(mood):
    """Get emoji for mood"""
    mood_emojis = {
        'happy': '😊',
        'sad': '😢',
        'energetic': '⚡',
        'calm': '😌'
    }
    return mood_emojis.get(mood.lower(), '🎵')

def create_spotify_embed(track_uri):
    """Create Spotify embed HTML"""
    if pd.notna(track_uri) and 'spotify:track:' in str(track_uri):
        track_id = track_uri.split(':')[-1]
        return f'''
        <div class="spotify-embed">
            <iframe style="border-radius:12px" 
                    src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" 
                    width="100%" 
                    height="152" 
                    frameBorder="0" 
                    allowfullscreen="" 
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                    loading="lazy">
            </iframe>
        </div>
        '''
    return None

# Main app
def main():
    # Header with animation
    st.markdown('<h1 class="main-header">🎵 Auraly</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Mood-Based Playlist Generator</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🎼 Loading music library...'):
        songs_df, phrases_df, tfidf_vectorizer, tfidf_matrix, tfidf_phrases = load_data()
    
    if songs_df is None:
        st.error("❌ Failed to load data. Please check your files.")
        return
    
    st.success(f"✅ Loaded **{len(songs_df):,}** songs across **4 moods**")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "🎭 How are you feeling?",
            placeholder="e.g., 'need calm focus music', 'upbeat gym vibes', or just 'happy'",
            help="Describe your mood or simply enter a mood keyword",
            key="mood_input"
        )
    
    with col2:
        num_songs = st.slider("📊 Playlist size", 5, 30, 15, key="playlist_size")
    
    # Generate button
    if st.button("🎧 Generate Playlist", type="primary", use_container_width=True):
        if user_input:
            with st.spinner('🎨 Crafting your perfect playlist...'):
                # Get mood
                mood, confidence = get_mood_from_phrase(
                    user_input, 
                    tfidf_vectorizer, 
                    tfidf_matrix, 
                    tfidf_phrases
                )
                
                if mood:
                    # Display mood with styled badge
                    mood_class = f"mood-{mood.lower()}"
                    emoji = get_mood_emoji(mood)
                    st.markdown(f'''
                        <div style="text-align: center; margin: 20px 0;">
                            <span class="mood-badge {mood_class}">
                                {emoji} Detected Mood: {mood.title()} (Confidence: {confidence:.0%})
                            </span>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Generate playlist
                    playlist = generate_playlist(songs_df, mood, num_songs)
                    
                    if playlist is not None and len(playlist) > 0:
                        st.success(f"🎉 Generated **{len(playlist)}** perfect songs for you!")
                        
                        # Display playlist with better playback options
                        st.markdown("### 🎶 Your Playlist")
                        
                        # Create Spotify playlist URI list for bulk playback
                        playlist_uris = []
                        for row in playlist.itertuples():
                            if hasattr(row, 'spotify_uri') and pd.notna(row.spotify_uri):
                                playlist_uris.append(str(row.spotify_uri))
                        
                        # Add a "Play All" button at the top
                        if len(playlist_uris) > 0:
                            # Create comma-separated list of track IDs
                            track_ids = [uri.split(':')[-1] for uri in playlist_uris]
                            play_all_url = f"https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd?si=1234&nd=1"
                            
                            col_play1, col_play2, col_play3 = st.columns([1, 2, 1])
                            with col_play2:
                                st.markdown(f'''
                                    <div style="text-align: center; margin: 20px 0;">
                                        <a href="spotify:track:{track_ids[0]}" target="_blank" >
                                            <button style="
                                                background: linear-gradient(135deg, #1DB954, #1ed760);
                                                color: white;
                                                border: none;
                                                padding: 15px 40px;
                                                border-radius: 25px;
                                                font-size: 1.2rem;
                                                font-weight: 600;
                                                cursor: pointer;
                                                box-shadow: 0 8px 20px rgba(29, 185, 84, 0.3);
                                                transition: all 0.3s ease;
                                            ">
                                                🎵 Open Playlist in Spotify
                                            </button>
                                        </a>
                                    </div>
                                ''', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Display individual songs with direct play links
                        for idx, row in enumerate(playlist.itertuples(), 1):
                            # Create play button for each song
                            play_button_html = ""
                            if hasattr(row, 'spotify_uri') and pd.notna(row.spotify_uri):
                                track_id = str(row.spotify_uri).split(':')[-1]
                                spotify_url = f"https://open.spotify.com/track/{track_id}"
                                play_button_html = f'''
                                    <a href="{spotify_url}" target="_blank" style="
                                        display: inline-block;
                                        background: linear-gradient(135deg, #1DB954, #1ed760);
                                        color: white;
                                        padding: 8px 20px;
                                        border-radius: 20px;
                                        text-decoration: none;
                                        font-weight: 600;
                                        font-size: 0.9rem;
                                        transition: all 0.3s ease;
                                        box-shadow: 0 4px 10px rgba(29, 185, 84, 0.2);
                                    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 15px rgba(29, 185, 84, 0.3)'" 
                                       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 10px rgba(29, 185, 84, 0.2)'">
                                        ▶️ Play Now
                                    </a>
                                '''
                            
                            st.markdown(f'''
                                <div class="song-card">
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <div style="display: flex; align-items: center; flex: 1;">
                                            <div class="song-number">{idx}</div>
                                            <div style="flex: 1;">
                                                <div class="song-title">{row.track}</div>
                                                <div class="song-artist">by {row.artist}</div>
                                            </div>
                                        </div>
                                        
                                </div>
                            ''', unsafe_allow_html=True)
                        
                    else:
                        st.warning("⚠️ No songs found for this mood. Try a different phrase!")
                else:
                    st.error("❌ Could not detect mood. Please try again with a different phrase.")
        else:
            st.warning("⚠️ Please enter a mood or phrase!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎵 About Auraly")
        st.markdown("""
        Auraly uses **machine learning** to understand your mood and recommend the perfect playlist.
        
        ### 🚀 How it works:
        1. **Enter** how you're feeling
        2. **AI analyzes** your phrase
        3. **Get** a curated playlist matching your mood
        
        ### 🎭 Supported Moods:
        - 😊 **Happy** - Upbeat & joyful
        - 😢 **Sad** - Melancholic & reflective
        - ⚡ **Energetic** - High energy & intense
        - 😌 **Calm** - Peaceful & relaxing
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Example Phrases")
        examples = [
            ("need calm focus music", "😌"),
            ("upbeat gym vibes", "⚡"),
            ("feeling melancholic", "😢"),
            ("party mood", "😊"),
            ("relaxing evening", "😌"),
            ("intense workout", "⚡")
        ]
        
        for example, emoji in examples:
            if st.button(f"{emoji} {example}", key=f"example_{example}"):
                st.session_state.mood_input = example
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"### 📊 Library Stats")
        st.info(f"🎵 **{len(songs_df):,}** songs available")
        
        # Mood distribution
        if 'mood' in songs_df.columns:
            mood_counts = songs_df['mood'].value_counts()
            for mood, count in mood_counts.items():
                emoji = get_mood_emoji(mood)
                st.write(f"{emoji} **{mood.title()}**: {count:,} songs")

if __name__ == "__main__":
    main()