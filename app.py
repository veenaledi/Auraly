
import streamlit as st
import pandas as pd
import pickle
import json
import os
from sklearn.metrics.pairwise import cosine_similarity


# 1. CONFIG & PATHS

BASE_DIR = os.path.dirname(__file__)
st.set_page_config(
    page_title="Auraly - Mood to Music",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. LOAD ASSETS (cached)

@st.cache_resource
def load_resources():
    """Load models, matrices, and datasets from disk."""
    try:
        tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb"))
        X_phr = pickle.load(open(os.path.join(BASE_DIR, "tfidf_phrase_matrix.pkl"), "rb"))
        phrases_df = pd.read_csv(os.path.join(BASE_DIR, "tfidf_phrases_lookup.csv"))
        songs_df = pd.read_csv(os.path.join(BASE_DIR, "spotify_mood_dataset.csv"))
        label_map = json.load(open(os.path.join(BASE_DIR, "label_map.json")))
        return tfidf, X_phr, phrases_df, songs_df, label_map
    except Exception as e:
        st.error(f"❌ Failed to load model/data: {e}")
        st.stop()

tfidf, X_phr, phrases_df, songs_df, label_map = load_resources()
ALL_MOODS = sorted(set(label_map.values()))  # Dynamic from data


# 3. CORE LOGIC

def top2_moods_from_phrase(phrase: str):
    """Return the top two most similar moods from a given phrase."""
    x = tfidf.transform([phrase.lower().strip()])
    sims = cosine_similarity(x, X_phr)[0]
    top_idx = sims.argsort()[::-1][:2]
    return [(phrases_df.iloc[i]["mood_label"], float(sims[i])) for i in top_idx]

def playlist_for_mood(mood_label: str, top_k: int = 10):
    """Generate a playlist for a detected mood."""
    sub = songs_df[songs_df["mood_label"].str.lower() == mood_label.lower()]
    if sub.empty:
        return pd.DataFrame()

    sort_cols = [c for c in ["valence", "energy", "danceability"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    else:
        sub = sub.sample(frac=1, random_state=None)  # fallback shuffle

    return sub.head(min(top_k, len(sub))).copy()

def playlist_from_phrase(phrase: str, top_k=10, ambiguity_margin=0.05, min_similarity=0.01):
    """Main logic — detect mood and fetch playlist."""
    if not phrase.strip():
        return {"ambiguous": True, "playlist": None}

    top2 = top2_moods_from_phrase(phrase)
    if not top2 or top2[0][1] < min_similarity:
        return {"ambiguous": True, "playlist": None}

    if len(top2) > 1 and (top2[0][1] - top2[1][1]) <= ambiguity_margin:
        return {"ambiguous": True, "playlist": None, "top2": top2}

    mood, score = top2[0]
    pl = playlist_for_mood(mood, top_k=top_k)
    return {"ambiguous": False, "mood": mood, "score": score, "playlist": pl}


# 4. USER INTERFACE

st.title("🎧 Auraly")
st.markdown("### *Turn your mood into a playlist*")

phrase = st.text_input(
    "How are you feeling?",
    placeholder="e.g., 'rainy afternoon', 'workout grind', 'cozy night'",
    help="We'll match your words to a mood and build a Spotify-ready playlist."
)

if phrase:
    with st.spinner("🎵 Detecting mood..."):
        result = playlist_from_phrase(phrase, top_k=10)

    if result["ambiguous"]:
        st.warning("Hmm, not sure about that vibe. Try something like:")
        st.code(", ".join(sorted(ALL_MOODS)), language=None)

        # Show top2 moods if available
        if "top2" in result:
            moods = [m[0].title() for m in result["top2"]]
            st.info(f"Your vibe seems between **{moods[0]}** and **{moods[1]}** — try a clearer phrase?")

    else:
        mood = result["mood"].title()
        st.success(f"**{mood}** mood detected! (confidence: {result['score']:.2f})")

        playlist = result["playlist"]
        if playlist.empty:
            st.info(f"No songs found for **{mood}** — try another mood!")
        else:
            st.markdown(f"### 🎶 Top {len(playlist)} Songs for **{mood}**")
            for _, row in playlist.iterrows():
                artist = row.get("artist", "Unknown")
                track = row.get("track", "Unknown Track")
                url = row.get("url_spotify") or row.get("uri", "")
                if pd.notna(url) and url.startswith("http"):
                    st.markdown(f"• **[{track} – {artist}]({url})**", unsafe_allow_html=True)
                else:
                    st.markdown(f"• {track} – {artist}")

else:
    st.info("Type a phrase above to generate your mood playlist!")


# 5. FOOTER

st.markdown("---")
st.caption("Auraly • Built using TF-IDF, XGBoost, and Spotify data")
