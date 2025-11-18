import numpy as np
import pandas as pd
from nrclex import NRCLex

EMOTIONS = [
    "combined_anger", "combined_fear", "combined_sadness", "combined_joy",
    "combined_disgust", "combined_surprise", "combined_trust", "combined_anticipation"
]

def load_movie_emotion_vectors(path="data/processed/movie_emotion_sentiment.csv"):
    """Reads the movie_emotion_sentiment.csv file."""
    df = pd.read_csv(path).set_index("movie_name")

    movie_vectors = {
        movie: df.loc[movie, EMOTIONS].to_numpy(dtype=float)
        for movie in df.index
    }

    return movie_vectors

def compute_query_emotion_vector(query: str):
    raw = NRCLex(query.lower()).raw_emotion_scores

    vec = np.array(
        [raw.get(e.replace("combined_", ""), 0) for e in EMOTIONS],
        dtype=float
    )

    s = np.sum(vec)
    if s == 0:
        return np.zeros(len(EMOTIONS))

    return vec / (np.linalg.norm(vec) + 1e-6)