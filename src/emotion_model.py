import numpy as np
import pandas as pd
from nrclex import NRCLex
from utils import normalize_movie_name

EMOTIONS = [
    "combined_anger", "combined_fear", "combined_sadness", "combined_joy",
    "combined_disgust", "combined_surprise", "combined_trust", "combined_anticipation"
]

def load_movie_emotion_vectors(path="data/processed/movie_emotion_sentiment.csv"):
    """Reads the movie_emotion_sentiment.csv file."""
    df = pd.read_csv(path)

    movie_vectors = {}

    for _, row in df.iterrows():
        movie = normalize_movie_name(row["movie_name"])
        vec = np.array([
            row["combined_anger"],
            row["combined_fear"],
            row["combined_sadness"],
            row["combined_joy"],
            row["combined_disgust"],
            row["combined_surprise"],
            row["combined_trust"],
            row["combined_anticipation"]
        ], dtype=float)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.zeros(len(EMOTIONS), dtype=float)

        movie_vectors[movie] = vec

    return movie_vectors

def compute_query_emotion_vector(query):
    raw = NRCLex(query.lower()).raw_emotion_scores

    vec = np.array(
        [raw.get(e.replace("combined_", ""), 0) for e in EMOTIONS],
        dtype=float
    )

    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm

    return vec