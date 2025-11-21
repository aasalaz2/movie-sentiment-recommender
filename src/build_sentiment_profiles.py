import pandas as pd
from nrclex import NRCLex
import nltk
from utils import normalize_movie_name
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


EMOTIONS = ["anger", "fear", "sadness", "joy", "disgust", "surprise", "trust", "anticipation"]

def compute_sentiment(text):
    """Computes an emotion vector for the given text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {emotion: 0 for emotion in EMOTIONS}
    
    emotions = NRCLex(text).raw_emotion_scores
    return {e: emotions.get(e, 0) for e in EMOTIONS}

# Load cleaned datasets
letterboxd = pd.read_csv("data/processed/letterboxd_reviews_clean.csv")
metacritic = pd.read_csv("data/processed/metacritic_reviews_clean.csv")

# Normalize movie names
letterboxd["movie_name"] = letterboxd["movie_name"].astype(str).apply(normalize_movie_name)
metacritic["movie_name"] = metacritic["movie_name"].astype(str).apply(normalize_movie_name)

# Compute emotion dictionary
letterboxd["emotion_dict"] = letterboxd["review"].apply(compute_sentiment)
metacritic["emotion_dict"] = metacritic["summary"].apply(compute_sentiment)

# Expand dictionary to columns
letterboxd_emotions_df = letterboxd["emotion_dict"].apply(pd.Series)
letterboxd_emotions_df["movie_name"] = letterboxd["movie_name"]

metacritic_emotions_df = metacritic["emotion_dict"].apply(pd.Series)
metacritic_emotions_df["movie_name"] = metacritic["movie_name"]

# Aggregate emotions per movie (mean)
letterboxd_group = letterboxd_emotions_df.groupby("movie_name")[EMOTIONS].mean().reset_index()
metacritic_group = metacritic_emotions_df.groupby("movie_name")[EMOTIONS].mean().reset_index()

letterboxd_group.columns = ["movie_name"] + [f"review_{e}" for e in EMOTIONS]
metacritic_group.columns = ["movie_name"] + [f"summary_{e}" for e in EMOTIONS]

# Merge
combined = pd.merge(letterboxd_group, metacritic_group, on="movie_name", how="outer")

# Fill missing values
for e in EMOTIONS:
    combined[f"review_{e}"] = combined[f"review_{e}"].fillna(combined[f"summary_{e}"])
    combined[f"summary_{e}"] = combined[f"summary_{e}"].fillna(combined[f"review_{e}"])

# Combine emotion scores per emotion
for e in EMOTIONS:
    combined[f"combined_{e}"] = (
        0.6 * combined[f"review_{e}"] +
        0.4 * combined[f"summary_{e}"]
    )

# Save output
combined.to_csv(
    "data/processed/movie_emotion_sentiment.csv",
    index=False
)