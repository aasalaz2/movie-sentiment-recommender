import json
import pandas as pd
from collections import defaultdict
from text_cleaning import clean_query


def load_indexes(letterboxd_path, metacritic_path):
    """Load the inverted index JSON file."""
    with open(letterboxd_path, "r", encoding="utf-8") as f:
        letterboxd_index = json.load(f)
    
    with open(metacritic_path, "r", encoding="utf-8") as f:
        metacritic_index = json.load(f)

    return letterboxd_index, metacritic_index

def load_movie_profiles():
    """Load sentiment and document movie profiles."""
    sentiment_df = pd.read_csv("data/processed/movie_sentiment_agg.csv")
    docs_df = pd.read_csv("data/processed/movie_docs.csv")

    sentiment_profile = sentiment_df.set_index("movie_name").to_dict(orient="index")
    docs_profile = docs_df.set_index("movie_name")["movie_doc"].to_dict()

    return sentiment_profile, docs_profile

def load_doc_lengths(path):
    """Load desired doc lengths index."""
    with open(path, "r", encoding="utf=8") as f:
        return json.load(f)

def rating_sentiment(movie, sentiment_profile):
    if movie not in sentiment_profile:
        return 0
    return sentiment_profile[movie]["pos_rate"]

def retrieve_candidates(tokens, inverted_index):
    """Retrieve candidate movies that contain one or more query tokens."""
    candidates = defaultdict(int)
    for token in tokens:
        if token in inverted_index:
            for entry in inverted_index[token]:
                movie = entry["movie_name"]
                freq = entry.get("count", 1)
                candidates[movie] += freq
    return candidates

def combine_scores(
        letterboxd_scores,
        metacritic_scores,
        sentiment_profile,
        weight_reviews=0.6,
        weight_summaries=0.3,
        weight_sentiment=0.1):
    """Combines letterboxd reviews, metacritic movie summaries scores, and their sentiment profiles."""
    final_scores = defaultdict(float)

    # Combine all movies
    all_movies = set(letterboxd_scores) | set(metacritic_scores) | set(sentiment_profile)

    for movie in all_movies:
        # Review score
        review_s = letterboxd_scores.get(movie, 0)

        # Summary score
        summary_s = metacritic_scores.get(movie, 0)

        # Rating-based sentiment score
        sentiment_s = sentiment_profile.get(movie, {}).get("pos_rate", 0)

        # Weighted combination
        score = (
            weight_reviews * review_s +
            weight_summaries * summary_s +
            weight_sentiment * sentiment_s
        )

        final_scores[movie] = score

    # Normalize
    if final_scores:
        max_score = max(final_scores.values())
        if max_score > 0:
            for m in final_scores:
                final_scores[m] /= max_score
    
    return final_scores

def normalize_scores(scores):
    if not scores:
        return {}
    max_score = max(scores.values())
    return {m: v / max_score for m, v in scores.items() if max_score > 0}

def process_query(query, letterboxd_path, metacritic_path):
    """Clean query, retrieve candidates, and rank."""
    letterboxd_index, metacritic_index = load_indexes(letterboxd_path, metacritic_path)
    sentiment_profile, doc_profile = load_movie_profiles()

    # Load doc lengths for leterboxd and metacritic to be used for BM25
    letterboxd_doc_lengths = load_doc_lengths("indexes/letterboxd_doc_lengths.json")
    metacritic_doc_lengths = load_doc_lengths("indexes/metacritic_doc_lengths.json")

    # Compute length averages, also to be used for BM25
    letterboxd_avg_len = sum(letterboxd_doc_lengths.values()) / len(letterboxd_doc_lengths)
    metacritic_avg_len = sum(metacritic_doc_lengths.values()) / len(metacritic_doc_lengths)
    
    tokens = clean_query(query)
    if not tokens:
        return []
    
    ####################################################################################
    # TODO: This whole area should be replaced with calls to our ranking algorithm (BM25?)

    # Retrieve raw term-frequency matches
    letterboxd_scores = retrieve_candidates(tokens, letterboxd_index)
    metacritic_scores = retrieve_candidates(tokens, metacritic_index)

    # Normalize scores
    letterboxd_scores = normalize_scores(letterboxd_scores)
    metacritic_scores = normalize_scores(metacritic_scores)

    # Combine scores
    combined_scores = combine_scores(letterboxd_scores, metacritic_scores, sentiment_profile)

    ranked = sorted(combined_scores.items(), key=lambda x:x[1], reverse=True)
    ####################################################################################


    return ranked

if __name__== "__main__":
    query = "dark somber revenge"
    letterboxd_path = "indexes/letterboxd_index.json"
    metacritic_path = "indexes/metacritic_index.json"

    results = process_query(query, letterboxd_path, metacritic_path)

    print(f"\nTop Results for Query: `{query}`\n")
    for movie, score in results[:10]:
        print(f"{movie}: {score}")