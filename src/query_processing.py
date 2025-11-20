import json
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from utils import clean_query, normalize_movie_name
from emotion_model import load_movie_emotion_vectors, compute_query_emotion_vector
from semantic_model import compute_semantic_scores

EMOTIONS = [
    "combined_anger", "combined_fear", "combined_sadness", "combined_joy",
    "combined_disgust", "combined_surprise", "combined_trust", "combined_anticipation"
]

# Score weights
weight_reviews=0.35
weight_summaries=0.3
weight_rating_sentiment=0.1
weight_emotion_match=0.15
weight_semantic_scores=0.1

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

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def tfidf(tf, df, N):
    return (1 + math.log(tf)) * math.log((N+1) / (df+1))

def retrieve_candidates(tokens, inverted_index, total_docs):
    """Retrieve candidate movies that contain one or more query tokens."""
    candidates = defaultdict(int)
    for token in tokens:
        if token not in inverted_index:
            continue

        posting_list = inverted_index[token]
        df = len(posting_list)

        for entry in posting_list:
            movie = entry["movie_name"]
            movie = normalize_movie_name(movie)
            tf = entry.get("count", 1)
            
            candidates[movie] += tfidf(tf, df, total_docs)

    return candidates

def combine_scores(
        letterboxd_scores,
        metacritic_scores,
        rating_profile,
        emotion_profile,
        query_emotion_vec,
        semantic_scores,
        candidate_movies
    ):
    """Combines letterboxd reviews, metacritic movie summaries scores, and their sentiment profiles."""
    final_scores = defaultdict(float)

    # Combine all movies
    # all_movies = set(letterboxd_scores) | set(metacritic_scores) | set(emotion_profile) | set(semantic_scores)

    for movie in candidate_movies:
        movie = normalize_movie_name(movie)
        
        # Review score
        review_s = letterboxd_scores.get(movie, 0)

        # Summary score
        summary_s = metacritic_scores.get(movie, 0)

        # Rating-based sentiment score
        rating_s = rating_profile.get(movie, {}).get("pos_rate", 0)

        # Emotion similarity
        movie_vec = emotion_profile.get(movie, np.zeros(len(EMOTIONS)))
        emotion_match = cosine_similarity(movie_vec, query_emotion_vec)

        # Semantic score for movie
        semantic_s = semantic_scores.get(movie, 0.0)

        # Weighted combination
        score = (
            weight_reviews * review_s +
            weight_summaries * summary_s +
            weight_rating_sentiment * rating_s +
            weight_emotion_match * emotion_match +
            weight_semantic_scores * semantic_s
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

def process_query(query, letterboxd_path, metacritic_path, candidate_k=200):
    """Clean query, retrieve candidates, and rank."""
    letterboxd_index, metacritic_index = load_indexes(letterboxd_path, metacritic_path)
    rating_profile, doc_profile = load_movie_profiles()
    emotion_profile = load_movie_emotion_vectors()
    query_emotion_vec = compute_query_emotion_vector(query)

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
    letterboxd_scores = retrieve_candidates(tokens, letterboxd_index, len(letterboxd_doc_lengths))
    metacritic_scores = retrieve_candidates(tokens, metacritic_index, len(metacritic_doc_lengths))

    # Normalize scores
    # letterboxd_scores = normalize_scores(letterboxd_scores)
    # metacritic_scores = normalize_scores(metacritic_scores)
    ####################################################################################

    combined_lex = {}
    for m, s in letterboxd_scores.items():
        combined_lex[m] = combined_lex.get(m, 0.0) + s
    for m, s in metacritic_scores.items():
        combined_lex[m] = combined_lex.get(m, 0.0) + s

    # Semantic scores (summary-based)
    semantic_scores = compute_semantic_scores(query)

    if combined_lex:
        sorted_lex = sorted(combined_lex.items(), key=lambda kv: kv[1], reverse=True)
        candidate_movies = [m for m, _ in sorted_lex[:candidate_k]]
    else:
        sorted_sem = sorted(semantic_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidate_movies = [m for m, _ in sorted_sem[:candidate_k]]

    # Combine scores
    combined_scores = combine_scores(
        letterboxd_scores,
        metacritic_scores,
        rating_profile,
        emotion_profile,
        query_emotion_vec,
        semantic_scores,
        candidate_movies
    )

    ranked = sorted(combined_scores.items(), key=lambda x:x[1], reverse=True)

    return ranked

if __name__== "__main__":
    query = "dark somber gritty"
    letterboxd_path = "indexes/letterboxd_index.json"
    metacritic_path = "indexes/metacritic_index.json"

    results = process_query(query, letterboxd_path, metacritic_path)

    print(f"\nTop Results for Query: `{query}`\n")
    for movie, score in results[:10]:
        print(f"{movie}: {score}")