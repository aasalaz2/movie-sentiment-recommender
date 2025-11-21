import json
import pandas as pd
import numpy as np
from collections import defaultdict
from utils import clean_query, normalize_movie_name, cosine_similarity, cosine
from emotion_model import load_movie_emotion_vectors, compute_query_emotion_vector
from semantic_model import compute_semantic_scores

EMOTIONS = [
    "combined_anger", "combined_fear", "combined_sadness", "combined_joy",
    "combined_disgust", "combined_surprise", "combined_trust", "combined_anticipation"
]

# Score weights
weight_reviews=0.2
weight_summaries=0.25
weight_rating_sentiment=0.15
weight_emotion_match=0.2
weight_semantic_scores=0.2

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
    
def is_invalid_emotion_vector(vec):
    """Returns True if the movie's emotion vector is invalid / misleading."""
    vec = np.array(vec)

    # Case 1: All zeros
    if np.all(vec == 0):
        return True
    
    # Case 2: Only one non-zero dimension
    nonzero = np.count_nonzero(vec > 0.05)
    if nonzero <= 1:
        return True
    
    # Case 3: One emotion dominating unrealistically
    if vec.max() > 0.9:
        return True
    
    # Case 4: Very low total emotion
    if vec.sum() < 0.1:
        return True
    
    return False

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def filter_candidates(candidate_movies, emotion_profile, query_emotion_vec,
                    emotion_threshold=0.12):
    """Filters candidates list."""
    # Emotion Filtering
    emotion_filtered = []
    for movie in candidate_movies:
        movie = normalize_movie_name(movie)
        movie_vec = emotion_profile.get(movie, np.zeros_like(query_emotion_vec))

        if is_invalid_emotion_vector(movie_vec):
            continue

        sim = cosine(movie_vec, query_emotion_vec)
        if sim >= emotion_threshold:
            emotion_filtered.append(movie)

    if not emotion_filtered:
        emotion_filtered = candidate_movies
    
    return emotion_filtered

def rating_sentiment(movie, sentiment_profile):
    if movie not in sentiment_profile:
        return 0
    return sentiment_profile[movie]["pos_rate"]

def retrieve_candidates(tokens, inverted_index, document_frequency, doc_lengths, avg_length, k1=1.5, b=0.75):
    """Retrieve candidate movies that contain one or more query tokens."""
    candidates = defaultdict(int)

    for token in tokens:
        if token in inverted_index:
            for entry in inverted_index[token]:
                IDF = document_frequency.get(token, 0)
                
                movie = normalize_movie_name(entry["movie_name"])
                freq = entry.get("count", 1)

                document_length = doc_lengths.get(movie, 0) 

                bm25 = IDF * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (document_length / avg_length)))

                candidates[movie] += bm25
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
    max_review = max(letterboxd_scores.values()) if letterboxd_scores else 1
    max_summary = max(metacritic_scores.values()) if metacritic_scores else 1

    # Combine all movies
    # all_movies = set(letterboxd_scores) | set(metacritic_scores) | set(emotion_profile) | set(semantic_scores)

    for movie in candidate_movies:
        movie = normalize_movie_name(movie)
        
        # Review score normalized
        review_s = letterboxd_scores.get(movie, 0) / max_review

        # Summary score normalized
        summary_s = metacritic_scores.get(movie, 0) / max_summary

        # Rating-based sentiment score
        rating_s = rating_profile.get(movie, {}).get("pos_rate", 0)

        # Emotion similarity
        movie_vec = emotion_profile.get(movie, np.zeros(len(EMOTIONS)))
        emotion_match = cosine_similarity(movie_vec, query_emotion_vec)
        if emotion_match < 0.08:
            continue

        # Semantic score for movie
        semantic_s = semantic_scores.get(movie, 0.0)

        lex_strength = review_s + summary_s
        if lex_strength < 0.01:
            semantic_s *= 0.3
            emotion_match *= 0.5

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
    
    letterboxd_document_frequency, metacritic_document_frequency = load_indexes("indexes/letterboxd_document_frequency.json", "indexes/metacritic_document_frequency.json")

    tokens = clean_query(query)
    if not tokens:
        return []
    
    # Retrieve raw term-frequency matches
    letterboxd_scores = retrieve_candidates(tokens, letterboxd_index, letterboxd_document_frequency, letterboxd_doc_lengths, letterboxd_avg_len)
    metacritic_scores = retrieve_candidates(tokens, metacritic_index, metacritic_document_frequency, metacritic_doc_lengths, metacritic_avg_len)

    combined_lex = {}
    for m, s in letterboxd_scores.items():
        combined_lex[m] = combined_lex.get(m, 0.0) + s
    for m, s in metacritic_scores.items():
        combined_lex[m] = combined_lex.get(m, 0.0) + s

    # Semantic scores (summary-based)
    semantic_scores = compute_semantic_scores(query)

    top_lex = sorted(combined_lex.items(), key=lambda x: x[1], reverse=True)[:200]
    top_sem = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:200]

    candidate_movies = list(
        { normalize_movie_name(m) for m, _ in top_lex } |
        { normalize_movie_name(m) for m, _ in top_sem }
    )

    candidate_movies = filter_candidates(
        candidate_movies,
        emotion_profile,
        query_emotion_vec
    )


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