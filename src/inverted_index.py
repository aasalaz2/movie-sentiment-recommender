import pandas as pd
import json
from collections import defaultdict
import math
from utils import normalize_movie_name

# Function to build inverted inddex from processed text
def build_inverted_index(df, text_column, movie_column, extra_columns=None):
    """Builds an inverted and document length indexes from the given df."""
    inverted_index = defaultdict(list)
    doc_lengths = defaultdict(int)

    for _, row in df.iterrows():
        movie_name = row[movie_column]
        movie_name = normalize_movie_name(movie_name)
        text = row[text_column] if isinstance(row[text_column], str) else ""
        tokens = text.split()

        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1

        # total length of this doc = number of tokens
        doc_lengths[movie_name] += len(tokens)

        for token, count in token_counts.items():
            entry = {"movie_name": movie_name, "count": count}
            if extra_columns:
                for col in extra_columns:
                    entry[col] = row[col]
            inverted_index[token].append(entry)

    return inverted_index, doc_lengths

def build_doc_frequency(df, inverted_index, movie_column):
    """Builds the document frequency index from the given df."""
    document_frequency = {}
    total_docs = df[movie_column].nunique()
    for token in inverted_index:
        document_frequency[token] = math.log((total_docs + 1) / (len(inverted_index[token]) + 1)) + 1
    return document_frequency


if __name__ == "__main__":
    # Letterboxd
    # Load clean Letterboxd dataset
    letterboxd_df = pd.read_csv("data/processed/letterboxd_reviews_clean.csv")
    # Build Letterboxd inverted and document length indexes
    letterboxd_index, letterboxd_doc_lengths = build_inverted_index(
        letterboxd_df,
        text_column="review",
        movie_column="movie_name",
        extra_columns=["rating", "like_count"]
    )
    # Build Letterboxd document frequency index
    letterboxd_docfreq = build_doc_frequency(
        letterboxd_df,
        inverted_index = letterboxd_index,
        movie_column="movie_name"
    )

    # Save Letterboxd inverted index
    with open("indexes/letterboxd_index.json", "w", encoding="utf-8") as f:
        json.dump(letterboxd_index, f, ensure_ascii=False, indent=2)
    # Save Letterboxd document length index
    with open("indexes/letterboxd_doc_lengths.json", "w", encoding="utf-8") as f:
        json.dump(letterboxd_doc_lengths, f, ensure_ascii=False, indent=2)
    # Save Letterboxd document frequency index
    with open("indexes/letterboxd_document_frequency.json", "w", encoding="utf-8") as f:
        json.dump(letterboxd_docfreq, f, ensure_ascii=False, indent=2)

    # Metacritic
    # Load clean Metacritic dataset
    metacritic_df = pd.read_csv("data/processed/metacritic_reviews_clean.csv")
    # Build Metacritic inverted and document length indexes 
    metacritic_index, metacritic_doc_lengths = build_inverted_index(
        metacritic_df,
        text_column="summary",
        movie_column="movie_name",
    )
    # Building Metacritic document frequency index
    metacritic_docfreq = build_doc_frequency(
        metacritic_df,
        inverted_index = metacritic_index,
        movie_column="movie_name"
    )

    # Save Metacritic inverted index
    with open("indexes/metacritic_index.json", "w", encoding="utf-8") as f:
        json.dump(metacritic_index, f, ensure_ascii=False, indent=2)
    # Save Metacritic document length index
    with open("indexes/metacritic_doc_lengths.json", "w", encoding="utf-8") as f:
        json.dump(metacritic_doc_lengths, f, ensure_ascii=False, indent=2)
    # Save Metacritic document frequency index
    with open("indexes/metacritic_document_frequency.json", "w", encoding="utf-8") as f:
        json.dump(metacritic_docfreq, f, ensure_ascii=False, indent=2)
