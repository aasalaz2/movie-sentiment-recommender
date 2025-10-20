import pandas as pd

# Read clean datasets
letterbox_df = pd.read_csv("data/processed/letterboxd_reviews_clean.csv")
metacritic_df = pd.read_csv("data/processed/metacritic_reviews_clean.csv")

merged_df = pd.merge(letterbox_df, metacritic_df, on="movie_name", how="left")

merged_df.to_csv("data/processed/merged_movies.csv", index=False)