import os
import pandas as pd
from pathlib import Path

merged = Path("data/processed/merged_movies.csv")

outdir = Path("data/processed")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(merged)
df["movie_name"] = df["movie_name"].astype(str).str.strip()


# >= 4.0 positive
# <= 2.0 negative
# else neutral

def rating_label(rating):
    if pd.isna(rating): 
            return None
    if rating >= 4.0:   
            return "positive"
    if rating <= 2.0:   
            return "negative"
    return "neutral"

df["label"] = df["rating"].apply(rating_label)

g = df.groupby("movie_name", dropna=True)
agg = g.agg(
    n_total=("label","size"),
    n_pos =("label", lambda x: (x=="positive").sum()),
    n_neu =("label", lambda x: (x=="neutral").sum()),
    n_neg =("label", lambda x: (x=="negative").sum()),
    avg_rating_raw=("rating","mean"),
).reset_index()

agg["avg_rating"] = (agg["avg_rating_raw"].fillna(0)/5.0).astype(float)
agg.loc[agg["avg_rating_raw"].isna(), "avg_rating"] = 0.5
agg["pos_rate"] = (agg["n_pos"]/agg["n_total"]).fillna(0.0)

agg_out = agg[["movie_name","n_total","n_pos","n_neu","n_neg","pos_rate","avg_rating"]]

def clean(group):
    texts = []
    texts += group["review"].dropna().astype(str).tolist()
    texts += group["summary"].dropna().astype(str).tolist()
    return " ".join(texts)

docs = g.apply(clean, include_groups=False).reset_index(name="movie_doc")
agg_out.to_csv(os.path.join(outdir, "movie_sentiment_agg.csv"), index=False)
docs.to_csv(os.path.join(outdir, "movie_docs.csv"), index=False)