import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import normalize_movie_name

_semantic_model = None
_semantic_embeddings = None
_semantic_movie_names = None
_semantic_loaded = False

def build_semantic_index(
        in_path="data/processed/metacritic_reviews_clean.csv",
        out_path="data/processed/movie_semantic_embeddings.npz"):
    """Builds a semantic index of the given path."""
    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)

    df["summary"] = df["summary"].fillna("")

    grouped = (
        df.groupby("movie_name")["summary"]
            .apply(lambda s: " ".join(str(x) for x in s.unique()))
            .reset_index()
    )

    print(f"Found {len(grouped)} movies for embeddings.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


    print("Encoding movie documents...")
    embeddings = model.encode(
        grouped["summary"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True
    )

    names = np.array(
        [normalize_movie_name(m) for m in grouped["movie_name"].astype(str)],
        dtype=object
    )

    print("Saving semantic index to NPZ...")
    np.savez(out_path, movie_names=names, embeddings=embeddings)
    print("Done. Saved to {out_path}")

def load_semantic_index(path="data/processed/movie_semantic_embeddings.npz"):
    """Load NPZ embedding + initialize sentence-transformer model."""
    global _semantic_model, _semantic_embeddings, _semantic_movie_names, _semantic_loaded

    if _semantic_loaded:
        return
    
    data = np.load(path, allow_pickle=True)
    _semantic_movie_names = data["movie_names"]
    _semantic_embeddings = data["embeddings"]
    # _semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    _semantic_loaded = True


def compute_semantic_scores(query):
    """Compute semantic similarity between a query and all movies."""
    if not query or not isinstance(query, str):
        return {}
    
    load_semantic_index()

    q_vec = _semantic_model.encode([query], normalize_embeddings=True)[0]
    sims = _semantic_embeddings @ q_vec

    scores = {}
    for name, sim in zip(_semantic_movie_names, sims):
        scores[str(name)] = max(float(sim), 0.0)

    max_s = max(scores.values()) if scores else 1.0
    if max_s > 0:
        scores = {k: v / max_s for k, v in scores.items()}

    return scores