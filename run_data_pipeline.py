import subprocess
import sys
import os

# This script will be used to run all overviews + preprocessing + inverted index building

# List of notebooks
NOTEBOOKS = [
    ("notebooks/letterbox_data_overview.ipynb",         "data/reports/letterbox_profile.html"),
    ("notebooks/letterbox_data_preprocessing.ipynb",    "data/processed/letterboxd_reviews_clean.csv"),
    ("notebooks/metacritic_data_overview.ipynb",        "data/reports/metacritic_profile.html"),
    ("notebooks/metacritic_data_preprocessing.ipynb",   "data/processed/metacritic_reviews_clean.csv"),
]
# List of scripts
SCRIPTS = [
    ("src/merge_datasets.py",            "data/processed/merged_movies.csv"),
    ("src/movie_profiles.py",            "data/processed/movie_docs.csv"),
    ("src/build_sentiment_profiles.py",  "data/processed/movie_sentiment_agg.csv"),
    ("src/inverted_index.py",            "indexes/letterboxd_index.json"),
    ("src/build_semantic_index.py",      "data/processed/movie_semantic_embeddings.npz")
]


def run_notebook(nb_path, out_path):
    """Exectue a Jupyter notebook."""
    if os.path.exists(out_path):
        print(f"Skipping {nb_path} (already built: {out_path})")
        return

    print(f"Running {nb_path}...")
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", nb_path, "--inplace"],
            check=True
        )
        print(f"Finished {nb_path}\n")
    except subprocess.CalledProcessError:
        print(f"Error running {nb_path}")
        sys.exit(1)


def run_script(script_path, out_path):
    """Execute a Python script."""
    if os.path.exists(out_path):
        print(f"Skipping {script_path} (already built: {out_path})")
        return

    print(f"Running {script_path}...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"Finished {script_path}\n")
    except subprocess.CalledProcessError:
        print(f"Error running {script_path}")
        sys.exit(1)


def main():
    print("\n===================================")
    print("          PIPELINE STARTED           ")
    print("===================================\n")

    # Run all notebooks
    for nb, out_path in NOTEBOOKS:
        run_notebook(nb, out_path)

    # Run all scripts
    for script, out_path in SCRIPTS:
        run_script(script, out_path)

    print("\n===================================")
    print("  PIPELINE COMPLETE — ALL GOOD!")
    print("===================================\n")

if __name__ == "__main__":
    main()