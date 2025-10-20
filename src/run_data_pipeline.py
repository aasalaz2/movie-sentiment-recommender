import subprocess
import sys

# This script will be used to run all overviews + preprocessing + inverted index building

# List of notebooks
NOTEBOOKS = [
    "notebooks/letterbox_data_overview.ipynb",
    "notebooks/letterbox_data_preprocessing.ipynb",
    "notebooks/metacritic_data_overview.ipynb",
    "notebooks/metacritic_data_preprocessing.ipynb"
]

def run_notebook(nb_path):
    """Exectue a Jupyter notebook."""
    print(f"Running {nb_path}...")
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", nb_path, "--inplace"]
            )
        print(f"Finished {nb_path}\n")
    except subprocess.CalledProcessError:
        print(f"Error running {nb_path}")
        sys.exit(1)

def main():
    # Run all notebooks
    for nb in NOTEBOOKS:
        run_notebook(nb)

    # Build inverted indexes


if __name__ == "__main__":
    main()