import os
from query_processing import process_query
from text_cleaning import clean_query

def main():
    print("\nMovie Search Engine\n")
    print("Type a search query (or 'exit' to quit):\n")

    letterboxd_path = "indexes/letterboxd_index.json"
    metacritic_path = "indexes/metacritic_index.json"

    # Confirm index files exist
    if not (os.path.exists(letterboxd_path) and os.path.exists(metacritic_path)):
        print("Error: index files not found. Run 'run_data_pipeline.py' first.")
        return
    
    while True:
        query = input("Enter query: ").strip()

        if query.lower() == "exit":
            print("\nGoodbye!\n")
            break

        if not query:
            continue

        print("\nSearching...\n")

        # Run query engine
        results = process_query(query, letterboxd_path, metacritic_path)

        if not results:
            print("No results found.\n")
            continue

        print("\n" + "-" * 60 + "\n")
        print("Top Results:\n")
        for i, (movie, score) in enumerate(results[:10], start=1):
            print(f"{i:2d}. {movie:<40}  Score: {score:.4f}")

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()