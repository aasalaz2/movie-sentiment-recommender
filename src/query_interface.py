import os
from query_processing import process_query, process_title_query
from ollama_integration import generate_text, build_prompt

def main():
    print("\nMovie Search Engine\n")
    print("Type a search query ('-help' for options, or 'exit' to quit):\n")

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
        if query.lower() == "-help":
            print("\nThe following is a sentiment based movie search engine. Enter a query about the emotional tone of a movie and\nwe will return the most relevant movies to your query.\nYou can also enter -title to do a standard title search to view available movies\n")
            continue
        if query.lower() == "-title":
            movie_prefix = input("\nEnter a movie title/prefix: ").strip()
            title_results = process_title_query(movie_prefix)
            if len(title_results) == 0:
                print("\nNo movies found with that prefix\n")
            else:
                print("\n" + "-" * 60 + "\n")
                print(f"Movies starting with '{movie_prefix}':\n")
                for i, movie in enumerate(title_results, start=1):
                    print(f"{i:2d}. {movie:<40}")
                print("\n" + "-" * 60 + "\n")
            continue


        if not query:
            continue

        print("\nSearching...\n")

        # Run query engine
        results = process_query(query, letterboxd_path, metacritic_path)

        if not results:
            print("No emotionally relevant matches were found. The query did not contain enough sentiment/emotion cues for the system to identify suitable movies.\n\nSearching using standard title search...")
            title_results = process_title_query(query)
            if len(title_results) == 0:
                print("""No movies found with that prefix""")
            else:
                print("\n" + "-" * 60 + "\n")
                print(f"Movies starting with {query}:\n")
                for i, movie in enumerate(title_results, start=1):
                    print(f"{i:2d}. {movie:<40}")
                print("\n" + "-" * 60 + "\n")
            continue
            
        prompt_input = build_prompt(query, results)
        output = generate_text(prompt_input)
        print("\n" + "-" * 60 + "\n")
        print(output)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()