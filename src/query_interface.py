import os
from query_processing import process_query
from ollama_integration import generate_text, build_prompt

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
            print("No emotionally relevant matches were found. The query did not contain enough sentiment/emotion cues for the system to identify suitable movies.\n")
            continue
            
        prompt_input = build_prompt(query, results)
        output = generate_text(prompt_input)
        print("\n" + "-" * 60 + "\n")
        print(output)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()