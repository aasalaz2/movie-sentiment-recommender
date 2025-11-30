import ollama
import pandas as pd
import re
# here's the general pipeline:
# First, we gather the context
# The top 10 documents, and their metacritic summaries and 
# the sentiment profiles
# we feed all these contexts into the prompt
# and ask the AI to generate a response as to which is the most relevant
# including the original question

def generate_text(prompt_input):
    result = ollama.generate(
        model="llama3",
        prompt=prompt_input
    )
    return result.text

def build_prompt(query, results):
    df = pre_process_metacritic()
    prompt = "You are a movie search assistant. Given the following contexts and a user query, determine which movie is relevant to .\n\n"
    prompt += "User Query:\n"
    prompt += f"{query}\n\n"
    prompt += "Top-result movies and their descriptions:\n"
    for i, (movie, score) in enumerate(results[:10], start=1):
        # you can add more context here, such as the sentiment 
        # prompt += f"Movie {i}: {movie} (Score: {score:.4f})\n"
        prompt += f"Movie {i}: {movie} \n"
        description = df.loc[df['Movie name'] == movie, 'summary'].values[0]
        prompt += f"Description {i}:\n{description}\n\n"
    prompt += "Based on the above descriptions, because describe which movies best fit the user's query."
    return prompt

def pre_process_metacritic():
    df = pd.read_csv("../data/raw/metacritic-reviews.csv", encoding='latin1', on_bad_lines="skip")

    def clean_movie_title(title):
        if pd.isna(title):
            return title
        title = re.sub(r"\s*\(\d{4}\)\s*", "", str(title))

        return title.strip().lower()

    df["Movie name"] = df["Movie name"].apply(clean_movie_title)
    return df
