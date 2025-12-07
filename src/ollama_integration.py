import ollama
import pandas as pd
import re
import os
from utils import normalize_movie_name
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
    return result.response

def clean_movie_title(title):
        if pd.isna(title):
            return title
        title = re.sub(r"\s*\(\d{4}\)\s*", "", str(title))

        return title.strip().lower()

def build_prompt(query, results):
    df = pre_process_metacritic()
    # prompt = "You are a movie search assistant. Given the following contexts and a user query, determine which movie is relevant to .\n\n"
    # prompt += "User Query:\n"
    # prompt += f"{query}\n\n"
    # prompt += "Top-result movies and their descriptions:\n"

    prompt = f"""
    You are a movie recommendation assistant.
    
    The user has entered the following query:
    "{query}"

    You are given a set of candidate movies and short descriptions.
    Your task is to select the best movies based on emotional tone, theme, or narrative that most clearly matches the user's query.
    Remove any queries that are not relevant to the query.

    Important Rules:
    - Do NOT use markdown (no asterisks, bold, italics, or bullet points)
    - Do NOT refer to movies as "Movie 1", "Movie 2", etc.
    - Do NOT add extra commentary or conversational fluff.
    - Do NOT lowercase movie titles.
    - Your final answer MUST use the exact format below.

    Output Format:
    <rank_number>: <movie_title>
    <2-4 sentences explaining why the movie matches the query>

    Candidate Movies:
    """

    for i, (movie, score) in enumerate(results[:10], start=1):
        # you can add more context here, such as the sentiment 
        # prompt += f"Movie {i}: {movie} (Score: {score:.4f})\n"
        prompt += f"{movie} \n"
        # description = df.loc[df['Movie name'] == movie, 'summary'].values[0]
        df['clean_name'] = df['Movie name'].apply(clean_movie_title).apply(normalize_movie_name)
        row = df.loc[df['clean_name'] == movie]

        if row.empty:
            description = "No summary available."
        else:
            description = row['summary'].values[0]

        prompt += f"Description {i}:\n{description}\n\n"
    prompt += "Based on the above descriptions, describe which movies best fit the user's query."
    return prompt

def pre_process_metacritic():
    # df = pd.read_csv("../data/raw/metacritic-reviews.csv", encoding='latin1', on_bad_lines="skip")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "metacritic-reviews.csv")
    df = pd.read_csv(DATA_PATH, encoding="latin1", on_bad_lines="skip")

    df["Movie name"] = df["Movie name"].apply(clean_movie_title).apply(normalize_movie_name)
    return df
