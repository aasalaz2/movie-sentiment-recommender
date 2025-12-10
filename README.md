# Movie Sentiment Recommender

## Overview
A hybrid movie recommendation system using BM25 lexical search, semantic embeddings, review sentiment, and emotion-vector similarity to align with the emotional tone a user's query.
This project combines:
* Letterboxd reviews
* Metacritic summaries
* MiniLM semantic embedding (`all-MiniLM-L6-v2`)
* VADER-style sentiment aggregation
* NRC-based emotion vectors
* Weighted hybrid scoring & ranking
* LLM-based refinement using LLaMA 3 via Ollama

## How the Software Works
The system operates in a modular pipeline:
1. **Data Preprocessing** \
   Cleans the raw Letterboxd and Metacritic datasets, computes sentiment and emotion vectors, and builds BM25 and semantic indexes. Implemented in `run_data_pipeline.py` and preprocessing scripts.
2. **Query Processing** \
   The user query is processed using BM25 lexical search, MiniLM semantic embeddings, sentiment alignment, and emotion-vector similarity. Implemented in `src/query_processing.py`.
3. **Hybrid Ranking** \
   The system combines lexical, semantic, sentiment, and emotion scores using a weighted formula to produce a ranked list of candidate movies.
4. **LLM Refinement (Ollama + LLaMA 3)** \
   The top movies and query context are passed to the LLM (`ollama_integration.py`) to generate an explanation and refined ranking.
5. **Command-Line Interface** \
   `run_movie_recommender.py` provides the interactive CLI, loads indexes, handles user input, and displays results.

## Data Source
This recommender uses the public **Movie Reviews Dataset: 10k+ Scraped Data** from Kaggle, available here:
https://www.kaggle.com/datasets/joyshil0599/movie-reviews-dataset-10k-scraped-data/data.

If needed, download the dataset and place the CSV files into:
```bash
data/raw
```

This is the only dataset required for the data pipeline.

## Files

### `run_data_pipeline.py`  
Runs the entire preprocessing workflow:
  - Executes overview & cleaning notebooks
  - Processes both datasets
  - Builds sentiment/emotion profiles
  - Creates BM25 + semantic indexes

### `run_movie_recommender.py`
Launches the interactive CLI and calls:
```bash
src/query_interface.py
```

## Setup and Usage

Follow these steps to set up the environment, run the data pipeline, and start the movie recommender.

### 1. Create a Virtual Environment
For reliable dependency installation, please use **Python 3.12 or higher**. 
```bash
python -m venv movie_env
```

### 2. Activate the Environment
#### Windows:
```bash
movie_env\Scripts\activate
```

#### macOS/Linux:
```bash
source movie_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
**Install TextBlob's linguistic corpa.**\
*(Required for review sentiment analysis)*
```bash
python -m textblob.download_corpora
```

### 4. Install and Configure Ollama
This project uses **LLaMA 3** via **Ollama** to generate emotionally-aware ranking explanations and final recommendations.

**Install Ollama**\
If you haven't installed it yet, download it here:
https://ollama.com/

**Pull the LLaMA 3 model**
```bash
ollama pull llama3
```

### 5. Run the Data Pipeline
This builds cleaned datasets, sentiment profiles, indexes, and semantic embeddings.
```bash
python run_data_pipeline.py
```

### 6. Run the Movie Recommender
```bash
python run_movie_recommender.py
```
