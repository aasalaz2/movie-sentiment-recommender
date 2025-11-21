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

## Data Source
This recommender uses the public **Movie Reviews Dataset: 10k+ Scraped Data** from Kaggle, available here:
https://www.kaggle.com/datasets/joyshil0599/movie-reviews-dataset-10k-scraped-data/data.

This is the only dataset required for the data pipeline to run successfully.

## Setup and Usage

Follow these steps to set up the environment, run the data pipeline, and start the movie recommender.

### 1. Create a Virtual Environment
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

### 4. Run the Data Pipeline
This builds all cleaned datasets, sentiment profiles, indexes, and semantic embeddings.
```bash
python run_data_pipeline.py
```

### 5. Run the Movie Recommender
```bash
python run_movie_recommender.py
```