from movie_profiles import rating_label
from query_processing import process_query, process_title_query
from emotion_model import load_movie_emotion_vectors
from semantic_model import compute_semantic_scores
from build_sentiment_profiles import compute_sentiment


def test_rating_label():
    assert rating_label(4.5) == "positive"
    assert rating_label(1.5) == "negative"
    assert rating_label(3.222) == "neutral"

def test_basic_query():
    query = "fun and exciting action"
    letterboxd_path = "indexes/letterboxd_index.json"
    metacritic_path = "indexes/metacritic_index.json"
    results = process_query(query, letterboxd_path, metacritic_path)
    assert results != None
    for i, (movie, score) in enumerate(results[:10], start=1):
        assert len(movie) > 0
        assert score > 0
        assert score <= 1

def test_expected_movie():
    query = "fun and exciting action"
    expected_movie = "toy story 2"
    letterboxd_path = "indexes/letterboxd_index.json"
    metacritic_path = "indexes/metacritic_index.json"
    results = process_query(query, letterboxd_path, metacritic_path)
    assert results != None
    movie_list = []
    for i, (movie, score) in enumerate(results[:10], start=1):
        movie_list.append(movie)
    #Substitute with an expected movie
    assert expected_movie in movie_list

def test_emotion_profile():
    emotion_vectors = load_movie_emotion_vectors()
    assert emotion_vectors is not None
    assert len(emotion_vectors) != 0

def test_semantic_scores():
    query = "fun and exciting action"
    semantic_scores = compute_semantic_scores(query)
    assert semantic_scores is not None
    assert len(semantic_scores) > 0

def test_compute_sentiment():
    emotions = ["anger", "fear", "sadness", "joy", "disgust", "surprise", "trust", "anticipation"]
    query = "I am so scared of the unknown"
    emotion_result = compute_sentiment(query)
    assert emotion_result is not None
    assert len(emotion_result) > 0
    for key in emotion_result.keys():
        assert key in emotions

def test_title_search():
    query = "t"
    movie_titles = process_title_query(query)
    assert len(movie_titles) > 0

