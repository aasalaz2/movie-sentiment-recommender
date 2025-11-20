import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_query(q):
    if pd.isna(q):
        return []
    q = re.sub(r"<.*?>", " ", q)
    q = re.sub(r"[^a-zA-Z\s]", " ", q)
    q = q.lower()
    tokens = q.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return tokens

def normalize_movie_name(name):
    return re.sub(r"[^\w\s]", "", name).lower().strip()