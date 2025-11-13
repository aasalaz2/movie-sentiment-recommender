import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_query(q):
    if pd.isna(q):
        return []
    q = re.sub(r"<.*?>", " ", q)
    q = re.sub(r"[^a-zA-Z\s]", " ", q)
    q = q.lower()
    tokens = [lemmatizer.lemmatize(w) for w in q.split() if w not in stop_words]
    return tokens