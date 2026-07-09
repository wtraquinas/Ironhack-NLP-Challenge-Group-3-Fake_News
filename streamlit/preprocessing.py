import re

import nltk

def download_nltk():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt": "punkt"
    }

    for path, resource in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

download_nltk()

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [

        lemmatizer.lemmatize(w)

        for w in words

        if w not in stop_words

    ]

    return " ".join(words)