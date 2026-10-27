import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

import src.config as config

# -------------------------------
# Word2Vec document embeddings (mean pooling of word vectors).
# Implemented as a scikit-learn transformer so that the embedding model is
# trained on the training folds only, inside the cross-validation.
# -------------------------------

class Word2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, sg=0, vector_size=100, window=5, min_count=2, seed=1):
        self.sg = sg
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.seed = seed

    def fit(self, X, y=None):
        corpus = [text.split() for text in X]
        # a single worker makes the training deterministic for a given seed
        self.model_ = Word2Vec(corpus, vector_size=self.vector_size, window=self.window,
                               min_count=self.min_count, sg=self.sg, seed=self.seed, workers=1)
        return self

    def transform(self, X):
        embeddings = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            words = [self.model_.wv[word] for word in text.split() if word in self.model_.wv]
            if words:
                embeddings[i] = np.mean(words, axis=0)
        return embeddings

# -------------------------------
# Return the (unfitted) feature extractor
# -------------------------------

def get_feature_extractor(featureExtract, seed):
    match featureExtract:
        case "BOW":
            return CountVectorizer(ngram_range=config.NGRAM_RANGE, min_df=config.MIN_DF,
                                   max_features=config.MAX_FEATURES)
        case "TFIDF":
            return TfidfVectorizer(ngram_range=config.NGRAM_RANGE, min_df=config.MIN_DF,
                                   max_features=config.MAX_FEATURES)
        case "WESG":
            return Word2VecVectorizer(sg=1, vector_size=config.EMBEDDING_SIZE, window=config.EMBEDDING_WINDOW,
                                      min_count=config.EMBEDDING_MIN_COUNT, seed=seed)
        case "WECBOW":
            return Word2VecVectorizer(sg=0, vector_size=config.EMBEDDING_SIZE, window=config.EMBEDDING_WINDOW,
                                      min_count=config.EMBEDDING_MIN_COUNT, seed=seed)
    raise ValueError("Invalid feature: " + featureExtract)

# -------------------------------
# -------------------------------
