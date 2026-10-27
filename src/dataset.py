import os
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import src.config as config
from src.preprocessing import preprocess_text

# -------------------------------
# Load a dataset with score-based labels, preprocessed texts and lexicon polarities.
# Everything here is deterministic and computed per review, so it is cached once
# and shared by all experiments (no information flows between reviews).
# -------------------------------

def load_dataset(dataset):
    cache_file = "data/processed/" + dataset + ".csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, keep_default_na=False)

    print(" - Reading dataset: " + dataset)
    data = pd.read_csv("data/" + dataset + ".csv")

    # English reviews only, without duplicated texts
    df = data.loc[data["language"] == "English", ["id", "review", "score"]].copy()
    df = df.drop_duplicates(subset="review", keep="first")

    # Labels from the user score; ambiguous scores are removed
    df = df.loc[(df["score"] >= config.POSITIVE_MIN_SCORE) | (df["score"] <= config.NEGATIVE_MAX_SCORE)]
    df["label"] = (df["score"] >= config.POSITIVE_MIN_SCORE).astype(int)

    print(" - Computing lexicon polarities (TextBlob, VADER)")
    vader = SentimentIntensityAnalyzer()
    df["textblob"] = df["review"].apply(lambda text: TextBlob(text).polarity)
    df["vader"] = df["review"].apply(lambda text: vader.polarity_scores(text)["compound"])

    print(" - Preprocessing texts")
    df["reviewText"] = df["review"].apply(preprocess_text)

    df = df.reset_index(drop=True)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df

# -------------------------------
# -------------------------------
