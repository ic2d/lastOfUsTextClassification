import os
import numpy as np
import pandas as pd

import src.config as config
from src.dataset import load_dataset

# -------------------------------
# Pretrained transformer models, applied to the raw reviews without fine-tuning.
# Their outputs do not depend on the labels nor on the other reviews, so they are
# computed once per dataset and cached (no leakage across folds).
# -------------------------------

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# RoBERTa sentiment classifier (baseline): probability of the positive class
# -------------------------------

def get_roberta_sentiment(dataset):
    cache_file = "data/processed/" + dataset + "_roberta_sentiment.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)["roberta"].values

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(" - RoBERTa sentiment: " + config.ROBERTA_SENTIMENT_MODEL)
    df = load_dataset(dataset)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.ROBERTA_SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.ROBERTA_SENTIMENT_MODEL).to(device).eval()
    positive = model.config.label2id.get("POSITIVE", 1)

    texts = df["review"].astype(str).tolist()
    scores = []
    with torch.no_grad():
        for start in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
            batch = tokenizer(texts[start:start + config.TRANSFORMER_BATCH_SIZE], truncation=True,
                              max_length=config.TRANSFORMER_MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            probabilities = torch.softmax(model(**batch).logits, dim=-1)[:, positive]
            scores.extend(probabilities.cpu().tolist())

    pd.DataFrame({"id": df["id"], "roberta": scores}).to_csv(cache_file, index=False)
    return np.array(scores)

# -------------------------------
# RoBERTa sentence embeddings (features for the ML algorithms)
# -------------------------------

def get_roberta_embeddings(dataset):
    cache_file = "data/processed/" + dataset + "_roberta_embeddings.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)

    from sentence_transformers import SentenceTransformer

    print(" - RoBERTa embeddings: " + config.ROBERTA_EMBEDDING_MODEL)
    df = load_dataset(dataset)
    model = SentenceTransformer(config.ROBERTA_EMBEDDING_MODEL, device=get_device())
    model.max_seq_length = config.TRANSFORMER_MAX_LENGTH
    embeddings = model.encode(df["review"].astype(str).tolist(), batch_size=config.TRANSFORMER_BATCH_SIZE,
                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(cache_file, embeddings)
    return embeddings

# -------------------------------
# -------------------------------
