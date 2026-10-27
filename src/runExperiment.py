import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

# -------------------------------
# For learning
# -------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# -------------------------------
# Customized functions
# -------------------------------

import src.config as config
from src.dataset import load_dataset
from src.featureExtraction import get_feature_extractor
from src.pretrained import get_roberta_embeddings, get_roberta_sentiment

# -------------------------------
# Classifiers (scikit-learn default hyperparameters)
# -------------------------------

def get_classifier(algorithm, seed):
    match algorithm:
        case "KNN":
            return KNeighborsClassifier()
        case "DT":
            return DecisionTreeClassifier(random_state=seed)
        case "RF":
            return RandomForestClassifier(random_state=seed, n_jobs=1)
        case "MNB":
            return MultinomialNB()
        case "GNB":
            return GaussianNB()
        case "SVM":
            return LinearSVC(random_state=seed)
    raise ValueError("Invalid algorithm: " + algorithm)

def to_dense(X):
    return X.toarray().astype(np.float32) if hasattr(X, "toarray") else X

# -------------------------------
# The whole chain (features -> [scaling] -> classifier) is fitted on the training folds only
# -------------------------------

def build_pipeline(featureExtract, algorithm, seed):
    # RoBERTa embeddings are precomputed by a frozen model: nothing to fit
    extractor = "passthrough" if featureExtract == "ROBERTA" else get_feature_extractor(featureExtract, seed)
    steps = [("features", extractor)]
    if featureExtract in ["WESG", "WECBOW", "ROBERTA"]:
        steps.append(("scaler", StandardScaler()))
    if algorithm == "GNB":
        steps.append(("dense", FunctionTransformer(to_dense, accept_sparse=True)))
    steps.append(("classifier", get_classifier(algorithm, seed)))
    return Pipeline(steps)

# -------------------------------
# Evaluation measures (class 0 = negative, class 1 = positive)
# -------------------------------

def evaluate(y_true, y_pred):
    return {
        "bac": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_neg": f1_score(y_true, y_pred, pos_label=0),
        "precision_neg": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_neg": recall_score(y_true, y_pred, pos_label=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
    }

def get_folds(df, seed):
    skf = StratifiedKFold(n_splits=config.N_FOLDS, random_state=seed, shuffle=True)
    return list(skf.split(df["reviewText"], df["label"]))

# -------------------------------
# Run Experiment: one (dataset, feature, algorithm, seed) setup, 10 folds
# -------------------------------

def runExperiment(dataset, featureExtract, algorithm, seed):

    df = load_dataset(dataset)
    X = get_roberta_embeddings(dataset) if featureExtract == "ROBERTA" else df["reviewText"].values
    y = df["label"].values

    metrics, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(get_folds(df, seed), start=1):
        model = build_pipeline(featureExtract, algorithm, seed)
        start = time.time()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        elapsed = time.time() - start

        metrics.append({"dataset": dataset, "feature": featureExtract, "algorithm": algorithm,
                        "seed": seed, "fold": fold, "time": elapsed, **evaluate(y[test_idx], y_pred)})
        predictions.append(pd.DataFrame({"row": test_idx, "id": df["id"].values[test_idx], "fold": fold,
                                         "y_true": y[test_idx], "y_pred": y_pred}))
        print(f"   fold {fold}: BAC = {metrics[-1]['bac']:.4f}")

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)

# -------------------------------
# Lexicon baselines (TextBlob, VADER): no training, evaluated on the same test folds
# -------------------------------

def runLexiconBaselines(dataset, seed):

    df = load_dataset(dataset)
    y = df["label"].values
    lexicon_pred = {
        "TextBlob": (df["textblob"].values > config.TEXTBLOB_THRESHOLD).astype(int),
        "VADER": (df["vader"].values >= config.VADER_THRESHOLD).astype(int),
    }

    metrics = []
    for fold, (_, test_idx) in enumerate(get_folds(df, seed), start=1):
        for name, y_pred in lexicon_pred.items():
            metrics.append({"dataset": dataset, "feature": "LEXICON", "algorithm": name,
                            "seed": seed, "fold": fold, "time": 0.0,
                            **evaluate(y[test_idx], y_pred[test_idx])})
    return pd.DataFrame(metrics)

# -------------------------------
# Pretrained RoBERTa sentiment classifier (baseline), evaluated on the same test folds
# -------------------------------

def runPretrainedBaseline(dataset, seed):

    df = load_dataset(dataset)
    y = df["label"].values
    y_pred = (get_roberta_sentiment(dataset) >= config.ROBERTA_THRESHOLD).astype(int)

    metrics = []
    for fold, (_, test_idx) in enumerate(get_folds(df, seed), start=1):
        metrics.append({"dataset": dataset, "feature": "PRETRAINED", "algorithm": "RoBERTa",
                        "seed": seed, "fold": fold, "time": 0.0,
                        **evaluate(y[test_idx], y_pred[test_idx])})
    return pd.DataFrame(metrics)

# -------------------------------
# -------------------------------
