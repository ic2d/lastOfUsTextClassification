import json
import time
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import src.config as config
from src.dataset import load_dataset
from src.pretrained import get_roberta_embeddings
from src.runExperiment import build_features, evaluate, get_classifier, get_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------------
# Hyperparameter search spaces (Bayesian optimization with TPE)
# -------------------------------

def suggest_params(trial, algorithm):
    match algorithm:
        case "SVM":
            return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
        case "RF":
            return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])}
        case "DT":
            return {"max_depth": trial.suggest_int("max_depth", 2, 50),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
        case "KNN":
            return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "metric": trial.suggest_categorical("metric", ["euclidean", "cosine"])}
        case "GNB":
            # equal priors play the role of class weighting
            return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True),
                    "priors": trial.suggest_categorical("priors", [None, "equal"])}
    raise ValueError("Invalid algorithm: " + algorithm)

def make_classifier(algorithm, params, seed):
    params = dict(params)
    if params.get("priors") == "equal":
        params["priors"] = [0.5, 0.5]
    return get_classifier(algorithm, seed).set_params(**params)

# -------------------------------
# Inner loop: TPE search over the training part of an outer fold, maximizing the
# mean BAC of an inner stratified k-fold. Trials whose partial mean is below the
# median of the previous trials are pruned.
# -------------------------------

def tune(X, y, algorithm, seed):
    inner = list(StratifiedKFold(n_splits=config.TUNING_INNER_FOLDS, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        model = make_classifier(algorithm, suggest_params(trial, algorithm), seed)
        scores = []
        for step, (fit_idx, val_idx) in enumerate(inner):
            m = clone(model).fit(X[fit_idx], y[fit_idx])
            scores.append(balanced_accuracy_score(y[val_idx], m.predict(X[val_idx])))
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=config.TUNING_TRIALS)
    return study.best_params, study.best_value

# -------------------------------
# Nested cross-validation: same outer folds as runExperiment; the features are
# fitted on the outer training part, and the test fold is never used for tuning
# -------------------------------

def runTunedExperiment(dataset, featureExtract, algorithm, seed):

    df = load_dataset(dataset)
    X = get_roberta_embeddings(dataset) if featureExtract == "ROBERTA" else df["reviewText"].values
    y = df["label"].values

    metrics, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(get_folds(df, seed), start=1):
        start = time.time()
        features = build_features(featureExtract, algorithm, seed)
        X_train = features.fit_transform(X[train_idx], y[train_idx])
        X_test = features.transform(X[test_idx])

        params, inner_bac = tune(X_train, y[train_idx], algorithm, seed)
        model = make_classifier(algorithm, params, seed).fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        metrics.append({"dataset": dataset, "feature": featureExtract, "algorithm": algorithm,
                        "seed": seed, "fold": fold, "time": elapsed, "inner_bac": inner_bac,
                        "params": json.dumps(params), **evaluate(y[test_idx], y_pred)})
        predictions.append(pd.DataFrame({"row": test_idx, "id": df["id"].values[test_idx], "fold": fold,
                                         "y_true": y[test_idx], "y_pred": y_pred}))
        print(f"   fold {fold}: BAC = {metrics[-1]['bac']:.4f} (inner {inner_bac:.4f}, {elapsed:.0f}s) {params}")

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)

# -------------------------------
# -------------------------------
