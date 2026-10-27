import glob
import os
import numpy as np
import pandas as pd

from collections import Counter
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

import src.config as config
from src.dataset import load_dataset
from src.pretrained import get_roberta_sentiment

# -------------------------------
# Aggregates the experiment outputs into the tables used by the paper (results/analysis/)
# -------------------------------

OUT = "results/analysis/"
BASELINES = ["LEXICON", "PRETRAINED"]
METRICS = ["bac", "f1_macro", "f1_neg", "precision_neg", "recall_neg", "f1_pos", "accuracy"]

def setup_name(df):
    return df["feature"] + "_" + df["algorithm"]

def load_metrics():
    metrics = pd.concat([pd.read_csv(f) for f in glob.glob("results/metrics/*.csv")], ignore_index=True)
    metrics["setup"] = setup_name(metrics)
    return metrics

# -------------------------------
# Dataset characteristics (Table 1)
# -------------------------------

def dataset_table():
    rows = []
    for dataset in config.VALID_DATASETS:
        raw = pd.read_csv("data/" + dataset + ".csv")
        english = raw.loc[raw["language"] == "English"]
        unique = english.drop_duplicates(subset="review")
        df = load_dataset(dataset)
        neg, pos = (df["label"] == 0).sum(), (df["label"] == 1).sum()
        rows.append({"dataset": dataset, "raw": len(raw), "english": len(english),
                     "duplicates": len(english) - len(unique), "ambiguous": len(unique) - len(df),
                     "examples": len(df), "negative": neg, "positive": pos, "imbalance": neg / pos,
                     "median_tokens": df["review"].str.split().str.len().median()})
    return pd.DataFrame(rows)

# -------------------------------
# Agreement of the lexicon labelers with the score labels (whole dataset)
# -------------------------------

def labeler_table():
    rows = []
    for dataset in config.VALID_DATASETS:
        df = load_dataset(dataset)
        y = df["label"]
        for name, pred in [("TextBlob", (df["textblob"] > config.TEXTBLOB_THRESHOLD).astype(int)),
                           ("VADER", (df["vader"] >= config.VADER_THRESHOLD).astype(int)),
                           ("RoBERTa", (get_roberta_sentiment(dataset) >= config.ROBERTA_THRESHOLD).astype(int))]:
            rows.append({"dataset": dataset, "labeler": name, "accuracy": accuracy_score(y, pred),
                         "bac": balanced_accuracy_score(y, pred), "kappa": cohen_kappa_score(y, pred),
                         "f1_neg": f1_score(y, pred, pos_label=0), "disagreement": (y != pred).mean()})
    return pd.DataFrame(rows)

# -------------------------------
# Friedman test + Nemenyi critical difference over the seed x fold blocks
# -------------------------------

def rank_tests(metrics, dataset, alpha=0.05):
    table = metrics.loc[metrics["dataset"] == dataset].pivot_table(index=["seed", "fold"], columns="setup", values="bac")
    table = table.dropna(axis=1)
    ranks = table.rank(axis=1, ascending=False).mean().sort_values()
    statistic, p_value = stats.friedmanchisquare(*[table[c].values for c in table.columns])
    k, n = table.shape[1], table.shape[0]
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    ranks = ranks.rename("avg_rank").reset_index().assign(dataset=dataset)
    test = pd.DataFrame([{"dataset": dataset, "k": k, "n_blocks": n, "chi2": statistic, "p_value": p_value, "cd": cd}])
    return ranks, test

# -------------------------------
# Corrected resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004):
# best setup vs. every other setup, over the r x k fold-level differences
# -------------------------------

def corrected_ttest(a, b, n_folds):
    d = a - b
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1 / n + 1 / (n_folds - 1)) * var)
    return t, 2 * stats.t.sf(abs(t), df=n - 1)

def pairwise_vs_best(metrics, dataset, summary):
    sub = metrics.loc[metrics["dataset"] == dataset].sort_values(["seed", "fold"])
    ranked = summary.loc[summary["dataset"] == dataset].sort_values("bac_mean", ascending=False)
    best = ranked["setup"].iloc[0]
    a = sub.loc[sub["setup"] == best, "bac"].values
    rows = []
    for other in ranked["setup"].iloc[1:]:
        b = sub.loc[sub["setup"] == other, "bac"].values
        t, p = corrected_ttest(a, b, config.N_FOLDS)
        rows.append({"dataset": dataset, "best": best, "other": other, "diff": a.mean() - b.mean(),
                     "t": t, "p_value": p, "wilcoxon_p": stats.wilcoxon(a, b).pvalue})
    out = pd.DataFrame(rows)
    # Holm correction for the multiple comparisons against the best setup
    order = out["p_value"].sort_values().index
    m = len(out)
    adjusted = np.maximum.accumulate([min(1, (m - i) * out.loc[j, "p_value"]) for i, j in enumerate(order)])
    out.loc[order, "p_holm"] = adjusted
    return out

# -------------------------------
# Out-of-fold predictions of a setup for one seed
# -------------------------------

def load_predictions(dataset, setup, seed):
    feature, algorithm = setup.split("_")
    file = f"results/predictions/{dataset}_{feature}_{algorithm}_seed_{seed}.csv.gz"
    return pd.read_csv(file).sort_values("row").reset_index(drop=True)

def confusion_tables(summary, top=2):
    rows = []
    for dataset in config.VALID_DATASETS:
        ml = summary.loc[(summary["dataset"] == dataset) & (~summary["feature"].isin(BASELINES))]
        for setup in ml.sort_values("bac_mean", ascending=False)["setup"].head(top):
            # mean confusion matrix over the seeds
            cms = [confusion_matrix(p["y_true"], p["y_pred"], labels=[0, 1])
                   for p in (load_predictions(dataset, setup, s) for s in config.SEEDS)]
            cm = np.mean(cms, axis=0)
            for i in range(2):
                for j in range(2):
                    rows.append({"dataset": dataset, "setup": setup, "true": i, "pred": j, "count": cm[i, j]})
    return pd.DataFrame(rows)

# -------------------------------
# Error analysis (seed 1): errors of the top-2 setups, per user score and review length
# -------------------------------

def error_analysis(summary, top=2):
    barcode, by_score, examples = [], [], []
    for dataset in config.VALID_DATASETS:
        df = load_dataset(dataset)
        ml = summary.loc[(summary["dataset"] == dataset) & (~summary["feature"].isin(BASELINES))]
        setups = list(ml.sort_values("bac_mean", ascending=False)["setup"].head(top))
        preds = {s: load_predictions(dataset, s, 1)["y_pred"].values for s in setups}
        frame = pd.DataFrame({"row": np.arange(len(df)), "truth": df["label"].values, **preds})
        barcode.append(frame.melt(id_vars="row", var_name="setup", value_name="label").assign(dataset=dataset))

        best = preds[setups[0]]
        df = df.assign(error=(best != df["label"].values).astype(int), pred=best,
                       both_wrong=np.all([p != df["label"].values for p in preds.values()], axis=0),
                       tokens=df["review"].str.split().str.len(), roberta=get_roberta_sentiment(dataset))
        by_score.append(df.groupby("score").agg(n=("error", "size"), error_rate=("error", "mean")).reset_index()
                        .assign(dataset=dataset, setup=setups[0]))
        wrong = df.loc[df["both_wrong"], ["id", "review", "score", "label", "pred", "textblob", "vader", "roberta", "tokens"]]
        examples.append(wrong.assign(dataset=dataset))
    return pd.concat(barcode), pd.concat(by_score), pd.concat(examples)

# -------------------------------
# Cost of lexicon labels: the same setups trained with TextBlob/VADER labels
# (results/labels_*/), all evaluated against the score labels on the same folds
# -------------------------------

def load_label_metrics(metrics):
    frames = [metrics.loc[~metrics["feature"].isin(BASELINES)].assign(labels="SCORE")]
    for labels in config.VALID_LABELS[1:]:
        files = glob.glob("results/labels_" + labels.lower() + "/metrics/*.csv")
        if files:
            frames.append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    out = pd.concat(frames, ignore_index=True)
    out["setup"] = setup_name(out)
    return out

def label_comparison(label_metrics):
    summary = label_metrics.groupby(["dataset", "labels", "feature", "algorithm", "setup"])[METRICS].agg(["mean", "std"])
    summary.columns = [m + "_" + s for m, s in summary.columns]
    summary = summary.reset_index()
    # corrected resampled t-test: score labels vs. lexicon labels, for every setup
    rows = []
    for (dataset, setup), g in label_metrics.sort_values(["seed", "fold"]).groupby(["dataset", "setup"]):
        a = g.loc[g["labels"] == "SCORE", "bac"].values
        for labels in config.VALID_LABELS[1:]:
            b = g.loc[g["labels"] == labels, "bac"].values
            if len(b) == len(a):
                t, p = corrected_ttest(a, b, config.N_FOLDS)
                rows.append({"dataset": dataset, "setup": setup, "labels": labels,
                             "diff": a.mean() - b.mean(), "t": t, "p_value": p})
    return summary, pd.DataFrame(rows)

# -------------------------------
# Most discriminative terms per class: weighted log-odds ratio with an
# informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008)
# -------------------------------

def discriminative_terms(n_terms=15):
    rows = []
    for dataset in config.VALID_DATASETS:
        df = load_dataset(dataset)
        counts = {c: Counter(" ".join(df.loc[df["label"] == c, "reviewText"]).split()) for c in [0, 1]}
        prior = counts[0] + counts[1]
        a0 = sum(prior.values())
        n = {c: sum(counts[c].values()) for c in [0, 1]}
        for word, aw in prior.items():
            if aw < 10:
                continue
            y0, y1 = counts[0][word], counts[1][word]
            l1 = np.log((y1 + aw) / (n[1] + a0 - y1 - aw))
            l0 = np.log((y0 + aw) / (n[0] + a0 - y0 - aw))
            z = (l1 - l0) / np.sqrt(1 / (y1 + aw) + 1 / (y0 + aw))
            rows.append({"dataset": dataset, "term": word, "z": z, "count": aw})
    terms = pd.DataFrame(rows)
    top = []
    for dataset, g in terms.groupby("dataset"):
        top.append(g.nlargest(n_terms, "z").assign(cls="positive"))
        top.append(g.nsmallest(n_terms, "z").assign(cls="negative"))
    return pd.concat(top)

# -------------------------------
# -------------------------------

if __name__ == "__main__":

    os.makedirs(OUT, exist_ok=True)
    metrics = load_metrics()

    # completeness check: every setup must have all seeds and folds
    counts = metrics.groupby(["dataset", "setup"]).size()
    incomplete = counts[counts != len(config.SEEDS) * config.N_FOLDS]
    if len(incomplete):
        print("WARNING: incomplete setups\n", incomplete)

    summary = metrics.groupby(["dataset", "feature", "algorithm", "setup"])[METRICS].agg(["mean", "std"])
    summary.columns = [m + "_" + s for m, s in summary.columns]
    summary = summary.reset_index().sort_values(["dataset", "bac_mean"], ascending=[True, False])
    summary.to_csv(OUT + "summary.csv", index=False)
    metrics.to_csv(OUT + "all_folds.csv", index=False)

    dataset_table().to_csv(OUT + "datasets.csv", index=False)
    labeler_table().to_csv(OUT + "labelers.csv", index=False)

    ranks, tests, pairs = [], [], []
    for dataset in config.VALID_DATASETS:
        r, t = rank_tests(metrics, dataset)
        ranks.append(r)
        tests.append(t)
        pairs.append(pairwise_vs_best(metrics, dataset, summary))
    pd.concat(ranks).to_csv(OUT + "ranks.csv", index=False)
    pd.concat(tests).to_csv(OUT + "friedman.csv", index=False)
    pd.concat(pairs).to_csv(OUT + "pairwise_vs_best.csv", index=False)

    confusion_tables(summary).to_csv(OUT + "confusion.csv", index=False)
    barcode, by_score, examples = error_analysis(summary)
    barcode.to_csv(OUT + "barcode.csv", index=False)
    by_score.to_csv(OUT + "errors_by_score.csv", index=False)
    examples.to_csv(OUT + "error_examples.csv", index=False)
    discriminative_terms().to_csv(OUT + "terms.csv", index=False)

    label_metrics = load_label_metrics(metrics)
    counts = label_metrics.groupby(["dataset", "labels", "setup"]).size()
    incomplete = counts[counts != len(config.SEEDS) * config.N_FOLDS]
    if len(incomplete):
        print("WARNING: incomplete setups (training labels)\n", incomplete)
    label_summary, label_tests = label_comparison(label_metrics)
    label_summary.to_csv(OUT + "summary_labels.csv", index=False)
    label_tests.to_csv(OUT + "labels_vs_score.csv", index=False)

    print(summary.groupby("dataset").head(6)[["dataset", "setup", "bac_mean", "bac_std", "f1_neg_mean"]].to_string(index=False))
    print(pd.concat(tests).to_string(index=False))

# -------------------------------
# -------------------------------
