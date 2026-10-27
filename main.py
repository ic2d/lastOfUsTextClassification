# ----------------------------------
# Creating virtualenv (Python 3.12):
# ----------------------------------

#   >> python3.12 -m venv last
#   >> source last/bin/activate
#   >> pip install -r requirements.txt

# ----------------------------------
# ----------------------------------

# How to run a single setup:
# main.py [dataset] [featureExtract] [algorithm] [seed] [tuned] [TEXTBLOB|VADER]
#
# How to run all setups (and the lexicon baselines):
# generalSetup.py [n_workers] [tuned] [TEXTBLOB|VADER]

# ----------------------------------
# ----------------------------------

import sys
import os
import src.config as config

from src.runExperiment import runExperiment
from src.tuning import runTunedExperiment

# ----------------------------------
# ----------------------------------

# default hyperparameters -> results/, tuned (nested CV) -> results/tuned/;
# models trained with lexicon labels -> results/[tuned/]labels_{textblob,vader}/
def output_dir(tuned=False, labels="SCORE"):
    out = "results/tuned/" if tuned else "results/"
    return out if labels == "SCORE" else out + "labels_" + labels.lower() + "/"

def parse_options(args):
    for arg in args:
        if arg != "tuned" and arg not in config.VALID_LABELS:
            raise ValueError("Invalid option: " + arg)
    labels = [arg for arg in args if arg in config.VALID_LABELS]
    return {"tuned": "tuned" in args, "labels": labels[0] if labels else "SCORE"}

def output_files(dataset, featureExtract, algorithm, seed, tuned=False, labels="SCORE"):
    name = dataset + "_" + featureExtract + "_" + algorithm + "_seed_" + str(seed)
    out = output_dir(tuned, labels)
    return out + "metrics/" + name + ".csv", out + "predictions/" + name + ".csv.gz"

def run_setup(dataset, featureExtract, algorithm, seed, tuned=False, labels="SCORE"):
    if tuned and labels != "SCORE":
        raise ValueError("Tuning is only available with the score labels")
    metrics_file, predictions_file = output_files(dataset, featureExtract, algorithm, seed, tuned, labels)

    # check if output file exists
    if os.path.exists(metrics_file):
        print(" - Output file already exists. Not running again: " + metrics_file)
        return

    print(" - Running: " + metrics_file)
    if tuned:
        metrics, predictions = runTunedExperiment(dataset=dataset, featureExtract=featureExtract,
                                                  algorithm=algorithm, seed=seed)
    else:
        metrics, predictions = runExperiment(dataset=dataset, featureExtract=featureExtract,
                                             algorithm=algorithm, seed=seed, labels=labels)

    os.makedirs(output_dir(tuned, labels) + "metrics", exist_ok=True)
    os.makedirs(output_dir(tuned, labels) + "predictions", exist_ok=True)
    predictions.to_csv(predictions_file, index=False)
    # metrics last: its presence marks the setup as complete
    metrics.to_csv(metrics_file, index=False)
    print(" - Finished: " + metrics_file)

# ----------------------------------
# ----------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: main.py [dataset] [featureExtract] [algorithm] [seed] [tuned] [TEXTBLOB|VADER]")
        sys.exit(1)

    dataset, featureExtract, algorithm, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    if dataset not in config.VALID_DATASETS:
        print("Dataset is not valid")
        sys.exit(1)

    if featureExtract not in config.VALID_FEATURES:
        print("Feature Type is not valid")
        sys.exit(1)

    if algorithm not in config.VALID_ALGORITHMS:
        print("Algorithm is not valid")
        sys.exit(1)

    if (featureExtract, algorithm) in config.INVALID_SETUPS:
        print("Algorithm " + algorithm + " does not support " + featureExtract + " features")
        sys.exit(1)

    if not seed.isdigit():
        print("Seed is not valid")
        sys.exit(1)

    run_setup(dataset, featureExtract, algorithm, int(seed), **parse_options(sys.argv[5:]))

# ----------------------------------
# ----------------------------------
