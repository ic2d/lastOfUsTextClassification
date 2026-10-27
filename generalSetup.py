import os
import sys
import itertools
import traceback
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import src.config as config
from src.dataset import load_dataset
from src.runExperiment import runLexiconBaselines, runPretrainedBaseline
from src.pretrained import get_roberta_embeddings
from main import output_dir, parse_options, run_setup

# -------------------------------
# Usage: generalSetup.py [n_workers] [tuned] [TEXTBLOB|VADER]
# -------------------------------

N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
OPTIONS = parse_options(sys.argv[2:])

# -------------------------------
# Run a single job (ML experiment), returning the error message if it fails
# -------------------------------

def runJob(dataset, feature, algorithm, seed):
    try:
        run_setup(dataset, feature, algorithm, seed, **OPTIONS)
        return None
    except Exception:
        return traceback.format_exc()

# -------------------------------
#  Run the complete experiment
# -------------------------------

if __name__ == "__main__":

    os.makedirs("results/metrics", exist_ok=True)

    # Preprocessing and the pretrained models are cached once before the parallel jobs;
    # the baselines need no training
    for dataset in config.VALID_DATASETS:
        load_dataset(dataset)
        get_roberta_embeddings(dataset)
        for seed in config.SEEDS:
            for name, run in [("LEXICON", runLexiconBaselines), ("PRETRAINED", runPretrainedBaseline)]:
                baseline_file = "results/metrics/" + dataset + "_" + name + "_seed_" + str(seed) + ".csv"
                if not os.path.exists(baseline_file):
                    run(dataset, seed).to_csv(baseline_file, index=False)

    seeds = config.TUNING_SEEDS if OPTIONS["tuned"] else config.SEEDS
    setups = [(d, f, a, s) for d, f, a, s in itertools.product(config.VALID_DATASETS, config.VALID_FEATURES,
                                                                 config.VALID_ALGORITHMS, seeds)
              if (f, a) not in config.INVALID_SETUPS]
    print(f"{len(setups)} setups, {N_WORKERS} workers")

    failed = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(runJob, *setup): setup for setup in setups}
        for done, future in enumerate(as_completed(futures), start=1):
            setup = futures[future]
            error = future.result()
            if error:
                failed.append(setup)
                print(f"[{done}/{len(setups)}] FAILED {setup}:\n{error}", flush=True)
            else:
                print(f"[{done}/{len(setups)}] done {setup}", flush=True)

    if failed:
        failed_file = output_dir(**OPTIONS) + "failed.csv"
        pd.DataFrame(failed, columns=["dataset", "feature", "algorithm", "seed"]).to_csv(failed_file, index=False)
        print(f"{len(failed)} setups failed; see " + failed_file)
    else:
        print("All setups completed")

# -------------------------------
# -------------------------------
