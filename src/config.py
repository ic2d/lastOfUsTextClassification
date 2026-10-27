# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

VALID_DATASETS = ["user_reviews_g1", "user_reviews_g2"]

VALID_FEATURES = ["TFIDF", "BOW", "WESG", "WECBOW", "ROBERTA"]

VALID_ALGORITHMS = ["KNN", "DT", "RF", "GNB", "SVM"]

# (feature, algorithm) pairs that cannot be combined (none with the current grid)
INVALID_SETUPS = []

SEEDS = list(range(1, 11))
N_FOLDS = 10

# -----------------------------------------------------------------------------------
# Labels come from the score (0-10) the user gave to the game:
#   score >= POSITIVE_MIN_SCORE -> positive (1)
#   score <= NEGATIVE_MAX_SCORE -> negative (0)
#   scores in between are ambiguous and removed
# -----------------------------------------------------------------------------------

POSITIVE_MIN_SCORE = 7
NEGATIVE_MAX_SCORE = 4

# Labels used to TRAIN the ML models: the score labels, or the labels a lexicon
# would assign (to measure the cost of lexicon labels). Testing always uses the score labels.
VALID_LABELS = ["SCORE", "TEXTBLOB", "VADER"]

# Lexicon-based labelers, evaluated as baselines against the score labels
TEXTBLOB_THRESHOLD = 0.0   # polarity > 0 -> positive
VADER_THRESHOLD = 0.05     # compound >= 0.05 -> positive (Hutto & Gilbert, 2014)

# -----------------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------------

NGRAM_RANGE = (1, 2)       # bigrams keep negated expressions such as "not good"
MAX_FEATURES = 5000
MIN_DF = 2

EMBEDDING_SIZE = 100
EMBEDDING_WINDOW = 5
EMBEDDING_MIN_COUNT = 2

# Hyperparameter tuning (nested CV): Bayesian optimization (TPE) with Optuna
TUNING_TRIALS = 25
TUNING_INNER_FOLDS = 3
TUNING_SEEDS = [1, 2, 3]

# -----------------------------------------------------------------------------------
# Pretrained transformers (no fine-tuning), applied to the raw reviews
# -----------------------------------------------------------------------------------

# RoBERTa-large fine-tuned for binary sentiment on reviews/tweets from 15 datasets (baseline)
ROBERTA_SENTIMENT_MODEL = "siebert/sentiment-roberta-large-english"
ROBERTA_THRESHOLD = 0.5    # P(positive) >= 0.5 -> positive

# DistilRoBERTa sentence embeddings (768 dimensions), used as features
ROBERTA_EMBEDDING_MODEL = "sentence-transformers/all-distilroberta-v1"

TRANSFORMER_MAX_LENGTH = 512
TRANSFORMER_BATCH_SIZE = 16

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
