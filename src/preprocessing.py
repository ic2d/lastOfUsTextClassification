import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------------------
# download nltk resources (first time only)
# -------------------------------

NLTK_RESOURCES = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab",
                  "stopwords": "corpora/stopwords", "wordnet": "corpora/wordnet.zip", "omw-1.4": "corpora/omw-1.4.zip"}

for resource, path in NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

# -------------------------------
# Stopwords: NLTK English list without negations, which carry sentiment
# ("not good" must not become "good")
# -------------------------------

NEGATIONS = {
    "no", "nor", "not", "don", "don't", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
    "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't", "needn",
    "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't", "but", "against", "very", "too",
}

STOPWORDS = set(stopwords.words("english")) - NEGATIONS

LEMMATIZER = WordNetLemmatizer()

# -------------------------------
# Preprocessing Text
# -------------------------------

def preprocess_text(text):
    # Tokenize the lowercased text ("didn't" -> "did", "n't")
    tokens = word_tokenize(str(text).lower())
    # Map the clitic negation to a full word
    tokens = ["not" if token == "n't" else token for token in tokens]
    # Keep alphabetic tokens only (drops punctuation, numbers and symbols)
    tokens = [token for token in tokens if token.isalpha()]
    # Remove stop words (negations are kept)
    tokens = [token for token in tokens if token not in STOPWORDS]
    # Lemmatize the tokens
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# -------------------------------
# -------------------------------
