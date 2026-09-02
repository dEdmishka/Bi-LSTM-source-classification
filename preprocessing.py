import os
import re
import json
import pickle
import argparse

import pandas as pd

from tqdm import tqdm
import spacy

from nltk.stem import SnowballStemmer

DEFAULT_INPUT = "./articles_with_domains_present.csv"
DEFAULT_OUTPUT = "./articles_with_domains_present_preprocessed.csv"

CHECKPOINT_DIR = "cache/preprocessing"
CHECKPOINT_EVERY = 1000

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(
    description="News article text preprocessing pipeline"
)

parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV file")

parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV file")

parser.add_argument(
    "--resume", action="store_true", help="Resume from checkpoint"
)

args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

print("=" * 70)
print("Loading spaCy model...")
print("=" * 70)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stemmer = SnowballStemmer("english")

ABBREVIATIONS = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "couldn't": "could not",
    "couldnt": "could not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "hadn't": "had not",
    "hadnt": "had not",
    "it's": "it is",
    "its": "it is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "whos": "who is",
    "where's": "where is",
    "wheres": "where is",
    "you're": "you are",
    "youre": "you are",
    "they're": "they are",
    "theyre": "they are",
    "we're": "we are",
    "were": "we are",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "i'll": "i will",
    "ill": "i will",
    "i'd": "i would",
    "id": "i would",
    "you've": "you have",
    "youve": "you have",
    "you'll": "you will",
    "youll": "you will",
    "they've": "they have",
    "theyve": "they have",
    "they'll": "they will",
    "theyll": "they will",
    "we've": "we have",
    "weve": "we have",
    "we'll": "we will",
    "well": "we will",
}

STOP_WORDS = nlp.Defaults.stop_words


def normalize_text(text):
    """
    Performs:
        1. Unicode-safe conversion to string
        2. Lowercasing
        3. Abbreviation expansion
        4. Whitespace normalization
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    for abbreviation, replacement in ABBREVIATIONS.items():

        pattern = r"\b" + re.escape(abbreviation) + r"\b"

        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_punctuation(text):
    """
    Removes punctuation and replaces it with whitespace.

    Example:

        "Hello, world!" -> "Hello world"
    """

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    return re.sub(r"\s+", " ", text).strip()


def process_document(text):
    """
    Full preprocessing pipeline.

    Returns:

        normalized_text
        tokens
        processed_text
        stemmed_text
    """

    normalized = normalize_text(text)

    normalized = remove_punctuation(normalized)

    doc = nlp(normalized)

    tokens = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_punct:
            continue

        if token.text in STOP_WORDS:
            continue

        if not token.is_alpha:
            continue

        token_text = token.text.lower().strip()

        if not token_text:
            continue

        tokens.append(token_text)

    lemmas = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_punct:
            continue

        if token.text in STOP_WORDS:
            continue

        if not token.is_alpha:
            continue

        lemma = token.lemma_.lower().strip()

        if not lemma:
            continue

        lemmas.append(lemma)

    stems = [stemmer.stem(token) for token in tokens]

    return {
        "normalized_text": normalized,
        "tokens": tokens,
        "processed_text": " ".join(lemmas),
        "stemmed_text": " ".join(stems),
    }


CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "preprocessing_checkpoint.pkl")


def save_checkpoint(dataframe, processed_count):

    checkpoint = {"dataframe": dataframe, "processed_count": processed_count}

    with open(CHECKPOINT_FILE, "wb") as f:

        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nCheckpoint saved: " f"{processed_count:,} rows")


print("=" * 70)
print("Loading dataset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")

print(f"Columns: {len(df.columns)}")

REQUIRED_COLUMNS = [
    "url",
    "title",
    "authors",
    "scrap_date",
    "publish_date",
    "text",
    "keywords",
    "domain",
    "bias_rating",
    "factuality_rating",
    "credibility",
]


missing_columns = [
    column for column in REQUIRED_COLUMNS if column not in df.columns
]


if missing_columns:

    raise ValueError("Missing columns: " + str(missing_columns))

df["input_text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
)

start_index = 0

if args.resume and os.path.exists(CHECKPOINT_FILE):

    print("\nCheckpoint found.")

    with open(CHECKPOINT_FILE, "rb") as f:

        checkpoint = pickle.load(f)

    df = checkpoint["dataframe"]

    start_index = checkpoint["processed_count"]

    print(f"Resuming from row " f"{start_index:,}")

if "normalized_text" not in df.columns:

    df["normalized_text"] = ""

if "tokens" not in df.columns:

    df["tokens"] = ""

if "processed_text" not in df.columns:

    df["processed_text"] = ""

if "stemmed_text" not in df.columns:

    df["stemmed_text"] = ""

print("\n")
print("=" * 70)
print("Starting preprocessing")
print("=" * 70)


for index in tqdm(
    range(start_index, len(df)),
    initial=start_index,
    total=len(df),
    desc="Preprocessing",
):

    text = df.at[index, "input_text"]

    result = process_document(text)

    df.at[index, "normalized_text"] = result["normalized_text"]

    df.at[index, "tokens"] = json.dumps(result["tokens"], ensure_ascii=False)

    df.at[index, "processed_text"] = result["processed_text"]

    df.at[index, "stemmed_text"] = result["stemmed_text"]

    processed_count = index + 1

    if processed_count % CHECKPOINT_EVERY == 0:

        save_checkpoint(df, processed_count)

print("\n")
print("=" * 70)
print("Saving preprocessed dataset...")
print("=" * 70)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

if os.path.exists(CHECKPOINT_FILE):

    os.remove(CHECKPOINT_FILE)

print("\n")
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Input rows:  {len(df):,}")

print(f"Output rows: {len(df):,}")

print(f"Unique domains: " f"{df['domain'].nunique():,}")

print("\nOutput file:")

print(os.path.abspath(OUTPUT_FILE))

print("\n")
print("=" * 70)
print("EXAMPLE")
print("=" * 70)

example = df.iloc[0]

print("\nORIGINAL:")
print(example["input_text"][:500])

print("\nPROCESSED:")
print(example["processed_text"][:500])

print("\nSTEMMED:")
print(example["stemmed_text"][:500])

print("\nTOKENS:")
print(example["tokens"][:500])
