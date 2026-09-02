import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DEFAULT_INPUT = "./articles_with_domains_present_preprocessed.csv"

FEATURE_DIR = "simple_preproc_features"

SPLIT_DIR = os.path.join(FEATURE_DIR, "splits")

os.makedirs(FEATURE_DIR, exist_ok=True)

os.makedirs(SPLIT_DIR, exist_ok=True)

SEED = 42

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

MAX_FEATURES = 50000

TFIDF_NGRAM_RANGE = (1, 1)

BOW_NGRAM_RANGE = (1, 1)

NGRAM_RANGE = (1, 2)

MIN_DF = 2
MAX_DF = 0.95

parser = argparse.ArgumentParser(
    description="Extract TF-IDF, Bag-of-Words and n-gram features"
)

parser.add_argument(
    "--input", default=DEFAULT_INPUT, help="Preprocessed input CSV"
)

parser.add_argument(
    "--force",
    action="store_true",
    help="Recalculate existing feature matrices",
)

args = parser.parse_args()

INPUT_FILE = args.input


def save_pickle(obj, path):

    with open(path, "wb") as f:

        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):

    with open(path, "rb") as f:

        return pickle.load(f)


def save_feature_names(vectorizer, path):

    feature_names = vectorizer.get_feature_names_out().tolist()

    with open(path, "w", encoding="utf-8") as f:

        json.dump(feature_names, f, ensure_ascii=False)


def print_matrix_info(name, matrix):

    print(f"{name:15s}: " f"shape={matrix.shape}, " f"non-zero={matrix.nnz:,}")


print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows:    {len(df):,}")

print(f"Domains: {df['domain'].nunique():,}")


required_columns = ["domain", "processed_text", "factuality_rating"]

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:

    raise ValueError(f"Missing columns: {missing_columns}")


df["processed_text"] = df["processed_text"].fillna("").astype(str)

df["domain"] = df["domain"].fillna("").astype(str)

before = len(df)

df = df[df["processed_text"].str.strip() != ""].copy()

after = len(df)

print(f"\nRemoved empty texts: " f"{before - after:,}")

print(f"Remaining rows: {after:,}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)

train_idx, temp_idx = next(gss.split(df, groups=df["domain"]))

train_df = df.iloc[train_idx].copy()

temp_df = df.iloc[temp_idx].copy()

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)

validation_relative_idx, test_relative_idx = next(
    gss_test.split(temp_df, groups=temp_df["domain"])
)

validation_df = temp_df.iloc[validation_relative_idx].copy()

test_df = temp_df.iloc[test_relative_idx].copy()

train_domains = set(train_df["domain"])

validation_domains = set(validation_df["domain"])

test_domains = set(test_df["domain"])

assert train_domains.isdisjoint(validation_domains)

assert train_domains.isdisjoint(test_domains)

assert validation_domains.isdisjoint(test_domains)

print("Domain leakage check: PASSED")

print(
    f"Train:      {len(train_domains):,} domains, "
    f"{len(train_df):,} articles"
)

print(
    f"Validation: {len(validation_domains):,} domains, "
    f"{len(validation_df):,} articles"
)

print(
    f"Test:       {len(test_domains):,} domains, " f"{len(test_df):,} articles"
)

train_indices = df.index[df["domain"].isin(train_domains)].tolist()

validation_indices = df.index[df["domain"].isin(validation_domains)].tolist()

test_indices = df.index[df["domain"].isin(test_domains)].tolist()


print("\nSPLIT STATISTICS")
print("-" * 70)

print(f"Train domains:      {len(train_domains):,}")

print(f"Validation domains: {len(validation_domains):,}")

print(f"Test domains:       {len(test_domains):,}")

print()

print(f"Train articles:      {len(train_indices):,}")

print(f"Validation articles: {len(validation_indices):,}")

print(f"Test articles:       {len(test_indices):,}")

split_data = {
    "seed": SEED,
    "train_domains": sorted(train_domains),
    "validation_domains": sorted(validation_domains),
    "test_domains": sorted(test_domains),
    "train_indices": train_indices,
    "validation_indices": validation_indices,
    "test_indices": test_indices,
}

save_pickle(split_data, os.path.join(SPLIT_DIR, "domain_split.pkl"))

train_texts = df.loc[train_indices, "processed_text"].tolist()

validation_texts = df.loc[validation_indices, "processed_text"].tolist()

test_texts = df.loc[test_indices, "processed_text"].tolist()

train_labels = df.loc[train_indices, "factuality_rating"].values

validation_labels = df.loc[validation_indices, "factuality_rating"].values

test_labels = df.loc[test_indices, "factuality_rating"].values

np.save(os.path.join(SPLIT_DIR, "train_labels.npy"), train_labels)

np.save(os.path.join(SPLIT_DIR, "validation_labels.npy"), validation_labels)

np.save(os.path.join(SPLIT_DIR, "test_labels.npy"), test_labels)

vectorizer_common_params = {
    "max_features": MAX_FEATURES,
    "min_df": MIN_DF,
    "max_df": MAX_DF,
    "lowercase": False,
}

print("\n")
print("=" * 70)
print("EXTRACTING TF-IDF")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=TFIDF_NGRAM_RANGE,
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    lowercase=False,
    sublinear_tf=True,
    dtype=np.float32,
)

tfidf_vectorizer.fit(train_texts)

X_train_tfidf = tfidf_vectorizer.transform(train_texts)

X_validation_tfidf = tfidf_vectorizer.transform(validation_texts)

X_test_tfidf = tfidf_vectorizer.transform(test_texts)

print_matrix_info("TF-IDF train", X_train_tfidf)

print_matrix_info("TF-IDF validation", X_validation_tfidf)

print_matrix_info("TF-IDF test", X_test_tfidf)

save_npz(os.path.join(FEATURE_DIR, "tfidf_train.npz"), X_train_tfidf)

save_npz(os.path.join(FEATURE_DIR, "tfidf_validation.npz"), X_validation_tfidf)

save_npz(os.path.join(FEATURE_DIR, "tfidf_test.npz"), X_test_tfidf)

save_pickle(
    tfidf_vectorizer, os.path.join(FEATURE_DIR, "tfidf_vectorizer.pkl")
)

save_feature_names(
    tfidf_vectorizer, os.path.join(FEATURE_DIR, "tfidf_features.json")
)

print("\n")
print("=" * 70)
print("EXTRACTING BAG-OF-WORDS")
print("=" * 70)

bow_vectorizer = CountVectorizer(
    ngram_range=BOW_NGRAM_RANGE,
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    lowercase=False,
    dtype=np.float32,
)

bow_vectorizer.fit(train_texts)

X_train_bow = bow_vectorizer.transform(train_texts)

X_validation_bow = bow_vectorizer.transform(validation_texts)

X_test_bow = bow_vectorizer.transform(test_texts)

print_matrix_info("BoW train", X_train_bow)

print_matrix_info("BoW validation", X_validation_bow)

print_matrix_info("BoW test", X_test_bow)

save_npz(os.path.join(FEATURE_DIR, "bow_train.npz"), X_train_bow)

save_npz(os.path.join(FEATURE_DIR, "bow_validation.npz"), X_validation_bow)

save_npz(os.path.join(FEATURE_DIR, "bow_test.npz"), X_test_bow)

save_pickle(bow_vectorizer, os.path.join(FEATURE_DIR, "bow_vectorizer.pkl"))

save_feature_names(
    bow_vectorizer, os.path.join(FEATURE_DIR, "bow_features.json")
)

print("\n")
print("=" * 70)
print("EXTRACTING N-GRAM FEATURES")
print("=" * 70)

ngram_vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    lowercase=False,
    dtype=np.float32,
)

ngram_vectorizer.fit(train_texts)

X_train_ngram = ngram_vectorizer.transform(train_texts)

X_validation_ngram = ngram_vectorizer.transform(validation_texts)

X_test_ngram = ngram_vectorizer.transform(test_texts)

print_matrix_info("N-gram train", X_train_ngram)

print_matrix_info("N-gram validation", X_validation_ngram)

print_matrix_info("N-gram test", X_test_ngram)

save_npz(os.path.join(FEATURE_DIR, "ngram_train.npz"), X_train_ngram)

save_npz(os.path.join(FEATURE_DIR, "ngram_validation.npz"), X_validation_ngram)

save_npz(os.path.join(FEATURE_DIR, "ngram_test.npz"), X_test_ngram)

save_pickle(
    ngram_vectorizer, os.path.join(FEATURE_DIR, "ngram_vectorizer.pkl")
)

save_feature_names(
    ngram_vectorizer, os.path.join(FEATURE_DIR, "ngram_features.json")
)

configuration = {
    "input_file": INPUT_FILE,
    "rows": len(df),
    "domains": df["domain"].nunique(),
    "seed": SEED,
    "split": {
        "train": TRAIN_SIZE,
        "validation": VALIDATION_SIZE,
        "test": TEST_SIZE,
    },
    "max_features": MAX_FEATURES,
    "min_df": MIN_DF,
    "max_df": MAX_DF,
    "tfidf_ngram_range": TFIDF_NGRAM_RANGE,
    "bow_ngram_range": BOW_NGRAM_RANGE,
    "ngram_range": NGRAM_RANGE,
    "text_column": "processed_text",
    "target": "factuality_rating",
}

with open(
    os.path.join(FEATURE_DIR, "feature_configuration.json"),
    "w",
    encoding="utf-8",
) as f:

    json.dump(configuration, f, indent=4)

print("\n")
print("=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nSaved files:")

print("\nTF-IDF:")

print("  features/tfidf_train.npz")

print("  features/tfidf_validation.npz")

print("  features/tfidf_test.npz")

print("  features/tfidf_vectorizer.pkl")

print("\nBag-of-Words:")

print("  features/bow_train.npz")

print("  features/bow_validation.npz")

print("  features/bow_test.npz")

print("  features/bow_vectorizer.pkl")

print("\nN-grams:")

print("  features/ngram_train.npz")

print("  features/ngram_validation.npz")

print("  features/ngram_test.npz")

print("  features/ngram_vectorizer.pkl")

print("\nSplit:")

print("  features/splits/domain_split.pkl")

print("\nConfiguration:")

print("  features/feature_configuration.json")

print("\nDone.")
