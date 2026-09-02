import os
import json
import pickle
import argparse
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

INPUT_CSV = "./articles_with_domains_present_preprocessed.csv"

FEATURE_DIR = "simple_preproc_features"

OUTPUT_DIR = "bilstm_experiments_v3"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

SVD_DIR = os.path.join(OUTPUT_DIR, "svd")


for directory in [
    OUTPUT_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    PLOT_DIR,
    RESULT_DIR,
    SVD_DIR,
]:

    os.makedirs(directory, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEQUENCE_LENGTH = 400

MAX_VOCAB_SIZE = 50000

SVD_COMPONENTS = 256

EMBEDDING_DIM = 200

HIDDEN_DIM = 128

NUM_LAYERS = 2

# exp_v1
# DROPOUT = 0.4

# exp_v2
# DROPOUT = 0.3

# exp_v3
DROPOUT = 0.5

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

PATIENCE = 3

GRADIENT_CLIP = 5.0

NUM_WORKERS = 0

EXPERIMENTS = ["tfidf", "bow", "ngram", "all"]

LABELS = ["HIGH", "LOW", "MIXED"]

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

parser = argparse.ArgumentParser()

parser.add_argument("--input", default=INPUT_CSV)

parser.add_argument(
    "--experiment",
    default="all",
    choices=["tfidf", "bow", "ngram", "all"],
    help="Run one experiment or all experiments",
)

parser.add_argument(
    "--resume", action="store_true", help="Resume interrupted experiments"
)

parser.add_argument(
    "--device", default="cuda", choices=["auto", "cpu", "directml", "cuda"]
)

args = parser.parse_args()


if args.device == "cpu":

    DEVICE = torch.device("cpu")

    DEVICE_NAME = "CPU"


elif args.device == "cuda":

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA requested but is not available.")

    DEVICE = torch.device("cuda")

    DEVICE_NAME = "CUDA"


elif args.device == "directml":

    import torch_directml

    DEVICE = torch_directml.device()

    DEVICE_NAME = "DirectML"


else:

    try:

        import torch_directml

        DEVICE = torch_directml.device()

        DEVICE_NAME = "DirectML"

    except ImportError:

        if torch.cuda.is_available():

            DEVICE = torch.device("cuda")

            DEVICE_NAME = "CUDA"

        else:

            DEVICE = torch.device("cpu")

            DEVICE_NAME = "CPU"


print("=" * 80)
print("BiLSTM FEATURE EXPERIMENTS")
print("=" * 80)

print("Device:", DEVICE_NAME)

print("Device object:", DEVICE)

print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(args.input)

required_columns = ["domain", "processed_text", "factuality_rating"]

missing = [column for column in required_columns if column not in df.columns]

if missing:

    raise ValueError(f"Missing columns: {missing}")

df["processed_text"] = df["processed_text"].fillna("").astype(str)

df["domain"] = df["domain"].fillna("").astype(str)

print("\nLoading saved domain split...")

with open(os.path.join(FEATURE_DIR, "splits", "domain_split.pkl"), "rb") as f:

    split_data = pickle.load(f)

train_indices = split_data["train_indices"]

validation_indices = split_data["validation_indices"]

test_indices = split_data["test_indices"]

train_indices = np.array(train_indices, dtype=np.int64)

validation_indices = np.array(validation_indices, dtype=np.int64)

test_indices = np.array(test_indices, dtype=np.int64)

train_domains = set(df.iloc[train_indices]["domain"])

validation_domains = set(df.iloc[validation_indices]["domain"])

test_domains = set(df.iloc[test_indices]["domain"])

assert train_domains.isdisjoint(validation_domains)

assert train_domains.isdisjoint(test_domains)

assert validation_domains.isdisjoint(test_domains)

print("Domain leakage check: PASSED")

print(
    f"Train:      {len(train_indices):,} articles / "
    f"{len(train_domains):,} domains"
)

print(
    f"Validation: {len(validation_indices):,} articles / "
    f"{len(validation_domains):,} domains"
)

print(
    f"Test:       {len(test_indices):,} articles / "
    f"{len(test_domains):,} domains"
)

VOCAB_PATH = os.path.join(FEATURE_DIR, "bilstm_vocabulary.pkl")


def build_vocabulary(texts, max_vocab_size=50000):

    print("\nBuilding BiLSTM vocabulary...")

    word_frequency = {}

    for text in tqdm(texts, desc="Counting words"):

        tokens = str(text).split()

        for token in tokens:

            word_frequency[token] = word_frequency.get(token, 0) + 1

    sorted_words = sorted(
        word_frequency.items(), key=lambda x: x[1], reverse=True
    )

    vocabulary = {"<PAD>": 0, "<UNK>": 1}

    available_slots = max_vocab_size - 2

    for word, _ in sorted_words[:available_slots]:

        if word not in vocabulary:

            vocabulary[word] = len(vocabulary)

    print(f"Vocabulary size: " f"{len(vocabulary):,}")

    return vocabulary


if os.path.exists(VOCAB_PATH):

    print("\nLoading existing BiLSTM vocabulary...")

    with open(VOCAB_PATH, "rb") as f:

        word_to_id = pickle.load(f)

else:

    word_to_id = build_vocabulary(
        df["processed_text"], max_vocab_size=MAX_VOCAB_SIZE
    )

    with open(VOCAB_PATH, "wb") as f:

        pickle.dump(word_to_id, f)

    print(f"Vocabulary saved to: " f"{VOCAB_PATH}")


PAD_ID = word_to_id["<PAD>"]

UNK_ID = word_to_id["<UNK>"]


def encode_text(text):

    tokens = str(text).split()

    tokens = tokens[:MAX_SEQUENCE_LENGTH]

    ids = [word_to_id.get(token, UNK_ID) for token in tokens]

    if len(ids) < MAX_SEQUENCE_LENGTH:

        ids.extend([PAD_ID] * (MAX_SEQUENCE_LENGTH - len(ids)))

    return ids


print("\nEncoding text sequences...")

encoded_text = np.zeros((len(df), MAX_SEQUENCE_LENGTH), dtype=np.int32)

for i, text in enumerate(tqdm(df["processed_text"], desc="Encoding")):

    encoded_text[i] = encode_text(text)

labels = np.array(
    [LABEL_TO_ID.get(label, -1) for label in df["factuality_rating"]]
)

if np.any(labels < 0):

    raise ValueError("Unknown factuality_rating detected.")

FEATURE_FILES = {
    "tfidf": {
        "train": "tfidf_train.npz",
        "validation": "tfidf_validation.npz",
        "test": "tfidf_test.npz",
    },
    "bow": {
        "train": "bow_train.npz",
        "validation": "bow_validation.npz",
        "test": "bow_test.npz",
    },
    "ngram": {
        "train": "ngram_train.npz",
        "validation": "ngram_validation.npz",
        "test": "ngram_test.npz",
    },
}


def save_pickle(obj, path):

    with open(path, "wb") as f:

        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):

    with open(path, "rb") as f:

        return pickle.load(f)


def load_feature_set(feature_name):

    print(f"\nLoading {feature_name.upper()}...")

    files = FEATURE_FILES[feature_name]

    train_matrix = load_npz(os.path.join(FEATURE_DIR, files["train"]))

    validation_matrix = load_npz(
        os.path.join(FEATURE_DIR, files["validation"])
    )

    test_matrix = load_npz(os.path.join(FEATURE_DIR, files["test"]))

    print("Train:", train_matrix.shape)

    print("Validation:", validation_matrix.shape)

    print("Test:", test_matrix.shape)

    return (train_matrix, validation_matrix, test_matrix)


def prepare_feature_svd(
    feature_name, train_matrix, validation_matrix, test_matrix
):

    print("\n")
    print("=" * 80)

    print(f"SVD: {feature_name.upper()}")

    print("=" * 80)

    svd_file = os.path.join(SVD_DIR, f"{feature_name}_svd.pkl")

    train_file = os.path.join(SVD_DIR, f"{feature_name}_train.npy")

    validation_file = os.path.join(SVD_DIR, f"{feature_name}_validation.npy")

    test_file = os.path.join(SVD_DIR, f"{feature_name}_test.npy")

    if (
        os.path.exists(svd_file)
        and os.path.exists(train_file)
        and os.path.exists(validation_file)
        and os.path.exists(test_file)
    ):

        print("Loading cached SVD...")

        svd = load_pickle(svd_file)

        X_train = np.load(train_file)

        X_validation = np.load(validation_file)

        X_test = np.load(test_file)

        return (X_train, X_validation, X_test)

    n_components = min(SVD_COMPONENTS, train_matrix.shape[1] - 1)

    print(f"SVD components: " f"{n_components}")

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)

    print("Fitting SVD on TRAIN...")

    X_train = svd.fit_transform(train_matrix).astype(np.float32)

    print("Transforming VALIDATION...")

    X_validation = svd.transform(validation_matrix).astype(np.float32)

    print("Transforming TEST...")

    X_test = svd.transform(test_matrix).astype(np.float32)

    save_pickle(svd, svd_file)

    np.save(train_file, X_train)

    np.save(validation_file, X_validation)

    np.save(test_file, X_test)

    print("SVD cached.")

    return (X_train, X_validation, X_test)


class FeatureDataset(Dataset):

    def __init__(self, sequence_data, feature_data, labels):

        self.sequence_data = sequence_data

        self.feature_data = feature_data

        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        sequence = torch.tensor(self.sequence_data[index], dtype=torch.long)

        feature = torch.tensor(self.feature_data[index], dtype=torch.float32)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return (sequence, feature, label)


class BiLSTMFeatureClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        feature_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        num_classes,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_ID,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequence, features):

        embedded = self.embedding(sequence)

        lstm_output, _ = self.lstm(embedded)

        text_features = torch.max(lstm_output, dim=1).values

        text_features = self.text_projection(text_features)

        feature_features = self.feature_projection(features)

        combined = torch.cat([text_features, feature_features], dim=1)

        logits = self.classifier(combined)

        return logits


def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_true = []

    all_pred = []

    with torch.no_grad():

        for sequences, features, labels_batch in loader:

            sequences = sequences.to(DEVICE)

            features = features.to(DEVICE)

            labels_batch = labels_batch.to(DEVICE)

            logits = model(sequences, features)

            loss = criterion(logits, labels_batch)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            all_true.extend(labels_batch.cpu().numpy())

            all_pred.extend(predictions.cpu().numpy())

    metrics = calculate_metrics(all_true, all_pred)

    metrics["loss"] = total_loss / len(loader)

    metrics["y_true"] = all_true

    metrics["y_pred"] = all_pred

    return metrics


def save_training_checkpoint(
    experiment, epoch, model, optimizer, history, best_f1, patience_counter
):

    path = os.path.join(CHECKPOINT_DIR, f"{experiment}_checkpoint.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
        },
        path,
    )


def save_plots(experiment, history):

    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:

        plt.figure(figsize=(9, 6))

        plt.plot(history[f"val_{metric}"], marker="o")

        plt.title(f"{experiment.upper()} - " f"Validation {metric.upper()}")

        plt.xlabel("Epoch")

        plt.ylabel(metric.upper())

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(PLOT_DIR, f"{experiment}_{metric}.png"), dpi=150
        )

        plt.close()

    plt.figure(figsize=(9, 6))

    plt.plot(history["train_loss"], marker="o", label="Train")

    plt.plot(history["val_loss"], marker="o", label="Validation")

    plt.title(f"{experiment.upper()} - Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, f"{experiment}_loss.png"), dpi=150)

    plt.close()


def train_experiment(experiment):

    print("\n\n")

    print("#" * 80)

    print(f"STARTING EXPERIMENT: " f"{experiment.upper()}")

    print("#" * 80)

    start_time = time.time()

    if experiment == "all":

        tfidf = load_feature_set("tfidf")

        bow = load_feature_set("bow")

        ngram = load_feature_set("ngram")

        tfidf_reduced = prepare_feature_svd("tfidf", *tfidf)

        bow_reduced = prepare_feature_svd("bow", *bow)

        ngram_reduced = prepare_feature_svd("ngram", *ngram)

        X_train = np.concatenate(
            [tfidf_reduced[0], bow_reduced[0], ngram_reduced[0]], axis=1
        )

        X_validation = np.concatenate(
            [tfidf_reduced[1], bow_reduced[1], ngram_reduced[1]], axis=1
        )

        X_test = np.concatenate(
            [tfidf_reduced[2], bow_reduced[2], ngram_reduced[2]], axis=1
        )

    else:

        matrices = load_feature_set(experiment)

        reduced = prepare_feature_svd(experiment, *matrices)

        X_train = reduced[0]

        X_validation = reduced[1]

        X_test = reduced[2]

    feature_dim = X_train.shape[1]

    print("\nFinal feature dimension:", feature_dim)

    train_dataset = FeatureDataset(
        encoded_text[train_indices], X_train, labels[train_indices]
    )

    validation_dataset = FeatureDataset(
        encoded_text[validation_indices],
        X_validation,
        labels[validation_indices],
    )

    test_dataset = FeatureDataset(
        encoded_text[test_indices], X_test, labels[test_indices]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = BiLSTMFeatureClassifier(
        vocab_size=len(word_to_id),
        feature_dim=feature_dim,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=len(LABELS),
    )

    model = model.to(DEVICE)

    train_labels = labels[train_indices]

    class_counts = np.bincount(train_labels, minlength=len(LABELS))

    total = len(train_labels)

    class_weights = []

    for count in class_counts:

        if count == 0:

            weight = 1.0

        else:

            weight = total / (len(LABELS) * count)

        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_f1 = -np.inf

    patience_counter = 0

    start_epoch = 0

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"{experiment}_checkpoint.pt"
    )

    if args.resume and os.path.exists(checkpoint_path):

        print("\nLoading checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        history = checkpoint["history"]

        best_f1 = checkpoint["best_f1"]

        patience_counter = checkpoint["patience_counter"]

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch " f"{start_epoch + 1}")

    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()

        print("\n" + "-" * 80)

        print(f"{experiment.upper()} " f"Epoch {epoch + 1}/{EPOCHS}")

        print("-" * 80)

        model.train()

        total_loss = 0.0

        progress = tqdm(train_loader, desc="Training")

        for sequences, features, labels_batch in progress:

            sequences = sequences.to(DEVICE)

            features = features.to(DEVICE)

            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(sequences, features)

            loss = criterion(logits, labels_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)

        validation = evaluate(model, validation_loader, criterion)

        history["train_loss"].append(train_loss)

        history["val_loss"].append(validation["loss"])

        history["val_accuracy"].append(validation["accuracy"])

        history["val_precision"].append(validation["precision"])

        history["val_recall"].append(validation["recall"])

        history["val_f1"].append(validation["f1"])

        elapsed = time.time() - epoch_start

        print(f"\nTrain loss: " f"{train_loss:.4f}")

        print(f"Validation loss: " f"{validation['loss']:.4f}")

        print(f"Validation accuracy: " f"{validation['accuracy']:.4f}")

        print(f"Validation precision: " f"{validation['precision']:.4f}")

        print(f"Validation recall: " f"{validation['recall']:.4f}")

        print(f"Validation F1: " f"{validation['f1']:.4f}")

        print(f"Epoch time: " f"{elapsed:.2f} sec")

        if validation["f1"] > best_f1:

            best_f1 = validation["f1"]

            patience_counter = 0

            model_path = os.path.join(MODEL_DIR, f"{experiment}_best.pt")

            torch.save(
                {
                    "experiment": experiment,
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
                    "vocab_size": len(word_to_id),
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "labels": LABELS,
                    "best_validation_f1": best_f1,
                },
                model_path,
            )

            print("\nNew best model saved.")

        else:

            patience_counter += 1

        save_training_checkpoint(
            experiment,
            epoch,
            model,
            optimizer,
            history,
            best_f1,
            patience_counter,
        )

        print("Checkpoint saved.")

        if patience_counter >= PATIENCE:

            print("\nEarly stopping.")

            break

    history_path = os.path.join(RESULT_DIR, f"{experiment}_history.json")

    with open(history_path, "w", encoding="utf-8") as f:

        json.dump(history, f, indent=4)

    best_model_path = os.path.join(MODEL_DIR, f"{experiment}_best.pt")

    checkpoint = torch.load(best_model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nEvaluating TEST...")

    test_results = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 80)

    print(f"FINAL TEST RESULTS - " f"{experiment.upper()}")

    print("=" * 80)

    print(f"Accuracy:  " f"{test_results['accuracy']:.4f}")

    print(f"Precision: " f"{test_results['precision']:.4f}")

    print(f"Recall:    " f"{test_results['recall']:.4f}")

    print(f"F1:        " f"{test_results['f1']:.4f}")

    report = classification_report(
        test_results["y_true"],
        test_results["y_pred"],
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    )

    print("\nClassification report:")

    print(report)

    report_path = os.path.join(
        RESULT_DIR, f"{experiment}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write(report)

    save_plots(experiment, history)

    elapsed_total = time.time() - start_time

    result = {
        "experiment": experiment,
        "accuracy": test_results["accuracy"],
        "precision": test_results["precision"],
        "recall": test_results["recall"],
        "f1": test_results["f1"],
        "best_validation_f1": best_f1,
        "epochs": len(history["train_loss"]),
        "feature_dimension": feature_dim,
        "training_time_seconds": elapsed_total,
    }

    with open(
        os.path.join(RESULT_DIR, f"{experiment}_result.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(result, f, indent=4)

    return result


if args.experiment == "all":

    experiments_to_run = ["tfidf", "bow", "ngram", "all"]

else:

    experiments_to_run = [args.experiment]

all_results = []

for experiment in experiments_to_run:

    result = train_experiment(experiment)

    all_results.append(result)

results_df = pd.DataFrame(all_results)

results_df = results_df[
    [
        "experiment",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "best_validation_f1",
        "epochs",
        "feature_dimension",
        "training_time_seconds",
    ]
]

results_df = results_df.sort_values("f1", ascending=False)

comparison_path = os.path.join(RESULT_DIR, "experiment_comparison.csv")

results_df.to_csv(comparison_path, index=False)

metrics = ["accuracy", "precision", "recall", "f1"]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    plt.bar(results_df["experiment"], results_df[metric])

    plt.title(f"BiLSTM Feature Comparison - " f"{metric.upper()}")

    plt.xlabel("Experiment")

    plt.ylabel(metric.upper())

    plt.ylim(0, 1)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, f"comparison_{metric}.png"), dpi=150)

    plt.close()

print("\n\n")

print("=" * 80)

print("ALL EXPERIMENTS FINISHED")

print("=" * 80)

print("\nFinal comparison:")

print(results_df.to_string(index=False))

print("\nComparison saved to:")

print(comparison_path)
