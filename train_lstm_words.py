"""
train_lstm_words.py
Trains an LSTM to classify dynamic word gestures (hello, thankyou, father,
mother, ...) from sequences over time. Each training example is a
(SEQUENCE_LENGTH, 143) array: per frame, 126 hand-shape features PLUS 17
features saying where the hands are relative to the face (utils.WORD_FEATURES).
The LSTM learns the hand shape, the movement AND the location -- location is
what separates words like FATHER (thumb at the forehead) from MOTHER (thumb
at the chin), which have the same handshape and movement.

Reads:  data/words/<word>/*.npy   (each file shape (SEQUENCE_LENGTH, 143))
Writes: models/asl_lstm_words.h5
        models/words_label_map.json

Run:    python train_lstm_words.py
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from utils import WORD_FEATURES, TOTAL_FEATURES, SEQUENCE_LENGTH

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "words")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


MIN_SEQUENCES_PER_WORD = 2  # stratified train/val split needs at least this many per class


def load_dataset():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"No data found at {DATA_DIR}. Run data_collection.py --mode words first.")

    words = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not words:
        raise ValueError("No word folders found. Collect data before training.")

    # First pass: count valid sequences per word, so we can skip any word
    # that doesn't have enough for a stratified train/val split (instead of
    # crashing the whole run).
    sequences_by_word = {}
    old_format = 0
    for word in words:
        word_dir = os.path.join(DATA_DIR, word)
        valid_seqs = []
        for fname in sorted(os.listdir(word_dir)):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(word_dir, fname))
            if seq.shape == (SEQUENCE_LENGTH, TOTAL_FEATURES):
                old_format += 1        # hand-shape only: recorded before location was added
                continue
            if seq.shape != (SEQUENCE_LENGTH, WORD_FEATURES):
                print(f"Skipping {word}/{fname}: unexpected shape {seq.shape} "
                      f"(expected {(SEQUENCE_LENGTH, WORD_FEATURES)})")
                continue
            valid_seqs.append(seq)
        sequences_by_word[word] = valid_seqs

    if old_format:
        print(f"\nNOTE: skipped {old_format} older recording(s) that only contain hand SHAPE "
              f"({TOTAL_FEATURES} numbers per frame). Words now also record where the hands are "
              f"relative to your face ({WORD_FEATURES} per frame), which the old files don't "
              "have and can't be converted -- re-record those words with:\n"
              "    python data_collection.py --mode words\n")

    usable_words = []
    X, y = [], []
    for word, seqs in sequences_by_word.items():
        if len(seqs) < MIN_SEQUENCES_PER_WORD:
            print(f"WARNING: skipping word '{word}' -- only {len(seqs)} sequence(s) recorded, "
                  f"need at least {MIN_SEQUENCES_PER_WORD}. Record more with: "
                  f"python data_collection.py --mode words")
            continue
        usable_words.append(word)
        for seq in seqs:
            X.append(seq)
            y.append(word)

    if not X:
        raise ValueError(
            "No word had enough sequences to train on. "
            f"Every word needs at least {MIN_SEQUENCES_PER_WORD} recordings."
        )

    return np.array(X, dtype=np.float32), np.array(y), usable_words


def build_lstm(num_classes):
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, WORD_FEATURES)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(128, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X, y_raw, words = load_dataset()
    print(f"Loaded {len(X)} sequences across {len(words)} words: {words}")

    label_map = {int(i): w for i, w in enumerate(words)}
    word_to_idx = {w: i for i, w in label_map.items()}
    y = np.array([word_to_idx[w] for w in y_raw])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_lstm(num_classes=len(words))
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=8,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    model.save(os.path.join(MODEL_DIR, "asl_lstm_words.h5"))
    with open(os.path.join(MODEL_DIR, "words_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("Saved model to models/asl_lstm_words.h5")
    print("Saved label map to models/words_label_map.json")


if __name__ == "__main__":
    main()