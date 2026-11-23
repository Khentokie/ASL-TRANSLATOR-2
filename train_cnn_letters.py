"""
train_cnn_letters.py
Trains a CNN to classify static ASL letters from normalized hand-landmark
"images". Each 126-length feature vector (2 hands x 21 landmarks x 3 coords)
is reshaped into a (21, 3, 2) tensor: 21 landmark "rows", 3 coordinate
"columns" (x, y, z), 2 "channels" (left hand, right hand). A small 2D CNN
then learns spatial patterns across landmarks the same way it would learn
patterns across pixels.

Reads:  data/letters/<LETTER>/<index>.npy   (each shape (SEQUENCE_LENGTH, 126))
        Every frame of every recording is unpacked into its own training
        sample -- the CNN still classifies a single static handshape, but
        each recording gives you ~30 naturally-varied frames instead of one.
Writes: models/asl_cnn_letters.h5
        models/letters_label_map.json

Run:    python train_cnn_letters.py
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from utils import NUM_LANDMARKS, COORDS, TOTAL_FEATURES, SEQUENCE_LENGTH

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "letters")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    X, y = [], []
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"No data found at {DATA_DIR}. Run data_collection.py --mode letters first.")

    letter_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not letter_dirs:
        raise ValueError("No letter folders found. Collect data before training.")

    # J and Z are motion letters, not static poses -- they're trained and
    # recognized separately by train_lstm_jz.py / the live app's motion
    # branch. Including their transitional "mid-swoop" frames here, all
    # labeled as if they were one static pose, would hurt this model rather
    # than help it.
    excluded = {"J", "Z"}
    letter_dirs = [d for d in letter_dirs if d not in excluded]
    if excluded & set(os.listdir(DATA_DIR)):
        print(f"Note: skipping {sorted(excluded)} -- those are trained by train_lstm_jz.py instead.")

    for letter in letter_dirs:
        letter_dir = os.path.join(DATA_DIR, letter)
        for fname in sorted(os.listdir(letter_dir)):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(letter_dir, fname))
            if seq.ndim != 2 or seq.shape[1] != TOTAL_FEATURES:
                print(f"Skipping {letter}/{fname}: unexpected shape {seq.shape}")
                continue
            # Unpack every frame of the recording as its own static-pose sample,
            # skipping frames where no hand was detected (all-zero feature rows).
            for frame in seq:
                if np.any(frame):
                    X.append(frame)
                    y.append(letter)

    if not X:
        raise ValueError("No samples found. Collect data before training.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def reshape_for_cnn(X):
    """
    (N, 126) -> (N, 21, 3, 2)
    First 63 columns = left hand (21 x 3), last 63 = right hand (21 x 3).
    """
    n = X.shape[0]
    left = X[:, :63].reshape(n, NUM_LANDMARKS, COORDS)
    right = X[:, 63:].reshape(n, NUM_LANDMARKS, COORDS)
    stacked = np.stack([left, right], axis=-1)  # (N, 21, 3, 2)
    return stacked


def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(NUM_LANDMARKS, COORDS, 2)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    X_raw, y_raw = load_dataset()
    print(f"Loaded {len(X_raw)} samples across {len(set(y_raw))} letters.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}

    X = reshape_for_cnn(X_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_cnn(num_classes=len(encoder.classes_))
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    model.save(os.path.join(MODEL_DIR, "asl_cnn_letters.h5"))
    with open(os.path.join(MODEL_DIR, "letters_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("Saved model to models/asl_cnn_letters.h5")
    print("Saved label map to models/letters_label_map.json")


if __name__ == "__main__":
    main()