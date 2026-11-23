"""
train_lstm_jz.py
Trains the small LSTM that recognizes the motion letters J and Z.

It learns THREE classes:
    J      -- recorded J motions
    Z      -- recorded Z motions
    OTHER  -- "some movement that is NOT J or Z"
OTHER is what stops ordinary hand movement (e.g. moving from one letter to
the next) from being forced into a J or Z guess. You don't record it: it's
built automatically from your existing static-letter recordings (held
poses, synthetic shape-to-shape transitions, and a held shape "moved
around"), so J and Z are the only letters you need motion recordings for.

How the data is prepared:
  * Recordings are split into train/validation BEFORE anything is
    augmented, so near-copies of one recording never land on both sides.
  * Each J/Z training recording also contributes a "trimmed" copy -- just
    the moving part, zero-padded back to full length -- because that is
    exactly what the live app hands the model (see utils.MotionSegmenter).

Reads:  data/letters/J/*.npy, data/letters/Z/*.npy   (needed)
        data/letters/<other letters>/*.npy            (used to build OTHER)
Writes: models/asl_lstm_jz.h5
        models/jz_label_map.json

Run:    python train_lstm_jz.py
"""

import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from utils import TOTAL_FEATURES, SEQUENCE_LENGTH, trim_to_motion

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "letters")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

JZ_LETTERS = ["J", "Z"]
NEGATIVE_LABEL = "OTHER"
MIN_SEQUENCES_PER_LETTER = 2
VAL_FRACTION = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_letter_sequences(letter):
    """All valid (SEQUENCE_LENGTH, 126) recordings for one letter folder."""
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_dir):
        return [], 0
    seqs, skipped = [], 0
    for fname in sorted(os.listdir(letter_dir)):
        if not fname.endswith(".npy"):
            continue
        seq = np.load(os.path.join(letter_dir, fname))
        if seq.shape != (SEQUENCE_LENGTH, TOTAL_FEATURES):
            skipped += 1
            continue
        seqs.append(seq.astype(np.float32))
    return seqs, skipped


def load_static_pool():
    """Recordings of every NON-J/Z letter (and SPACE), keyed by letter."""
    pool = {}
    if not os.path.isdir(DATA_DIR):
        return pool
    for name in sorted(os.listdir(DATA_DIR)):
        if name in JZ_LETTERS or not os.path.isdir(os.path.join(DATA_DIR, name)):
            continue
        seqs, _ = load_letter_sequences(name)
        if seqs:
            pool[name] = seqs
    return pool


# ---------------------------------------------------------------------------
# Building examples
# ---------------------------------------------------------------------------

def pad_to_length(frames, length=SEQUENCE_LENGTH):
    """Zero-pads (or truncates) a list/array of frames to `length` rows --
    the same right-padding the live app applies (the Masking layer ignores it)."""
    arr = np.asarray(frames, dtype=np.float32)
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.zeros((length - len(arr), arr.shape[1]), np.float32)], axis=0)


def split_recordings(seqs, rng, val_fraction=VAL_FRACTION):
    """Shuffles and splits a list of recordings into (train, val)."""
    idx = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_fraction))) if len(seqs) >= 2 else 0
    val = [seqs[i] for i in idx[:n_val]]
    train = [seqs[i] for i in idx[n_val:]]
    return train, val


def trimmed_copy(seq):
    """The 'just the movement' version of a recording, padded back to full
    length -- or None if trimming wouldn't change anything meaningful."""
    trimmed = trim_to_motion(seq)
    if trimmed is None or len(trimmed) >= len(seq) - 2:
        return None
    return pad_to_length(trimmed)


def _random_valid_frame(seq, rng):
    present = np.where(np.any(seq != 0, axis=1))[0]
    return seq[int(rng.choice(present))] if len(present) else None


def make_held(pool, rng):
    """A static letter held still, exactly as recorded."""
    letter = rng.choice(list(pool))
    return pool[letter][int(rng.integers(len(pool[letter])))]


def make_transition(pool, rng):
    """A hand changing from one letter's shape to another's: hold, blend, hold."""
    letters = list(pool)
    if len(letters) < 2:
        return None
    a, b = rng.choice(letters, size=2, replace=False)
    fa = _random_valid_frame(pool[a][int(rng.integers(len(pool[a])))], rng)
    fb = _random_valid_frame(pool[b][int(rng.integers(len(pool[b])))], rng)
    if fa is None or fb is None:
        return None
    hold_a = int(rng.integers(0, 6))
    n_blend = int(rng.integers(6, 21))
    hold_b = int(rng.integers(0, 5))
    frames = [fa] * hold_a
    frames += [(1 - t) * fa + t * fb for t in np.linspace(0.0, 1.0, n_blend)]
    frames += [fb] * hold_b
    frames = np.array(frames, dtype=np.float32)
    frames += rng.normal(0, 0.003, frames.shape).astype(np.float32)
    return pad_to_length(frames)


def make_constant(pool, rng):
    """One handshape held constant for a short stretch -- what the model sees
    when the whole hand is moved around without its shape changing."""
    letter = rng.choice(list(pool))
    f = _random_valid_frame(pool[letter][int(rng.integers(len(pool[letter])))], rng)
    if f is None:
        return None
    n = int(rng.integers(8, 41))
    frames = np.tile(f, (n, 1)) + rng.normal(0, 0.003, (n, f.shape[0])).astype(np.float32)
    return pad_to_length(frames)


def build_other(pool, n, rng):
    """n OTHER examples: ~40% held poses, ~35% transitions, ~25% constant-shape."""
    out = []
    makers = [(make_held, 0.40), (make_transition, 0.35), (make_constant, 0.25)]
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        fn = makers[int(rng.choice(len(makers), p=[w for _, w in makers]))][0]
        ex = fn(pool, rng)
        if ex is not None:
            out.append(ex)
    return out


def build_datasets(jz, pool, rng):
    """
    jz:   {"J": [recordings], "Z": [recordings]}
    pool: {letter: [recordings]} for the static letters (may be empty)
    Returns X_train, y_train, X_val, y_val (label strings), classes.
    """
    X_tr, y_tr, X_va, y_va = [], [], [], []

    for letter, seqs in jz.items():
        train, val = split_recordings(seqs, rng)
        for seq in train:
            X_tr.append(seq); y_tr.append(letter)
            extra = trimmed_copy(seq)          # augmentation: training split only
            if extra is not None:
                X_tr.append(extra); y_tr.append(letter)
        for seq in val:
            X_va.append(seq); y_va.append(letter)

    classes = list(JZ_LETTERS)
    if pool:
        # Split the SOURCE recordings first so synthetic examples built from
        # a validation recording can't leak into training.
        pool_tr, pool_va = {}, {}
        for letter, seqs in pool.items():
            tr, va = split_recordings(seqs, rng)
            if tr: pool_tr[letter] = tr
            if va: pool_va[letter] = va
        per_class_tr = max(sum(1 for y in y_tr if y == l) for l in JZ_LETTERS)
        per_class_va = max(1, max(sum(1 for y in y_va if y == l) for l in JZ_LETTERS))
        for ex in build_other(pool_tr, int(per_class_tr * 1.5), rng):
            X_tr.append(ex); y_tr.append(NEGATIVE_LABEL)
        for ex in build_other(pool_va or pool_tr, max(6, int(per_class_va * 1.5)), rng):
            X_va.append(ex); y_va.append(NEGATIVE_LABEL)
        classes.append(NEGATIVE_LABEL)

    classes = sorted(classes)
    return (np.array(X_tr, np.float32), np.array(y_tr),
            np.array(X_va, np.float32), np.array(y_va), classes)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_lstm(num_classes):
    """Small on purpose: few classes, short sequences, modest dataset."""
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, TOTAL_FEATURES)),
        layers.Masking(mask_value=0.0),   # skips zero-padded rows (see utils.MotionSegmenter)
        layers.LSTM(48, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False, activation="tanh"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def balanced_class_weights(y_idx, num_classes):
    counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def main():
    rng = np.random.default_rng(SEED)

    jz, skipped_total = {}, 0
    for letter in JZ_LETTERS:
        seqs, skipped = load_letter_sequences(letter)
        skipped_total += skipped
        jz[letter] = seqs
        print(f"{letter}: {len(seqs)} usable recording(s)" + (f", {skipped} skipped (wrong shape)" if skipped else ""))

    if skipped_total:
        print(f"\nNOTE: {skipped_total} file(s) were skipped because they aren't ({SEQUENCE_LENGTH}, {TOTAL_FEATURES}). "
              "That happens with recordings made at a different SEQUENCE_LENGTH or the old single-frame 'legacy.npy'. "
              "Re-record those with: python data_collection.py --mode letters\n")

    for letter, seqs in jz.items():
        if len(seqs) < MIN_SEQUENCES_PER_LETTER:
            raise SystemExit(
                f"Need at least {MIN_SEQUENCES_PER_LETTER} recordings of '{letter}' (found {len(seqs)}). "
                "Record them with: python data_collection.py --mode letters  (press the letter key, "
                "and perform the motion while it says RECORDING)."
            )

    pool = load_static_pool()
    if pool:
        print(f"Building the OTHER class from {sum(len(v) for v in pool.values())} static-letter recordings "
              f"({len(pool)} letters).")
    else:
        print("WARNING: no static-letter recordings found, so there's no OTHER class -- ANY motion will be "
              "forced into J or Z. Record some static letters first for better results.")

    X_train, y_train_s, X_val, y_val_s, classes = build_datasets(jz, pool, rng)
    label_map = {i: c for i, c in enumerate(classes)}
    to_idx = {c: i for i, c in label_map.items()}
    y_train = np.array([to_idx[y] for y in y_train_s])
    y_val = np.array([to_idx[y] for y in y_val_s])

    print(f"\nClasses: {classes}")
    print(f"Training examples: {len(X_train)}  (per class: "
          f"{ {c: int((y_train == to_idx[c]).sum()) for c in classes} })")
    print(f"Validation examples: {len(X_val)}  (per class: "
          f"{ {c: int((y_val == to_idx[c]).sum()) for c in classes} })\n")

    model = build_lstm(num_classes=len(classes))
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=16,
        class_weight=balanced_class_weights(y_train, len(classes)),
        callbacks=[early_stop], verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    from sklearn.metrics import confusion_matrix, classification_report
    pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nConfusion matrix (rows = true, columns = predicted), order:", classes)
    print(confusion_matrix(y_val, pred, labels=list(range(len(classes)))))
    print(classification_report(y_val, pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))

    model.save(os.path.join(MODEL_DIR, "asl_lstm_jz.h5"))
    with open(os.path.join(MODEL_DIR, "jz_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print("Saved model to models/asl_lstm_jz.h5")
    print("Saved label map to models/jz_label_map.json")


if __name__ == "__main__":
    main()