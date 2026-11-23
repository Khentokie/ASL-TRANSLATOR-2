import os
import pandas as pd
import numpy as np

from utils import TOTAL_FEATURES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "letters")


def main():
    if not os.path.isdir(DATA_DIR):
        print("No data/letters directory found -- nothing to migrate.")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No legacy .csv files found -- nothing to migrate.")
        return

    migrated = 0
    for fname in sorted(csv_files):
        letter = os.path.splitext(fname)[0]
        csv_path = os.path.join(DATA_DIR, fname)

        df = pd.read_csv(csv_path, header=None)
        if df.empty:
            continue

        features = df.iloc[:, :TOTAL_FEATURES].values.astype(np.float32)

        out_dir = os.path.join(DATA_DIR, letter)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "legacy.npy")
        np.save(out_path, features)

        print(f"Migrated {len(features)} samples for '{letter}' -> {out_path}")
        migrated += len(features)

    print(f"\nDone. Migrated {migrated} total samples.")
    print("Your original .csv files were left untouched (not deleted).")


if __name__ == "__main__":
    main()


"""
migrate_legacy_letters.py
One-time helper: converts old data/letters/<LETTER>.csv samples (from the
previous single-frame capture workflow) into the new folder format
data/letters/<LETTER>/legacy.npy, so train_lstm_letters.py can still use
them alongside newly-recorded motion sequences.

Safe to run multiple times -- it just overwrites legacy.npy each time
from the current CSV contents. Run once after updating to the new
data_collection.py, before your next training run. (Each legacy single frame is
treated as a held-still sequence by the letters LSTM.)

Run:  python migrate_legacy_letters.py
"""