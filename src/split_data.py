"""Train/Val/Test Split & Cross-Validation
This script:
- loads domains and labels
- creates and then saves split
- helper function for K-fold cross-validation
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

#train (70%), val (15%), test (15%)
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

def load_data():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"

    domains_path = data_dir / "domains.csv"
    y_class_path = processed_dir / "y_class.npy"
    y_score_path = processed_dir / "y_score.npy"

    domains = pd.read_csv(domains_path)["domain"].to_numpy()
    y_class = np.load(y_class_path)
    y_score = np.load(y_score_path)

    if not (len(domains) == len(y_class) == len(y_score)):
        raise ValueError("Mismatch in number of samples between files.")

    return domains, y_class, y_score


def create_train_val_test_splits(y_class):
    """Split dataset indices into train, val, test"""
    n = len(y_class)
    indices = np.arange(n)

    #1: train vs Temp (Val + Test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y_class,
        train_size=TRAIN_FRAC,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_class
    )

    #2:split Temp into Val and Test equally
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_temp
    )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def save_splits(train_idx, val_idx, test_idx, domains):
    """Saves split indices to .npz and .csv"""
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        processed_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    # Save csv
    split_labels = np.full(len(domains), "unassigned", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    pd.DataFrame({"domain": domains, "split": split_labels}).to_csv(
        processed_dir / "splits.csv", index=False
    )


def make_cv_folds(indices, y=None, n_splits=5, random_state=42, stratify=True):
    """returns a list of (cv_train_idx, cv_val_idx) pairs"""
    indices = np.asarray(indices)
    if y is not None and stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_sub = y[indices]
        splits = [(indices[tr], indices[val]) for tr, val in splitter.split(indices, y_sub)]
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = [(indices[tr], indices[val]) for tr, val in splitter.split(indices)]
    return splits


def main():
    print("[split_data] Loading data...")
    domains, y_class, y_score = load_data()
    print(f"[split_data] Loaded {len(domains)} domains.")

    print("[split_data] Creating Train/Val/Test splits...")
    train_idx, val_idx, test_idx = create_train_val_test_splits(y_class)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print("[split_data] Saving results...")
    save_splits(train_idx, val_idx, test_idx, domains)
    print("[split_data] Done.")

if __name__ == "__main__":
    main()