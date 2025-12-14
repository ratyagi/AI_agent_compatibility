from pathlib import Path
from typing import Dict, Any, List, Tuple
import ast

import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2]
LABEL_NAMES = ["hostile", "neutral", "friendly"]

#helper func
#get feature matrices, labels, splits, and summary CSVs from train_model.py
def load_data_and_model_results():
    X_tfidf = sp.load_npz(DATA_DIR / "X_tfidf.npz")
    X_embed = np.load(DATA_DIR / "X_embed.npy")
    try:
        X_numeric = np.load(DATA_DIR / "X_numeric.npy")
    except FileNotFoundError:
        X_numeric = np.zeros((X_embed.shape[0], 0), dtype=float)

    y_class = np.load(DATA_DIR / "y_class.npy")
    y_score = np.load(DATA_DIR / "y_score.npy")
    splits = np.load(DATA_DIR / "splits.npz")
    clf_results = pd.read_csv(DATA_DIR / "clf_results.csv")
    reg_results = pd.read_csv(DATA_DIR / "reg_results.csv")

    return X_tfidf, X_embed, X_numeric, y_class, y_score, splits, clf_results, reg_results

def make_split_views(X, splits):
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    if sp.issparse(X):
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]
    else:
        X_train = X[train_idx, :]
        X_val = X[val_idx, :]
        X_test = X[test_idx, :]

    return X_train, X_val, X_test

def combine_dense_features(X_embed, X_numeric):
    if X_numeric.size == 0:
        return X_embed
    return np.concatenate([X_embed, X_numeric], axis=1)

def make_non_negative(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        data = X.data
        data[data < 0] = 0.0
        return X
    X = np.array(X, copy=True)
    X[X < 0] = 0.0
    return X

def parse_best_params(s: str) -> Dict[str, Any]:
    """Parse best_params string from CSV into a dict."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


#rebuidlding classiers and regressors
def build_classifier(model_name: str, best_params: Dict[str, Any]):
    if model_name == "NB_TFIDF":
        base = {"alpha": 1.0}
        base.update(best_params)
        return MultinomialNB(**base), "tfidf"

    if model_name == "LR_TFIDF":
        base = {
            "multi_class": "multinomial",
            "solver": "lbfgs",
            "max_iter": 1000,
            "n_jobs": -1,
        }
        base.update(best_params)
        return LogisticRegression(**base), "tfidf"

    if model_name == "LinearSVC_TFIDF":
        base = {"max_iter": 5000}
        base.update(best_params)
        return LinearSVC(**base), "tfidf"

    if model_name == "RF_DENSE":
        base = {"random_state": 42}
        base.update(best_params)
        return RandomForestClassifier(**base), "dense"

    if model_name == "GB_DENSE":
        base = {"random_state": 42}
        base.update(best_params)
        return GradientBoostingClassifier(**base), "dense"

    if model_name == "MLP_DENSE":
        base = {"max_iter": 500, "random_state": 42}
        base.update(best_params)
        return MLPClassifier(**base), "dense"

    base = {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
        "n_jobs": -1,
    }
    base.update(best_params)
    return LogisticRegression(**base), "tfidf"


def build_regressor(model_name: str, best_params: Dict[str, Any]):
    if model_name == "Ridge_TFIDF":
        base = {"alpha": 1.0}
        base.update(best_params)
        return Ridge(**base), "tfidf"

    if model_name == "RFReg_DENSE":
        base = {"random_state": 42}
        base.update(best_params)
        return RandomForestRegressor(**base), "dense"

    if model_name == "MLPReg_DENSE":
        base = {"max_iter": 500, "random_state": 42}
        base.update(best_params)
        return MLPRegressor(**base), "dense"

    # Fallback
    base = {"alpha": 1.0}
    base.update(best_params)
    return Ridge(**base), "tfidf"


# re-stating results from train_models.py
def summarize_phase5_results(clf_results: pd.DataFrame, reg_results: pd.DataFrame):

    print("\nClassification model summary (from model training)")
    clf_view = clf_results[
        ["model", "train_acc", "train_f1", "val_acc", "val_f1", "test_acc", "test_f1"]
    ].sort_values("val_f1", ascending=False)
    print(clf_view.to_string(index=False))

    print("\nRegression model summary (from model training)")
    reg_view = reg_results[
        ["model", "train_mae", "train_rmse", "val_mae", "val_rmse", "test_mae", "test_rmse"]
    ].sort_values("val_rmse", ascending=True)
    print(reg_view.to_string(index=False))


#further 
def _save_confusion_matrix(y_true, y_pred, model_name: str, split: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} – {split}")
    fig.tight_layout()
    out_path = FIGURES_DIR / f"confusion_{model_name}_{split}.png"
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_classifiers(X_tfidf, X_embed,X_numeric,y_class, splits, clf_results: pd.DataFrame) -> pd.DataFrame:
    """
    Refit best classifiers on the train split and compute:precision/recall/F1
    avg PR f1 and confusion matrices (val & test)
    """

    Xtf_train, Xtf_val, Xtf_test = make_split_views(X_tfidf, splits)
    X_dense = combine_dense_features(X_embed, X_numeric)
    Xd_train, Xd_val, Xd_test = make_split_views(X_dense, splits)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    y_train = y_class[train_idx]
    y_val = y_class[val_idx]
    y_test = y_class[test_idx]

    rows: List[Dict[str, Any]] = []

    for _, row in clf_results.iterrows():
        model_name = row["model"]
        params = parse_best_params(row["best_params"])
        clf, feature_type = build_classifier(model_name, params)

        if feature_type == "tfidf":
            X_train = Xtf_train
            X_val = Xtf_val
            X_test = Xtf_test
            if isinstance(clf, MultinomialNB):
                X_train = make_non_negative(X_train)
                X_val = make_non_negative(X_val)
                X_test = make_non_negative(X_test)
        else:
            X_train = Xd_train
            X_val = Xd_val
            X_test = Xd_test

        clf.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            y_pred = clf.predict(X_split)
            acc = accuracy_score(y_split, y_pred)
            prec, rec, f1, support = precision_recall_fscore_support(
                y_split,
                y_pred,
                labels=LABELS,
                zero_division=0,
            )

            avg_prec = prec.mean()
            avg_rec = rec.mean()
            avg_f1 = f1.mean()

            result: Dict[str, Any] = {
                "model": model_name,
                "split": split_name,
                "accuracy": acc,
                "macro_precision": avg_prec,
                "macro_recall": avg_rec,
                "macro_f1": avg_f1,
            }

            for lab, name, p, r, f, sup in zip(LABELS, LABEL_NAMES, prec, rec, f1, support):
                result[f"precision_{name}"] = p
                result[f"recall_{name}"] = r
                result[f"f1_{name}"] = f
                result[f"support_{name}"] = int(sup)

            rows.append(result)
            if split_name in ("val", "test"):
                _save_confusion_matrix(y_split, y_pred, model_name, split_name)

    metrics_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "classification_metrics.csv"
    metrics_df.to_csv(out_path, index=False)

    summary = metrics_df[metrics_df["split"].isin(["val", "test"])][
        ["model", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ].sort_values(["split", "accuracy"], ascending=[True, False])

    print("\ndetailed classification metrics (val/test):")
    print(summary.to_string(index=False))
    return metrics_df

#copying results
def export_regression_metrics(reg_results: pd.DataFrame) -> None:
    out_path = RESULTS_DIR / "regression_metrics.csv"
    reg_results.to_csv(out_path, index=False)
    print("\nregression metrics copied from reg_results.csv to regression_metrics.csv")


#Learning curve for best classifier
def _pick_best_classifier(clf_results: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    best_idx = clf_results["val_f1"].idxmax()
    best_row = clf_results.loc[best_idx]
    model_name = best_row["model"]
    params = parse_best_params(best_row["best_params"])
    print(f"\nBest classifier (by val_f1): {model_name}")
    print(best_row.to_string())
    return model_name, params


def plot_learning_curve(
    model_name: str,
    best_params: Dict[str, Any],
    X_tfidf,
    X_embed,
    X_numeric,
    y_class,
    splits,
    fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
):
    #train and validation accuracy vs fraction of training data.
    Xtf_train, Xtf_val, _ = make_split_views(X_tfidf, splits)
    X_dense = combine_dense_features(X_embed, X_numeric)
    Xd_train, Xd_val, _ = make_split_views(X_dense, splits)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

    y_train_full = y_class[train_idx]
    y_val = y_class[val_idx]
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(len(train_idx))

    frac_used: List[float] = []
    train_scores: List[float] = []
    val_scores: List[float] = []

    for frac in fractions:
        n_sub = max(1, int(round(frac * len(train_idx))))
        idx_sub = shuffled[:n_sub]

        clf, feature_type = build_classifier(model_name, best_params)

        if feature_type == "tfidf":
            X_train_full = Xtf_train
            X_val = Xtf_val
            if isinstance(clf, MultinomialNB):
                X_train_full = make_non_negative(X_train_full)
                X_val = make_non_negative(X_val)
        else:
            X_train_full = Xd_train
            X_val = Xd_val

        X_train_sub = X_train_full[idx_sub]
        y_train_sub = y_train_full[idx_sub]

        clf.fit(X_train_sub, y_train_sub)

        y_train_pred = clf.predict(X_train_sub)
        y_val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train_sub, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        frac_used.append(frac * 100.0)
        train_scores.append(train_acc)
        val_scores.append(val_acc)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(frac_used, train_scores, marker="o", label="Train accuracy")
    ax.plot(frac_used, val_scores, marker="o", label="Validation accuracy")
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve – {model_name}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    out_path = FIGURES_DIR / "learning_curve_best_classifier.png"
    fig.savefig(out_path)
    plt.close(fig)

    print("\n=== Phase 7: learning curve (best classifier) ===")
    for f, tr, va in zip(frac_used, train_scores, val_scores):
        print(f"{f:5.1f}% data | train_acc={tr:.3f}, val_acc={va:.3f}")

    if train_scores[-1] - val_scores[-1] > 0.05:
        print("Comment: train accuracy noticeably higher than val accuracy → overfitting / high variance.")
    elif val_scores[-1] > train_scores[-1]:
        print("Comment: validation slightly higher than train accuracy → strong regularization / noise.")
    else:
        print("Comment: train and val accuracies close → lower variance, likely bias-limited.")


def main():
    (
        X_tfidf,
        X_embed,
        X_numeric,
        y_class,
        y_score,
        splits,
        clf_results,
        reg_results,
    ) = load_data_and_model_results()

    #summary
    summarize_phase5_results(clf_results, reg_results)

    #deeper eval
    _ = evaluate_classifiers(
        X_tfidf=X_tfidf,
        X_embed=X_embed,
        X_numeric=X_numeric,
        y_class=y_class,
        splits=splits,
        clf_results=clf_results,
    )

    export_regression_metrics(reg_results)
    #learning curve
    best_model_name, best_params = _pick_best_classifier(clf_results)
    plot_learning_curve(
        model_name=best_model_name,
        best_params=best_params,
        X_tfidf=X_tfidf,
        X_embed=X_embed,
        X_numeric=X_numeric,
        y_class=y_class,
        splits=splits,
    )

if __name__ == "__main__":
    main()