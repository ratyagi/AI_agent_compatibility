"""
1.Classification task: predict friendly_label (y_class)
2.Regression task:predict compat_score (y_score)
Features:X_tfidf, X_embed, X_numeric:
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

def make_non_negative(X):
    """Ensure all entries are >= 0 (got error with MultinomialNB)."""
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        data = X.data
        data[data < 0] = 0.0
        return X
    else:
        X = np.array(X, copy=True)
        X[X < 0] = 0.0
        return X

def make_stratified_cv(y, max_splits=5):
    """Choose a valid number for fied folds based on the smallest class size. (min 2)"""
    _, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    n_splits = max(2, min(max_splits, int(min_class_count)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)


DATA_DIR = Path("data/processed")

def load_features():
    X_tfidf = sp.load_npz(DATA_DIR / "X_tfidf.npz")
    X_embed = np.load(DATA_DIR / "X_embed.npy")
    X_numeric = np.load(DATA_DIR / "X_numeric.npy")
    y_class = np.load(DATA_DIR / "y_class.npy")
    y_score = np.load(DATA_DIR / "y_score.npy")
    splits = np.load(DATA_DIR / "splits.npz")
    return X_tfidf, X_embed, X_numeric, y_class, y_score, splits


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
    """tree-based models and MLPs work better and faster
     on a moderate-size dense representation.
    """
    return np.concatenate([X_embed, X_numeric], axis=1)


## 1. Problem Type: Classification (friendly_label)

def run_classification_experiment(
    name: str,
    model,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    # to handles class imbalance
    cv = make_stratified_cv(y_train, max_splits=5)
    # macro-F1 focuses on all classes, not just majority
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_

    def eval_split(X, y):
        y_pred = best.predict(X)
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        return acc, f1_macro

    train_acc, train_f1 = eval_split(X_train, y_train)
    val_acc, val_f1 = eval_split(X_val, y_val)
    test_acc, test_f1 = eval_split(X_test, y_test)

    result = {
        "model": name,
        "best_params": grid.best_params_,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }

    print(f"\n==== {name} ====")
    print("Best params:", grid.best_params_)
    print(
        f"Train  acc={train_acc:.3f}, F1={train_f1:.3f} | "
        f"Val acc={val_acc:.3f}, F1={val_f1:.3f} | "
        f"Test acc={test_acc:.3f}, F1={test_f1:.3f}"
    )

    return result


def run_all_classifiers(X_tfidf, X_embed, X_numeric, y_class, splits):
    """
    NB / LR / LinearSVC use TF-IDF:
      - These linear models handle high-dim sparse data well.
    RF / GB / MLP use (embed + numeric):
      - Trees and MLPs are better on dense, lower-dim features.
    """
    results = []

    #split TF-IDF
    Xtf_train, Xtf_val, Xtf_test = make_split_views(X_tfidf, splits)

    #split dense representation (embeddings + numeric)
    X_dense = combine_dense_features(X_embed, X_numeric)
    Xd_train, Xd_val, Xd_test = make_split_views(X_dense, splits)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    y_train = y_class[train_idx]
    y_val = y_class[val_idx]
    y_test = y_class[test_idx]

    # Model 1: Multinomial Naive Bayes (TF-IDF only)
    # alpha ∈ {0.1, 0.5, 1.0}
    nb = MultinomialNB()
    nb_grid = {"alpha": [0.1, 0.5, 1.0]}
    Xtf_train_nb = make_non_negative(Xtf_train)
    Xtf_val_nb = make_non_negative(Xtf_val)
    Xtf_test_nb = make_non_negative(Xtf_test)
    results.append(
        run_classification_experiment(
            "NB_TFIDF",
            nb,
            nb_grid,
            Xtf_train_nb,
            y_train,
            Xtf_val_nb,
            y_val,
            Xtf_test_nb,
            y_test,
        )
    )

    # Model 2: Logistic Regression (multinomial)
    # C ∈ {0.01, 0.1, 1.0, 10.0}:
    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )
    lr_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    results.append(
        run_classification_experiment(
            "LR_TFIDF",
            lr,
            lr_grid,
            Xtf_train,
            y_train,
            Xtf_val,
            y_val,
            Xtf_test,
            y_test,
        )
    )

    # Model 3: Linear SVM (hinge loss)
    # Same C rationale as LR.
    svm = LinearSVC(max_iter=5000)
    svm_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    results.append(
        run_classification_experiment(
            "LinearSVC_TFIDF",
            svm,
            svm_grid,
            Xtf_train,
            y_train,
            Xtf_val,
            y_val,
            Xtf_test,
            y_test,
        )
    )

    #Model 4: Random Forest (dense)
    # n_estimators ∈ {100, 300}:
    # max_depth ∈ {None, 5, 10}:
    #  None -> fully grown trees (low bias, high variance).
    #  5, 10 -> shallower trees (more bias, less variance) for small data.
    rf = RandomForestClassifier(random_state=0)
    rf_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
    }
    results.append(
        run_classification_experiment(
            "RF_DENSE",
            rf,
            rf_grid,
            Xd_train,
            y_train,
            Xd_val,
            y_val,
            Xd_test,
            y_test,
        )
    )

    # Model 5: Gradient Boosting (dense)
    # n_estimators ∈ {50, 100}:
    # learning_rate ∈ {0.05, 0.1}:
    gb = GradientBoostingClassifier(random_state=0)
    gb_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3],  # depth 3 = standard for boosted trees
    }
    results.append(
        run_classification_experiment(
            "GB_DENSE",
            gb,
            gb_grid,
            Xd_train,
            y_train,
            Xd_val,
            y_val,
            Xd_test,
            y_test,
        )
    )

    # Model 6: MLP Classifier (dense)
    # hidden_layer_sizes:
    # 64,-> small model
    # 128,-> medium capacity
    # 128, 64-> deeper network with more power
    # alpha ∈ {1e-4, 1e-3}:L2 regularization strength
    mlp = MLPClassifier(
        max_iter=500,
        random_state=0,
    )
    mlp_grid = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "alpha": [1e-4, 1e-3],
    }
    results.append(
        run_classification_experiment(
            "MLP_DENSE",
            mlp,
            mlp_grid,
            Xd_train,
            y_train,
            Xd_val,
            y_val,
            Xd_test,
            y_test,
        )
    )

    return pd.DataFrame(results)



## Problem Type: Regression (compat_score)

def run_regression_experiment(
    name: str,
    model,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_

    def eval_split(X, y):
        y_pred = best.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return mae, rmse

    train_mae, train_rmse = eval_split(X_train, y_train)
    val_mae, val_rmse = eval_split(X_val, y_val)
    test_mae, test_rmse = eval_split(X_test, y_test)

    result = {
        "model": name,
        "best_params": grid.best_params_,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }

    print(f"\n==== {name} (regression) ====")
    print("Best params:", grid.best_params_)
    print(
        f"Train  MAE={train_mae:.3f}, RMSE={train_rmse:.3f} | "
        f"Val MAE={val_mae:.3f}, RMSE={val_rmse:.3f} | "
        f"Test MAE={test_mae:.3f}, RMSE={test_rmse:.3f}"
    )

    return result


def run_all_regressors(X_tfidf, X_embed, X_numeric, y_score, splits):
    """Train regression models to predict compat_score."""
    results = []

    #tf-idf splits (sparse)
    Xtf_train, Xtf_val, Xtf_test = make_split_views(X_tfidf, splits)

    # Dense splits
    X_dense = combine_dense_features(X_embed, X_numeric)
    Xd_train, Xd_val, Xd_test = make_split_views(X_dense, splits)

    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    y_train = y_score[train_idx]
    y_val = y_score[val_idx]
    y_test = y_score[test_idx]

    # Model 1: Ridge Regression (TF-IDF)
    # alpha ∈ {0.01, 0.1, 1.0, 10.0}: L2 regularization strength.
    ridge = Ridge()
    ridge_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
    results.append(
        run_regression_experiment(
            "Ridge_TFIDF",
            ridge,
            ridge_grid,
            Xtf_train,
            y_train,
            Xtf_val,
            y_val,
            Xtf_test,
            y_test,
        )
    )

    # Model 2: Random Forest Regressor (dense) 
    rf_reg = RandomForestRegressor(random_state=0)
    rf_reg_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
    }
    results.append(
        run_regression_experiment(
            "RFReg_DENSE",
            rf_reg,
            rf_reg_grid,
            Xd_train,
            y_train,
            Xd_val,
            y_val,
            Xd_test,
            y_test,
        )
    )

    # Model 3: MLP Regressor (dense)
    mlp_reg = MLPRegressor(
        max_iter=500,
        random_state=0,
    )
    mlp_reg_grid = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "alpha": [1e-4, 1e-3],
    }
    results.append(
        run_regression_experiment(
            "MLPReg_DENSE",
            mlp_reg,
            mlp_reg_grid,
            Xd_train,
            y_train,
            Xd_val,
            y_val,
            Xd_test,
            y_test,
        )
    )

    return pd.DataFrame(results)


def main():
    X_tfidf, X_embed, X_numeric, y_class, y_score, splits = load_features()

    print("Running classification experiments...")
    clf_results = run_all_classifiers(X_tfidf, X_embed, X_numeric, y_class, splits)
    clf_path = DATA_DIR / "clf_results.csv"
    clf_results.to_csv(clf_path, index=False)
    print(f"\nSaved classification results to {clf_path}")

    print("\nRunning regression experiments...")
    reg_results = run_all_regressors(X_tfidf, X_embed, X_numeric, y_score, splits)
    reg_path = DATA_DIR / "reg_results.csv"
    reg_results.to_csv(reg_path, index=False)
    print(f"\nSaved regression results to {reg_path}")

    print("\ncomplete.")

if __name__ == "__main__":
    main()
