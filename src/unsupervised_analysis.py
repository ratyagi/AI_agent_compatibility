"""
Unsupervised Analysis
purpose: to understanf the latent structure
goals:
  1. PCA on SBERT embeddings
  2. k-means clustering
  3. Outlier detection with IsolationForest
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

LABEL_NAMES = {
    0: "Hostile",
    1: "Neutral",
    2: "Friendly",
}

def run_pca(X_embed: np.ndarray,y_class: np.ndarray,figures_dir: Path):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_embed)

    var_ratios = pca.explained_variance_ratio_
    total_var_2d = var_ratios.sum()

    print(f"PC1 variance: {var_ratios[0]:.4f}")
    print(f"PC2 variance: {var_ratios[1]:.4f}")
    print(f"Total variance (PC1+PC2): {total_var_2d:.4f}")

    # prepare for plotting
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": y_class,
    })
    df_plot["label_name"] = df_plot["label"].map(LABEL_NAMES)

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="label_name",
        style="label_name",
        s=60,
    )
    plt.title("PCA of SBERT Embeddings (colored by true label)")
    plt.tight_layout()
    pca_fig_path = figures_dir / "pca_scatter.png"
    plt.savefig(pca_fig_path, dpi=200)
    plt.close()

    print(f"PCA scatter plot saved to: {pca_fig_path}")
    return X_pca, pca


def run_kmeans(X_pca: np.ndarray,y_class: np.ndarray,results_dir: Path,figures_dir: Path):

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for k in [3, 4, 5]:
        print(f"\nk-means with k = {k}")

        kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
        cluster_ids = kmeans.fit_predict(X_pca)

        df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "cluster": cluster_ids,
            "true_label": y_class,
        })
        df["true_label_name"] = df["true_label"].map(LABEL_NAMES)

        #cluster vs true label
        ctab = pd.crosstab(df["cluster"], df["true_label"])
        csv_path = results_dir / f"cluster_summary_k{k}.csv"
        ctab.to_csv(csv_path)

        print("Cluster vs true-label:")
        print(ctab)
        print(f"Saved contingency table to: {csv_path}")

        print("summary of clusters by main label:")
        for cluster_id in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == cluster_id]
            counts = subset["true_label"].value_counts()
            majority_label = counts.idxmax()
            majority_name = LABEL_NAMES.get(majority_label, str(majority_label))
            print(
                f"  - Cluster {cluster_id}: mostly {majority_name}, "
                f"label counts = {counts.to_dict()}"
            )

        # scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="PC1",
            y="PC2",
            hue="cluster",
            s=60,
        )
        plt.title(f"k-means Clusters in PCA Space (k = {k})")
        plt.tight_layout()
        fig_path = figures_dir / f"kmeans_k{k}_scatter.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"Cluster scatter plot saved to: {fig_path}")



def run_outlier_detection(X_embed: np.ndarray, y_class: np.ndarray, domains: pd.Series, results_dir: Path,top_n: int = 10):

    from sklearn.ensemble import IsolationForest
    results_dir.mkdir(parents=True, exist_ok=True)

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    iso.fit(X_embed)

    # decision_function, higher is normal
    decision_scores = iso.decision_function(X_embed)
    anomaly_scores = -decision_scores

    df_outliers = pd.DataFrame({
        "domain": domains,
        "score": anomaly_scores,
        "true_label": y_class,
    })
    df_outliers["true_label_name"] = df_outliers["true_label"].map(LABEL_NAMES)

    out_path = results_dir / "anomaly_scores.csv"
    df_outliers.to_csv(out_path, index=False)
    print(f"Anomaly scores saved to: {out_path}")

    #print top-N most anomalous
    df_sorted = df_outliers.sort_values("score", ascending=False)
    top_n = min(top_n, len(df_sorted))

    print(f"\nTop {top_n} most anomalous domains (higher score = more anomalous):")
    for _, row in df_sorted.head(top_n).iterrows():
        print(
            f"  - {row['domain']:25s} "
            f"score = {row['score']:.4f}, "
            f"label = {row['true_label_name']}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    figures_dir = project_root / "figures"
    results_dir = project_root / "results"

    print(f"{processed_dir}")
    print(f"{data_dir / 'domains.csv'}")

    X_embed = np.load(processed_dir / "X_embed.npy")
    y_class = np.load(processed_dir / "y_class.npy")
    domains_df = pd.read_csv(data_dir / "domains.csv")

    if "domain" not in domains_df.columns:
        raise ValueError("domains.csv must contain a 'domain' column.")

    domains = domains_df["domain"]

    # verifying
    n_embed = X_embed.shape[0]
    n_labels = y_class.shape[0]
    n_domains = len(domains)
    print(f"X_embed shape: {X_embed.shape}")
    print(f"y_class shape: {y_class.shape}")
    print(f"Number of domains: {n_domains}")

    if not (n_embed == n_labels == n_domains):
        raise ValueError(
            f"Mismatch in lengths: embeddings={n_embed}, labels={n_labels}, "
            f"domains={n_domains}. They should all match."
        )

    #PCA
    X_pca, _ = run_pca(X_embed, y_class, figures_dir)
    #k-means on PCA
    run_kmeans(X_pca, y_class, results_dir, figures_dir)
    #IsolationForest
    run_outlier_detection(X_embed, y_class, domains, results_dir, top_n=10)

if __name__ == "__main__":
    main()