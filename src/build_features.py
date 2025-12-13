"""
Goal: Feature Construction
Output file:
    X_tfidf.npz: TF-IDF TOS + numeric
    X_embed.npy: SBERT TOS + numeric
    X_numeric.npy: numeric-only (robots.txt + HTML)
    y_class.npy: friendly_label
    y_score.npy: compat_score
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# to read data properly.
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_paths():
    root = get_project_root()
    data = root / "data"
    raw = data / "raw"
    processed = data / "processed"

    return {
        "root": root,
        "data": data,
        "raw": raw,
        "processed": processed,
        "domains": data / "domains.csv",
        "domains_normalized": data / "domains_normalized.csv",
        "robots_dir": raw / "robots",
        "tos_dir": raw / "tos_texts",
        "html_stats": processed / "html_stats.csv",
        "labels": processed / "labels.csv",
    }

#load data
def load_domain_list(paths) -> pd.DataFrame:
    if paths["domains_normalized"].exists():
        df = pd.read_csv(paths["domains_normalized"])
    elif paths["domains"].exists():
        df = pd.read_csv(paths["domains"])
    else:
        raise FileNotFoundError("No domains_normalized.csv or domains.csv in data/")

    if "domain" not in df.columns:
        raise ValueError("Domain file must have a 'domain' column.")

    df = df.drop_duplicates(subset=["domain"]).reset_index(drop=True)
    return df


def load_labels(paths) -> pd.DataFrame:
    labels_path = paths["labels"]
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")

    df = pd.read_csv(labels_path)
    for col in ["domain", "friendly_label", "compat_score"]:
        if col not in df.columns:
            raise ValueError(f"labels.csv missing required column: {col}")

    if df["domain"].duplicated().any():
        raise ValueError("labels.csv has duplicate domains.")

    return df


def load_html_stats(paths) -> pd.DataFrame:
    html_path = paths["html_stats"]
    if not html_path.exists():
        raise FileNotFoundError(f"html_stats.csv not found at {html_path}")

    df = pd.read_csv(html_path)
    required = [
        "domain",
        "status_code",
        "response_time",
        "content_length",
        "num_scripts",
        "num_links",
        "num_forms",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"html_stats.csv missing column: {col}")

    df = df.drop_duplicates(subset=["domain"]).reset_index(drop=True)
    return df


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")


def load_tos_texts(domains, tos_dir: Path):
    texts = []
    for d in domains:
        p = tos_dir / f"{d}.txt"
        texts.append(read_text_file(p))
    return texts


#robots.txt features
MENTIONS_AI_OR_BOT_PATTERN = re.compile(
    r"\b(bot|bots|crawler|crawl|scrape|scraping|automated|automation|ai|"
    r"machine learning|ml)\b",
    flags=re.IGNORECASE,
)


def extract_robots_features(text: str):
    if not text:
        return 0, 0, 0, 0, 0, 0

    num_disallow = 0
    num_allow = 0
    has_disallow_root = 0
    has_user_agent_star = 0
    crawl_delay_present = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        if "disallow:" in lower:
            num_disallow += 1
            # check if disallow root "/"
            core = line.split("#", 1)[0]
            m = re.search(r"disallow\s*:\s*(.*)", core, flags=re.IGNORECASE)
            if m:
                path_value = m.group(1).strip()
                if path_value == "/" or path_value == "/*":
                    has_disallow_root = 1

        if "allow:" in lower:
            num_allow += 1

        if lower.startswith("user-agent:") and "*" in lower:
            has_user_agent_star = 1

        if "crawl-delay" in lower:
            crawl_delay_present = 1

    mentions_ai_or_bot = 1 if MENTIONS_AI_OR_BOT_PATTERN.search(text) else 0

    return (
        num_disallow,
        num_allow,
        has_disallow_root,
        has_user_agent_star,
        crawl_delay_present,
        mentions_ai_or_bot,
    )


def build_robots_features(domains, robots_dir: Path) -> pd.DataFrame:
    rows = []
    for d in domains:
        p = robots_dir / f"{d}.txt"
        text = read_text_file(p)
        (
            num_disallow,
            num_allow,
            has_disallow_root,
            has_user_agent_star,
            crawl_delay_present,
            mentions_ai_or_bot,
        ) = extract_robots_features(text)

        rows.append(
            {
                "domain": d,
                "num_disallow": num_disallow,
                "num_allow": num_allow,
                "has_disallow_root": has_disallow_root,
                "has_user_agent_star": has_user_agent_star,
                "crawl_delay_present": crawl_delay_present,
                "mentions_ai_or_bot": mentions_ai_or_bot,
            }
        )

    return pd.DataFrame(rows)



#TF-IDF
def build_tfidf(tos_texts):
    """using TOS, output sparse matrix."""
    print("TF-IDF on TOS texts.")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        stop_words="english",
    )
    X = vectorizer.fit_transform(tos_texts)
    print(f"TF-IDF shape: {X.shape}")
    return X

#sbert
def build_sbert_embeddings(tos_texts, model_name="all-MiniLM-L6-v2"):
    """Return SBERT embeddings (N, 384) from TOS texts."""
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    # For 150 documents, we can just call encode() once with progress bar.
    print("Encoding TOS texts with SBERT...")
    X = model.encode(
        tos_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    X = X.astype(np.float32)
    print(f"SBERT embedding shape: {X.shape}")
    return X



def build_numeric_features(df_domains, df_robots, df_html):
    """create robots.txt + html, and concatenated raw numeric matrix."""
    domains = df_domains["domain"]

    # robots in canonical order
    robots_cols = [
        "num_disallow",
        "num_allow",
        "has_disallow_root",
        "has_user_agent_star",
        "crawl_delay_present",
        "mentions_ai_or_bot",
    ]
    df_r = df_robots.set_index("domain").reindex(domains)
    df_r = df_r[robots_cols].fillna(0.0)
    X_robots = df_r.to_numpy(dtype=np.float64)

    html_cols = [
        "status_code",
        "response_time",
        "content_length",
        "num_scripts",
        "num_links",
        "num_forms",
    ]
    df_h = df_html.set_index("domain").reindex(domains)
    df_h = df_h[html_cols].fillna(0.0)
    X_html = df_h.to_numpy(dtype=np.float64)

    X_numeric_raw = np.hstack([X_robots, X_html])
    return X_numeric_raw


def build_labels_for_domains(df_domains, df_labels):
    """return y_class, y_score aligned to domain order."""
    domains = df_domains["domain"]
    df_l = df_labels.set_index("domain").reindex(domains)

    if df_l["friendly_label"].isna().any() or df_l["compat_score"].isna().any():
        missing = df_l[df_l["friendly_label"].isna() | df_l["compat_score"].isna()]
        raise ValueError(
            f"Labels missing for domains: {missing.index.tolist()}"
        )

    y_class = df_l["friendly_label"].to_numpy(dtype=np.int64)
    y_score = df_l["compat_score"].to_numpy(dtype=np.float32)
    return y_class, y_score


def main():
    paths = get_paths()
    processed_dir = paths["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    #Load domain list, labels, html stats
    df_domains = load_domain_list(paths)
    domains = df_domains["domain"].tolist()
    N = len(domains)
    print(f"Number of domains: {N}")

    df_labels = load_labels(paths)
    df_html = load_html_stats(paths)
    tos_texts = load_tos_texts(domains, paths["tos_dir"])

    # build robots.txt features
    print("robots.txt features.")
    df_robots = build_robots_features(domains, paths["robots_dir"])

    #robots + HTML features
    print("robots.txt + HTML - numeric features.")
    X_numeric_raw = build_numeric_features(df_domains, df_robots, df_html)
    print(f"X_numeric_raw shape: {X_numeric_raw.shape}")

    #tf-idf and sbert
    X_tfidf_text = build_tfidf(tos_texts)
    X_embed_text = build_sbert_embeddings(tos_texts)

    #labels
    print("aligning labels.")
    y_class, y_score = build_labels_for_domains(df_domains, df_labels)
    print(f"y_class shape: {y_class.shape}, y_score shape: {y_score.shape}")

    #standardizing features
    print("standardizing numeric features.")
    X_numeric_raw = np.nan_to_num(X_numeric_raw, copy=False)
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X_numeric_raw).astype(np.float32)
    print(f"X_numeric (scaled) shape: {X_numeric.shape}")

    #combine TF-IDF + numeric (sparse)
    print("Combining TF-IDF text with numeric features.")
    X_numeric_sparse = sparse.csr_matrix(X_numeric.astype(np.float64))
    X_tfidf = sparse.hstack([X_tfidf_text, X_numeric_sparse]).tocsr()
    print(f"Final X_tfidf shape: {X_tfidf.shape}")

    #combine SBERT + numeric (dense)
    print("Combining SBERT embeddings with numeric features.")
    if X_embed_text.shape[0] != X_numeric.shape[0]:
        raise ValueError("Row count mismatch between embeddings and numeric features.")
    X_embed = np.hstack([X_embed_text, X_numeric]).astype(np.float32)
    print(f"Final X_embed shape: {X_embed.shape}")

    #save features
    print("Saving feature to data/processed/...")

    sparse.save_npz(processed_dir / "X_tfidf.npz", X_tfidf)
    np.save(processed_dir / "X_embed.npy", X_embed)
    np.save(processed_dir / "X_numeric.npy", X_numeric)
    np.save(processed_dir / "y_class.npy", y_class)
    np.save(processed_dir / "y_score.npy", y_score)

if __name__ == "__main__":
    main()