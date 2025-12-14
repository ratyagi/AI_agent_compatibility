"""
Semi-automatic labeling pipeline with data quality tracking.
"""

import argparse
import os
from typing import Dict, Any, Optional, List

import pandas as pd
from tqdm import tqdm

from . import labeling_utils


def load_html_stats(path: str) -> Dict[str, Dict[str, Any]]:
    """Load html_stats.csv into a dict by domain."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "domain" not in df.columns:
        raise ValueError(f"html_stats file at {path} has no 'domain' column")
    df = df.drop_duplicates(subset=["domain"], keep="last")
    html_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        domain = str(row["domain"])
        html_dict[domain] = row.to_dict()
    return html_dict


def load_log_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a log CSV the same way."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "domain" not in df.columns:
        raise ValueError(f"log file at {path} has no 'domain' column")
    df = df.drop_duplicates(subset=["domain"], keep="last")
    log_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        domain = str(row["domain"])
        log_dict[domain] = row.to_dict()
    return log_dict


def read_text_file(path: str) -> str:
    """Handles missing data- v imp. Return file contents or empty string if file does not exist."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def compute_data_quality(
    domain: str,
    tos_text: str,
    tos_log: Optional[Dict[str, Any]],
    robots_text: str,
    robots_log: Optional[Dict[str, Any]],
    html_info: Optional[Dict[str, Any]],
) -> str:
    """report missing entry"""
    flags: List[str] = []

    # TOS
    if not tos_text.strip():
        flags.append("missing_tos_file")
    if tos_log is None:
        flags.append("missing_tos_log")
    else:
        status = str(tos_log.get("status", "unknown"))
        http_status = tos_log.get("http_status")
        if status != "success":
            flags.append(f"tos_status_{status}")
        if not pd.isna(http_status):
            try:
                hs = int(float(http_status))
                if hs >= 400:
                    flags.append(f"tos_http_{hs}")
                elif hs >= 300:
                    flags.append(f"tos_http_{hs}")
            except Exception:
                flags.append("tos_http_unparseable")
        else:
            flags.append("tos_http_missing")

    # Tiny TOS
    if tos_text and len(tos_text) < 500:
        flags.append("tiny_tos")

    # robots
    if not robots_text.strip():
        flags.append("missing_robots_file")
    if robots_log is None:
        flags.append("missing_robots_log")
    else:
        status = str(robots_log.get("status", "unknown"))
        http_status = robots_log.get("http_status")
        if status != "success":
            flags.append(f"robots_status_{status}")
        if not pd.isna(http_status):
            try:
                hs = int(float(http_status))
                if hs >= 400:
                    flags.append(f"robots_http_{hs}")
                elif hs >= 300:
                    flags.append(f"robots_http_{hs}")
            except Exception:
                flags.append("robots_http_unparseable")
        else:
            flags.append("robots_http_missing")

    # HTML
    if html_info is None:
        flags.append("missing_html_stats")
    else:
        status_code = html_info.get("status_code")
        if status_code is None:
            flags.append("html_status_missing")
        else:
            try:
                sc = int(status_code)
                if sc >= 400:
                    flags.append(f"html_status_{sc}")
            except Exception:
                flags.append("html_status_unparseable")

    if not flags:
        return "complete"
    else:
        return ",".join(sorted(set(flags)))


def main():
    parser = argparse.ArgumentParser(description="Automatic labeling for AI agent compatibility with data quality tracking.")
    parser.add_argument(
        "--domains-csv",
        type=str,
        default="data/domains_normalized.csv",
        help="CSV file containing at least a 'domain' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/labels.csv",
        help="Path to output labels CSV.",
    )
    parser.add_argument(
        "--html-stats",
        type=str,
        default="data/processed/html_stats.csv",
        help="Path to html_stats.csv.",
    )
    parser.add_argument(
        "--tos-log",
        type=str,
        default="data/processed/tos_log.csv",
        help="Path to tos_log.csv.",
    )
    parser.add_argument(
        "--robots-log",
        type=str,
        default="data/processed/robots_log.csv",
        help="Path to robots_log.csv.",
    )
    parser.add_argument(
        "--tos-dir",
        type=str,
        default="data/raw/tos_texts",
        help="Directory with TOS text files {domain}.txt",
    )
    parser.add_argument(
        "--robots-dir",
        type=str,
        default="data/raw/robots",
        help="Directory with robots.txt files {domain}.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )

    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise SystemExit(
            f"Output file {args.output} already exists. "
            f"Use --overwrite to replace it."
        )

    if not os.path.exists(args.domains_csv):
        raise SystemExit(f"Domains CSV not found: {args.domains_csv}")

    domains_df = pd.read_csv(args.domains_csv)
    if "domain" not in domains_df.columns:
        raise SystemExit("Domains CSV must contain a 'domain' column.")

    html_stats = load_html_stats(args.html_stats)
    tos_log_dict = load_log_csv(args.tos_log)
    robots_log_dict = load_log_csv(args.robots_log)

    rows = []
    for _, row in tqdm(domains_df.iterrows(), total=len(domains_df), desc="Auto-labeling domains"):
        domain = str(row["domain"])

        tos_path = os.path.join(args.tos_dir, f"{domain}.txt")
        robots_path = os.path.join(args.robots_dir, f"{domain}.txt")

        tos_text = read_text_file(tos_path)
        robots_text = read_text_file(robots_path)
        html_info: Optional[Dict[str, Any]] = html_stats.get(domain)

        tos_log = tos_log_dict.get(domain)
        robots_log = robots_log_dict.get(domain)

        data_quality = compute_data_quality(
            domain=domain,
            tos_text=tos_text,
            tos_log=tos_log,
            robots_text=robots_text,
            robots_log=robots_log,
            html_info=html_info,
        )

        label_row = labeling_utils.auto_label_domain(
            domain=domain,
            tos_text=tos_text,
            robots_text=robots_text,
            html_info=html_info,
        )

        # If everything is essentially missing, force a safe neutral baseline
        all_missing = (
            "missing_tos_file" in data_quality
            and "missing_robots_file" in data_quality
            and "missing_html_stats" in data_quality
        )
        if all_missing:
            label_row["friendly_label"] = 1
            label_row["scraping_perm"] = 1
            label_row["ai_training_perm"] = 1
            label_row["derivative_perm"] = 1
            label_row["compat_score"] = 0.5
            if label_row["notes"] == "auto-labeled":
                label_row["notes"] = "all_sources_missing_default_neutral"
            else:
                label_row["notes"] += "; all_sources_missing_default_neutral"

        label_row["data_quality"] = data_quality
        rows.append(label_row)

    labels_df = pd.DataFrame(rows, columns=[
        "domain",
        "friendly_label",
        "compat_score",
        "scraping_perm",
        "ai_training_perm",
        "derivative_perm",
        "notes",
        "data_quality",
    ])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    labels_df.to_csv(args.output, index=False)

    # quick summary
    print("\n=== Auto-labeling Summary ===")
    print("friendly_label value counts:")
    print(labels_df["friendly_label"].value_counts(dropna=False).sort_index())
    print("\ncompat_score summary:")
    print(labels_df["compat_score"].describe())
    print("\ndata_quality examples:")
    print(labels_df["data_quality"].value_counts().head(10))
    print(f"\nSaved labels to {args.output}")


if __name__ == "__main__":
    main()