"""
collect_domains.py
Validate and optionally normalize the domain list in data/domains.csv.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple


DOMAIN_REGEX = re.compile(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")


def normalize_domain(raw: str) -> str:
    """
    Normalize a domain string:
    - strip whitespace
    - remove protocol (http://, https://)
    - remove leading 'www.'
    - remove path and query components
    """
    if raw is None:
        return ""

    d = raw.strip()

    if d.startswith("http://"):
        d = d[len("http://") :]
    elif d.startswith("https://"):
        d = d[len("https://") :]

    # Split off path/query/fragment
    for sep in ["/", "?", "#"]:
        if sep in d:
            d = d.split(sep, 1)[0]

    # Remove leading www.
    if d.startswith("www."):
        d = d[4:]

    return d.strip()


def is_valid_domain(domain: str) -> bool:
    """Return True if the domain string looks like a valid domain."""
    if not domain:
        return False
    return DOMAIN_REGEX.match(domain) is not None


def read_domains_csv(path: Path) -> List[dict]:
    """Read the domains CSV into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"domains.csv not found at {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if "domain" not in reader.fieldnames:
        raise ValueError("domains.csv must have at least a 'domain' column")

    return rows


def write_normalized_csv(rows: List[dict], out_path: Path) -> None:
    """
    Write a normalized domains CSV with an extra 'normalized_domain' column.
    Does not overwrite the original domains.csv.
    """
    if not rows:
        print("No rows to write.")
        return

    fieldnames = list(rows[0].keys())
    if "normalized_domain" not in fieldnames:
        fieldnames.append("normalized_domain")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_and_normalize(domains_path: Path, write_normalized: bool) -> None:
    rows = read_domains_csv(domains_path)

    valid_count = 0
    invalid_entries: List[Tuple[int, str, str]] = []

    for idx, row in enumerate(rows, start=2):  # header is line 1
        raw_domain = row.get("domain", "")
        normalized = normalize_domain(raw_domain)
        row["normalized_domain"] = normalized

        if is_valid_domain(normalized):
            valid_count += 1
        else:
            invalid_entries.append((idx, raw_domain, normalized))

    print(f"Total rows: {len(rows)}")
    print(f"Valid domains: {valid_count}")
    print(f"Invalid domains: {len(invalid_entries)}")

    if invalid_entries:
        print("\nInvalid entries (line_number, original, normalized):")
        for line_no, original, normalized in invalid_entries:
            print(f"  Line {line_no}: '{original}' -> '{normalized}'")

    if write_normalized:
        out_path = domains_path.parent / "domains_normalized.csv"
        write_normalized_csv(rows, out_path)
        print(f"\nWrote normalized domains to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and normalize domains.csv")
    parser.add_argument(
        "--domains-csv",
        type=str,
        default="data/domains.csv",
        help="Path to domains.csv (default: data/domains.csv)",
    )
    parser.add_argument(
        "--write-normalized",
        action="store_true",
        help="Write data/domains_normalized.csv with a normalized_domain column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains_path = Path(args.domains_csv)
    validate_and_normalize(domains_path, args.write_normalized)


if __name__ == "__main__":
    main()
