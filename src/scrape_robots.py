"""
scrape_robots.py
Fetch robots.txt for each domain in data/domains.csv.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm


DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3

ROBOTS_LOG_COLUMNS = [
    "domain",
    "status",
    "http_status",
    "content_length",
    "error_message",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def normalize_domain(raw: str) -> str:
    if raw is None:
        return ""
    d = raw.strip()
    if d.startswith("http://"):
        d = d[len("http://") :]
    elif d.startswith("https://"):
        d = d[len("https://") :]
    for sep in ["/", "?", "#"]:
        if sep in d:
            d = d.split(sep, 1)[0]
    if d.startswith("www."):
        d = d[4:]
    return d.strip()


def read_domains(domains_csv: Path) -> List[str]:
    if not domains_csv.exists():
        raise FileNotFoundError(f"domains.csv not found at {domains_csv}")

    domains: List[str] = []
    with domains_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "domain" not in reader.fieldnames:
            raise ValueError("domains.csv must have a 'domain' column")

        has_active = "active" in reader.fieldnames

        for row in reader:
            if has_active and str(row.get("active", "1")).strip() not in ("1", "true", "True"):
                continue
            dom = normalize_domain(row.get("domain", ""))
            if dom:
                domains.append(dom)
    return domains


def ensure_dirs() -> Tuple[Path, Path]:
    raw_dir = Path("data/raw/robots")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, proc_dir


def request_with_retries(
    url: str, session: requests.Session, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[Optional[requests.Response], Optional[Exception]]:
    last_exc: Optional[Exception] = None
    for _ in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=timeout)
            return resp, None
        except (requests.RequestException, Exception) as exc:
            last_exc = exc
            time.sleep(1.0)
    return None, last_exc


def append_log_row(log_path: Path, row: dict) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROBOTS_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_domain(domain: str, out_dir: Path, log_path: Path, force: bool, session: requests.Session) -> None:
    out_file = out_dir / f"{domain}.txt"
    if out_file.exists() and not force:
        append_log_row(
            log_path,
            {
                "domain": domain,
                "status": "skipped_existing",
                "http_status": "",
                "content_length": 0,
                "error_message": "",
            },
        )
        return

    url = f"https://{domain}/robots.txt"
    resp, err = request_with_retries(url, session)

    http_status = 0
    content = ""
    error_message = ""
    status = "success"

    if resp is not None:
        http_status = resp.status_code
        # Always save content, even for 404/500
        content = resp.text
    else:
        status = "error"
        error_message = str(err) if err else "Unknown error"

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(content, encoding="utf-8")
    except Exception as exc:
        logging.exception("Error writing robots.txt for %s", domain)
        status = "error"
        if error_message:
            error_message += f"; write_error={exc}"
        else:
            error_message = f"write_error={exc}"

    append_log_row(
        log_path,
        {
            "domain": domain,
            "status": status,
            "http_status": http_status if http_status else "",
            "content_length": len(content.encode("utf-8")) if content else 0,
            "error_message": error_message,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape robots.txt for each domain.")
    parser.add_argument(
        "--domains-csv",
        type=str,
        default="data/domains.csv",
        help="Path to domains.csv (default: data/domains.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download robots.txt even if output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    domains_csv = Path(args.domains_csv)

    domains = read_domains(domains_csv)
    if not domains:
        logging.warning("No domains found in %s", domains_csv)
        return

    out_dir, proc_dir = ensure_dirs()
    log_path = proc_dir / "robots_log.csv"

    session = requests.Session()

    for domain in tqdm(domains, desc="Scraping robots.txt"):
        process_domain(domain, out_dir, log_path, args.force, session)


if __name__ == "__main__":
    main()