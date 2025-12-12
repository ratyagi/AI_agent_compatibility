"""
probe_homepage.py
Fetch homepage for each domain in data/domains.csv, collect basic stats, and save HTML.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3

HTML_STATS_COLUMNS = [
    "domain",
    "status_code",
    "response_time",
    "content_length",
    "num_scripts",
    "num_links",
    "num_forms",
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
    raw_dir = Path("data/raw/html_raw")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, proc_dir


def request_with_retries(
    url: str, session: requests.Session, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[Optional[requests.Response], Optional[Exception], float]:
    """
    Request a URL with basic retry logic, returning (response, exception, response_time).
    response_time is measured only on the last attempt that returns a response.
    """
    last_exc: Optional[Exception] = None
    last_time = 0.0
    for _ in range(MAX_RETRIES):
        start = time.perf_counter()
        try:
            resp = session.get(url, timeout=timeout)
            last_time = time.perf_counter() - start
            return resp, None, last_time
        except (requests.RequestException, Exception) as exc:
            last_exc = exc
            last_time = time.perf_counter() - start
            time.sleep(1.0)
    return None, last_exc, last_time


def append_stats_row(stats_path: Path, row: dict) -> None:
    file_exists = stats_path.exists()
    with stats_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HTML_STATS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_html_stats(html: str) -> Tuple[int, int, int]:
    """
    Parse HTML and return (num_scripts, num_links, num_forms).
    If parsing fails, return zeros.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        num_scripts = len(soup.find_all("script"))
        num_links = len(soup.find_all("a"))
        num_forms = len(soup.find_all("form"))
        return num_scripts, num_links, num_forms
    except Exception:
        return 0, 0, 0


def process_domain(
    domain: str, out_dir: Path, stats_path: Path, force: bool, session: requests.Session
) -> None:
    out_file = out_dir / f"{domain}.html"
    if out_file.exists() and not force:
        # Idempotent: do not re-probe or duplicate stats rows.
        return

    url = f"https://{domain}"

    resp, err, response_time = request_with_retries(url, session)

    status_code = 0
    content = ""
    content_length = 0
    num_scripts = 0
    num_links = 0
    num_forms = 0

    if resp is not None:
        status_code = resp.status_code
        try:
            content = resp.text
            content_length = len(content)
        except Exception:
            content = ""
            content_length = 0

    if content and status_code and "html" in resp.headers.get("Content-Type", "").lower():
        num_scripts, num_links, num_forms = parse_html_stats(content)

    # Save HTML if we got any content
    if content:
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding="utf-8")
        except Exception as exc:
            logging.exception("Error writing HTML for %s: %s", domain, exc)

    row = {
        "domain": domain,
        "status_code": status_code,
        "response_time": round(response_time, 6),
        "content_length": content_length,
        "num_scripts": num_scripts,
        "num_links": num_links,
        "num_forms": num_forms,
    }
    append_stats_row(stats_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe homepage for each domain.")
    parser.add_argument(
        "--domains-csv",
        type=str,
        default="data/domains.csv",
        help="Path to domains.csv (default: data/domains.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download homepage even if HTML file already exists.",
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
    stats_path = proc_dir / "html_stats.csv"

    session = requests.Session()

    for domain in tqdm(domains, desc="Probing homepages"):
        process_domain(domain, out_dir, stats_path, args.force, session)


if __name__ == "__main__":
    main()