"""
scrape_tos.py
Discover and download Terms of Service text for each domain in data/domains.csv.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3

TOS_LOG_COLUMNS = [
    "domain",
    "status",
    "http_status",
    "method",
    "chosen_url",
    "error_message",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def normalize_domain(raw: str) -> str:
    """Normalize domain similar to collect_domains.py."""
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
    raw_dir = Path("data/raw/tos_texts")
    proc_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, proc_dir


def request_with_retries(
    url: str, session: requests.Session, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[Optional[requests.Response], Optional[Exception]]:
    """Request a URL with basic retry logic."""
    last_exc: Optional[Exception] = None
    for _ in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=timeout)
            return resp, None
        except (requests.RequestException, Exception) as exc:
            last_exc = exc
            time.sleep(1.0)
    return None, last_exc


def extract_visible_text(html: str) -> str:
    """Extract visible text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script/style/noscript and similar.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # Get text and normalize whitespace.
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove empty lines
    return "\n".join(lines)


def find_tos_by_patterns(domain: str, session: requests.Session) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Try common TOS URL patterns."""
    patterns = [
        "/terms",
        "/terms/",
        "/terms-of-service",
        "/terms-of-service/",
        "/terms-and-conditions",
        "/terms-and-conditions/",
        "/legal",
        "/legal/",
        "/legal/terms",
        "/legal/terms-of-service",
    ]

    base = f"https://{domain}"
    for path in patterns:
        url = base + path
        resp, err = request_with_retries(url, session)
        if resp is None:
            continue
        if resp.status_code == 200 and "html" in resp.headers.get("Content-Type", "").lower():
            return url, resp.status_code, resp.text

    return None, None, None


def find_tos_by_link_search(domain: str, session: requests.Session) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Fetch the homepage and search anchor tags for TOS/Legal links.
    """
    base_url = f"https://{domain}"
    resp, err = request_with_retries(base_url, session)
    if resp is None or "html" not in resp.headers.get("Content-Type", "").lower():
        return None, None, None

    soup = BeautifulSoup(resp.text, "lxml")
    candidates: List[Tuple[int, str]] = []  # (priority, href)

    keywords = [("terms", 1), ("legal", 2), ("conditions", 3)]

    for a in soup.find_all("a"):
        anchor_text = (a.get_text() or "").strip().lower()
        href = a.get("href")
        if not href or not anchor_text:
            continue

        for word, priority in keywords:
            if word in anchor_text:
                candidates.append((priority, href))
                break

    if not candidates:
        return None, None, None

    # Sort by priority (lowest is best) and then by href length
    candidates.sort(key=lambda x: (x[0], len(x[1])))
    best_href = candidates[0][1]
    tos_url = urljoin(base_url, best_href)

    resp2, err2 = request_with_retries(tos_url, session)
    if resp2 is None or resp2.status_code != 200 or "html" not in resp2.headers.get("Content-Type", "").lower():
        return None, None, None

    return tos_url, resp2.status_code, resp2.text


def append_log_row(log_path: Path, row: dict) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TOS_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_domain(
    domain: str, out_dir: Path, log_path: Path, force: bool, session: requests.Session
) -> None:
    out_file = out_dir / f"{domain}.txt"
    if out_file.exists() and not force:
        append_log_row(
            log_path,
            {
                "domain": domain,
                "status": "skipped_existing",
                "http_status": "",
                "method": "",
                "chosen_url": "",
                "error_message": "",
            },
        )
        return

    # Try patterns
    resp_html = None
    http_status = None
    chosen_url = None
    method = None
    error_message = ""

    url, status, html = find_tos_by_patterns(domain, session)
    if url and html:
        resp_html = html
        http_status = status
        chosen_url = url
        method = "pattern"
    else:
        # Try link search
        url2, status2, html2 = find_tos_by_link_search(domain, session)
        if url2 and html2:
            resp_html = html2
            http_status = status2
            chosen_url = url2
            method = "link_search"

    if resp_html is None:
        append_log_row(
            log_path,
            {
                "domain": domain,
                "status": "not_found",
                "http_status": http_status or "",
                "method": method or "none",
                "chosen_url": chosen_url or "",
                "error_message": "",
            },
        )
        return

    try:
        text = extract_visible_text(resp_html)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")
        append_log_row(
            log_path,
            {
                "domain": domain,
                "status": "success",
                "http_status": http_status,
                "method": method,
                "chosen_url": chosen_url,
                "error_message": "",
            },
        )
    except Exception as exc:
        logging.exception("Error processing TOS for domain %s", domain)
        append_log_row(
            log_path,
            {
                "domain": domain,
                "status": "error",
                "http_status": http_status or "",
                "method": method or "",
                "chosen_url": chosen_url or "",
                "error_message": str(exc),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape TOS text for each domain.")
    parser.add_argument(
        "--domains-csv",
        type=str,
        default="data/domains.csv",
        help="Path to domains.csv (default: data/domains.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download TOS even if output file already exists.",
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
    log_path = proc_dir / "tos_log.csv"

    session = requests.Session()

    for domain in tqdm(domains, desc="Scraping TOS"):
        process_domain(domain, out_dir, log_path, args.force, session)


if __name__ == "__main__":
    main()