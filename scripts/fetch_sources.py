"""Manual, one-time source fetcher for the mutual fund FAQ corpus.

This script is NOT imported by the application. The app never touches the network;
it only reads the text snapshots this script produces in data/raw/.

Usage:
    python scripts/fetch_sources.py                 # dry run: prints the plan, fetches nothing
    python scripts/fetch_sources.py --fetch         # download PDFs into data/raw/_pdf/
    python scripts/fetch_sources.py --extract       # convert downloaded PDFs to data/raw/**.txt
    python scripts/fetch_sources.py --fetch --extract

Exit codes:
    0  success
    2  a URL is outside the allow-list  (refused by design)
    3  download or extraction failed
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# PRD 4.4: public official sources only. No third-party blogs, no aggregators.
ALLOWED_DOMAINS = (
    "files.hdfcfund.com",  # HDFC AMC document CDN
    "www.hdfcfund.com",    # HDFC AMC site (bot-blocked; listed for completeness)
    "www.amfiindia.com",   # AMFI
    "portal.amfiindia.com",  # AMFI document portal
    "www.sebi.gov.in",     # SEBI
    "www.camsonline.com",  # CAMS - RTA for HDFC MF
)

RETRIEVED = "2026-09-28"

KIM_URL = (
    "https://files.hdfcfund.com/s3fs-public/KIM/2025-11/"
    "KIM%20-%20{disp}%20dated%20November%2021%2C%202025.pdf"
)
FACTSHEET_PDF = (
    "https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20June%202026.pdf"
)
SAI_URL = (
    "https://files.hdfcfund.com/s3fs-public/2025-09/SAI%20with%20updated%20Addendum"
    "%20dated%20September%2025%2C%202025.pdf"
)
CHARTER_URL = "https://files.hdfcfund.com/s3fs-public/2024-05/Investor%20Charter%20-%20MF.pdf"

# printed page -> PDF index is +1 in the June 2026 master factsheet
FACT_SCHEMES = {
    "hdfc-flexi-cap": ("HDFC Flexi Cap Fund", [6, 7]),
    "hdfc-large-cap": ("HDFC Large Cap Fund", [10, 11]),
    "hdfc-small-cap": ("HDFC Small Cap Fund", [14, 15]),
    "hdfc-balanced-advantage": ("HDFC Balanced Advantage Fund", [41, 42, 43, 44]),
    "hdfc-elss": ("HDFC ELSS Tax Saver", [60, 61]),
}
RISKOMETER_PAGES = list(range(122, 138))
SAI_PAGES = [85, 86, 169]

KIM_SCHEMES = {
    "hdfc-large-cap": "HDFC Large Cap Fund",
    "hdfc-flexi-cap": "HDFC Flexi Cap Fund",
    "hdfc-elss": "HDFC ELSS Tax Saver",
    "hdfc-small-cap": "HDFC Small Cap Fund",
    "hdfc-balanced-advantage": "HDFC Balanced Advantage Fund",
}


@dataclass
class Job:
    slug: str
    title: str
    source_type: str
    url: str
    pdf_name: str
    as_of: str
    doc_date: str = ""
    pages: list[int] | None = None
    out_name: str = ""
    publisher: str = "HDFC Asset Management Company Limited"
    notes: list[str] = field(default_factory=list)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for slug, (disp, pages) in FACT_SCHEMES.items():
        jobs.append(
            Job(
                slug=slug,
                title=f"HDFC Mutual Fund Factsheet - June 2026 ({disp})",
                source_type="factsheet",
                url=FACTSHEET_PDF,
                pdf_name="hdfc-mf-factsheet-2026-06.pdf",
                as_of="2026-06-30",
                doc_date="2026-06-30",
                pages=pages,
                out_name="factsheet-2026-06.txt",
            )
        )
    jobs.append(
        Job(
            slug="general",
            title="HDFC Mutual Fund Factsheet - June 2026 (Annexure: Benchmark and Scheme riskometers)",
            source_type="riskometer",
            url=FACTSHEET_PDF,
            pdf_name="hdfc-mf-factsheet-2026-06.pdf",
            as_of="2026-06-30",
            doc_date="2026-06-30",
            pages=RISKOMETER_PAGES,
            out_name="riskometer-benchmark-annexure-2026-06.txt",
        )
    )
    for slug, disp in KIM_SCHEMES.items():
        jobs.append(
            Job(
                slug=slug,
                title=f"Key Information Memorandum (KIM) - {disp}",
                source_type="kim",
                url=KIM_URL.format(disp=disp.replace(" ", "%20")),
                pdf_name=f"{slug}-kim-2025-11.pdf",
                as_of="2025-11-21",
                doc_date="2025-11-21",
                out_name="kim-2025-11.txt",
            )
        )
    jobs.append(
        Job(
            slug="general",
            title=(
                "Statement of Additional Information (SAI) - extract: Rights of Unit Holders / "
                "Disclosure and Reports (Account Statements, Consolidated Account Statement, MFCentral)"
            ),
            source_type="guide",
            url=SAI_URL,
            pdf_name="SAI_with_updated_Addendum_dated_September_25,_2025.pdf",
            as_of="2025-09-25",
            doc_date="2025-06-30 (addendum 2025-09-25)",
            pages=SAI_PAGES,
            out_name="sai-investor-statements-2025-09.txt",
        )
    )
    jobs.append(
        Job(
            slug="general",
            title="HDFC Mutual Fund - Investor Charter",
            source_type="guide",
            url=CHARTER_URL,
            pdf_name="Investor_Charter_-_MF.pdf",
            as_of="2024-05-01",
            doc_date="2024-05-01",
            out_name="investor-charter-2024-05.txt",
        )
    )
    return jobs


def check_domain(url: str) -> None:
    from urllib.parse import urlparse

    host = urlparse(url).netloc.lower()
    if host not in ALLOWED_DOMAINS:
        raise SystemExit(
            f"REFUSED: {url}\n  host '{host}' is not in the allow-list.\n"
            f"  allowed: {', '.join(ALLOWED_DOMAINS)}\n"
            "  PRD 4.4 requires public official sources only."
        )


def download(url: str, dest: Path) -> int:
    check_domain(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return len(data)


def clean_page(page) -> str:
    text = page.extract_text() or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def header_for(job: Job, n_pages: int) -> str:
    page_line = (
        f"{', '.join(str(p) for p in job.pages)} (PDF index)"
        if job.pages
        else (str(n_pages) if n_pages else "all")
    )
    return (
        "SOURCE DOCUMENT\n"
        f"title        : {job.title}\n"
        f"scheme       : {job.title}\n"
        f"source_type  : {job.source_type}\n"
        f"publisher    : {job.publisher}\n"
        f"source_url   : {job.url}\n"
        f"doc_date     : {job.doc_date}\n"
        f"as_of        : {job.as_of}\n"
        f"pages        : {page_line}\n"
        f"retrieved    : {RETRIEVED}\n"
        + "=" * 80
        + "\n\n"
    )


def extract(job: Job, pdf_path: Path) -> Path:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if job.pages:
        body = "\n\n".join(clean_page(reader.pages[p - 1]) for p in job.pages)
    else:
        body = "\n".join(clean_page(page) for page in reader.pages)
    out = Path("data/raw") / job.slug / job.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header_for(job, len(reader.pages)) + body, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="actually download PDFs")
    parser.add_argument("--extract", action="store_true", help="convert PDFs to data/raw/**.txt")
    args = parser.parse_args()

    if not args.fetch and not args.extract:
        args.fetch = args.extract = False
        mode = "DRY RUN (default) - pass --fetch --extract to act"
    else:
        mode = "LIVE"

    jobs = build_jobs()
    print(f"Mode: {mode}")
    print(f"Allow-list: {', '.join(ALLOWED_DOMAINS)}\n")
    print(f"{len(jobs)} source jobs\n" + "=" * 78)

    seen_pdfs: dict[str, Path] = {}
    failures: list[str] = []

    for job in jobs:
        print(f"\n[{job.slug}] {job.title}")
        print(f"    type    : {job.source_type}")
        print(f"    url     : {job.url}")
        print(f"    as_of   : {job.as_of}")
        try:
            check_domain(job.url)
        except SystemExit as exc:
            print(f"    !! {exc}")
            failures.append(job.url)
            continue

        if args.fetch and job.pdf_name not in seen_pdfs:
            pdf_path = Path("data/raw/_pdf") / job.pdf_name
            try:
                size = download(job.url, pdf_path)
                seen_pdfs[job.pdf_name] = pdf_path
                print(f"    fetched : {size:,} bytes -> {pdf_path}")
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                print(f"    FAILED  : {exc}")
                failures.append(job.url)
                continue
        elif not args.fetch:
            print("    (not fetched - dry run)")

        if args.extract:
            pdf_path = seen_pdfs.get(job.pdf_name) or (Path("data/raw/_pdf") / job.pdf_name)
            if not pdf_path.exists():
                print(f"    SKIP    : {pdf_path} missing, run --fetch first")
                failures.append(job.url)
                continue
            try:
                out = extract(job, pdf_path)
                print(f"    extracted -> {out} ({out.stat().st_size:,} bytes)")
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED  : {exc}")
                failures.append(job.url)

    print("\n" + "=" * 78)
    if not args.fetch and not args.extract:
        print("Dry run complete. Re-run with --fetch --extract to build data/raw/.")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for url in failures:
            print(f"  - {url}")
        return 3
    print("All jobs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
