"""SEC EDGAR downloader for ticker lookup, filing listing, and document fetch."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from rag_platform.config import get_settings

_REQUEST_DELAY = 0.5  # SEC Fair Access: >= 100ms between requests; 500ms to be safe


TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik}/{accession_nodash}/{document}"
)


@dataclass(frozen=True)
class FilingMeta:
    ticker: str
    cik: str
    form: str
    accession: str
    primary_document: str
    filing_date: str


def _headers() -> dict[str, str]:
    """Build the identification header required by SEC EDGAR."""
    user_agent = get_settings().sec_edgar_user_agent.strip()

    if not user_agent:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT must be set before calling SEC EDGAR."
        )

    return {"User-Agent": user_agent}


def lookup_cik(ticker: str, client: httpx.Client) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK."""
    time.sleep(_REQUEST_DELAY)
    response = client.get(TICKERS_URL, headers=_headers())
    response.raise_for_status()

    normalized_ticker = ticker.strip().upper()

    for entry in response.json().values():
        if entry["ticker"].upper() == normalized_ticker:
            return f"{entry['cik_str']:010d}"

    raise ValueError(f"Unknown ticker: {ticker}")


def list_filings(
    ticker: str,
    cik: str,
    client: httpx.Client,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    years: tuple[int, ...] = (),
) -> list[FilingMeta]:
    """Return filings filtered by form type and filing year."""
    url = SUBMISSIONS_URL.format(cik=cik)

    time.sleep(_REQUEST_DELAY)
    response = client.get(url, headers=_headers())
    response.raise_for_status()

    recent = response.json()["filings"]["recent"]
    filings: list[FilingMeta] = []

    for index, form in enumerate(recent["form"]):
        if form not in forms:
            continue

        filing_date = recent["filingDate"][index]

        if years and int(filing_date[:4]) not in years:
            continue

        filings.append(
            FilingMeta(
                ticker=ticker.strip().upper(),
                cik=cik,
                form=form,
                accession=recent["accessionNumber"][index],
                primary_document=recent["primaryDocument"][index],
                filing_date=filing_date,
            )
        )

    return filings


def download_filing(
    filing: FilingMeta,
    client: httpx.Client,
    raw_dir: Path,
) -> Path:
    """Download a filing into data/raw/{ticker}/{accession}.html."""
    accession_without_dashes = filing.accession.replace("-", "")

    url = ARCHIVES_URL.format(
        cik=int(filing.cik),
        accession_nodash=accession_without_dashes,
        document=filing.primary_document,
    )

    time.sleep(_REQUEST_DELAY)
    output_directory = raw_dir / filing.ticker
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{filing.accession}.html"

    if output_path.exists():
        return output_path

    time.sleep(_REQUEST_DELAY)
    response = client.get(url, headers=_headers())
    response.raise_for_status()
    output_path.write_bytes(response.content)

    return output_path