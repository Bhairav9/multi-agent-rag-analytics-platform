"""Unit tests for SEC EDGAR downloader (hermetic, respx-mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

from rag_platform.ingestion.sec_edgar import (
    TICKERS_URL,
    SUBMISSIONS_URL,
    FilingMeta,
    download_filing,
    lookup_cik,
    list_filings,
)

SAMPLE_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2024-10-30", "2024-07-30", "2023-05-01"],
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000098",
                "0000320193-23-000050",
            ],
            "primaryDocument": ["aapl-10k.htm", "aapl-10q.htm", "aapl-8k.htm"],
        }
    }
}

@respx.mock
def test_lookup_cik_finds_ticker() -> None:
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_TICKERS))
    with httpx.Client() as client:
        cik = lookup_cik("AAPL", client)
    assert cik == "0000320193"


@respx.mock
def test_lookup_cik_unknown_ticker_raises() -> None:
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_TICKERS))
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="Unknown ticker"):
            lookup_cik("ZZZZ", client)


@respx.mock
def test_list_filings_filters_by_form() -> None:
    cik = "0000320193"
    respx.get(SUBMISSIONS_URL.format(cik=cik)).mock(
        return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
    )
    with httpx.Client() as client:
        filings = list_filings("AAPL", cik, client)
    assert len(filings) == 2
    assert all(f.form in ("10-K", "10-Q") for f in filings)

@respx.mock
def test_list_filings_filters_by_year() -> None:
    cik = "0000320193"
    respx.get(SUBMISSIONS_URL.format(cik=cik)).mock(
        return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
    )
    with httpx.Client() as client:
        filings = list_filings("AAPL", cik, client, years=(2024,))
    assert len(filings) == 2
    assert all(f.filing_date.startswith("2024") for f in filings)


@respx.mock
def test_download_filing_writes_file(tmp_path) -> None:
    filing = FilingMeta(
        ticker="AAPL",
        cik="0000320193",
        form="10-K",
        accession="0000320193-24-000123",
        primary_document="aapl-10k.htm",
        filing_date="2024-10-30",
    )
    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        "320193/000032019324000123/aapl-10k.htm"
    )
    respx.get(url).mock(return_value=httpx.Response(200, content=b"<html>fake</html>"))

    with httpx.Client() as client:
        path = download_filing(filing, client, raw_dir=tmp_path)

    assert path == tmp_path / "AAPL" / "0000320193-24-000123.html"
    assert path.read_bytes() == b"<html>fake</html>"


def test_headers_missing_user_agent_raises(monkeypatch) -> None:
    from rag_platform.config import Settings

    monkeypatch.setattr(
        "rag_platform.ingestion.sec_edgar.get_settings",
        lambda: Settings(sec_edgar_user_agent=""),
    )
    with httpx.Client() as client:
        with pytest.raises(RuntimeError, match="SEC_EDGAR_USER_AGENT"):
            lookup_cik("AAPL", client)