import json
import hashlib

import pytest

from finsight.filing_parser import parse_filing, process_filing
from finsight.sec_client import build_filing_url
from finsight.storage import SnapshotStore


SAMPLE_HTML = b"""
<html>
<body>
<h1>Financial Report</h1>
<table>
    <tr><th>Metric</th><th>2026</th></tr>
    <tr><td>Revenue</td><td>100</td></tr>
</table>
</body>
</html>
"""


def test_parse_valid_html():
    result = parse_filing(SAMPLE_HTML)

    assert "Financial Report" in result["text"]
    assert result["html_table_count"] == 1
    assert result["parsed_table_count"] == 1

    table = result["tables"][0]

    assert table["shape"] == [1, 2]
    assert table["data"]["data"][0] == ["Revenue", 100]


def test_reject_invalid_html():
    with pytest.raises(ValueError, match="Invalid filing HTML"):
        parse_filing(b"Invalid document")


def test_process_filing(tmp_path):
    cik = "0001045810"
    accession = "0001045810-26-000075"
    document = "nvda-20260726.htm"

    store = SnapshotStore(tmp_path)

    snapshot = store.save_snapshot(
        cik,
        "filings",
        SAMPLE_HTML,
        True,
        accession_number=accession,
    )

    manifest = {
        "status": "validated",
        "ticker": "NVDA",
        "cik": cik,
        "sources": {
            "latest_10q": {
                **snapshot,
                "form": "10-Q",
                "filing_date": "2026-08-26",
                "accession_number": accession,
                "primary_document": document,
                "url": build_filing_url(cik, accession, document),
            }
        },
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_path = process_filing(manifest_path, tmp_path)

    assert output_path.exists()

    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["ticker"] == "NVDA"
    assert result["accession_number"] == accession
    assert result["parsed_table_count"] == 1

    # Verify repeat processing.
    second_path = process_filing(manifest_path, tmp_path)

    assert second_path == output_path

    # Verify original file integrity protection.
    raw_path = tmp_path / snapshot["path"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == snapshot["sha256"]

    raw_path.write_bytes(b"Modified data")

    with pytest.raises(OSError, match="SHA-256"):
        process_filing(manifest_path, tmp_path)

def test_narrative_sections():
    html = b"""
    <html>
    <body>

        <table>
            <tr>
                <td>Item 2. Management Discussion</td>
            </tr>
        </table>

        <div>Part I. Financial Information</div>

        <div>Item 2. Management Discussion and Analysis</div>
        <p>Revenue growth was driven by strong demand.</p>

        <div>Part II. Other Information</div>

        <div>Item 1A. Risk Factors</div>
        <p>Supply constraints could affect operations.</p>

    </body>
    </html>
    """

    result = parse_filing(html)

    assert result["section_count"] == 2

    sections = result["sections"]

    assert sections[0]["section_id"] == "part_i_item_2"
    assert sections[1]["section_id"] == "part_ii_item_1a"

    assert "Revenue growth" in sections[0]["text"]
    assert "Supply constraints" in sections[1]["text"]


def test_8k_full_document_fallback(tmp_path):
    cik = "0001045810"
    accession = "0001045810-26-000099"
    document = "event.htm"

    html = b"""
    <html><body>
        <h1>Form 8-K</h1>
        <p>Material corporate event disclosed.</p>
    </body></html>
    """

    store = SnapshotStore(tmp_path)

    snapshot = store.save_snapshot(
        cik, "filings", html, True,
        accession_number=accession,
    )

    manifest = {
        "status": "validated",
        "ticker": "NVDA",
        "cik": cik,
        "sources": {
            "filings": [{
                **snapshot,
                "form": "8-K",
                "filing_date": "2026-09-10",
                "accession_number": accession,
                "primary_document": document,
                "url": build_filing_url(cik, accession, document),
            }]
        },
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_path = process_filing(
        manifest_path, tmp_path, accession=accession
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["section_count"] == 1
    assert result["sections"][0]["section_id"] == "full_document"
    assert "Material corporate event" in result["sections"][0]["text"]
    assert result["sections"][0]["source_url"] == manifest["sources"]["filings"][0]["url"]