import re
import argparse
import json
import hashlib
import pandas as pd
from pathlib import Path
from io import BytesIO
from bs4 import BeautifulSoup

from finsight.validation import normalize_cik
from finsight.sec_client import build_filing_url
from finsight.storage import write_atomic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSER_VERSION ='v4'

def extract_sections(body) -> list[dict]:
    """Extract narrative sections from SEC 10-Q HTML."""

    block_tags = ["div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]

    sections = []
    current_part = None
    current_section = None

    for tag in body.find_all(block_tags):

        # Ignore table contents, including the table of contents.
        if tag.find_parent("table") or tag.find("table"):
            continue

        # Avoid reading the same text twice from nested blocks.
        if tag.find(block_tags):
            continue

        line = " ".join(tag.get_text(" ", strip=True).split())

        if not line:
            continue

        # Detect Part I or Part II.
        part_match = re.fullmatch(
            r"Part\s+(I|II)(?:\s*[.:\-]\s*"
            r"(?:Financial|Other)\s+Information)?\s*\.?",
            line,
            flags=re.IGNORECASE,
        )

        if part_match:
            current_part = part_match.group(1).upper()
            current_section = None
            continue

        # Detect Item 1, Item 2, Item 1A, etc.
        item_match = re.fullmatch(
            r"Item\s+(\d{1,2}[A-C]?)\s*[.:\-]?\s*(.{0,150})",
            line,
            flags=re.IGNORECASE,
        )

        if item_match and current_part:

            item = item_match.group(1).upper()

            current_section = {
                "section_id": f"part_{current_part.lower()}_item_{item.lower()}",
                "part": current_part,
                "item": item,
                "heading": line,
                "text_parts": [],
            }

            sections.append(current_section)
            continue

        # Add text to the current section.
        if current_section is not None:
            current_section["text_parts"].append(line)

    # Convert text lists into strings.
    for section in sections:
        section["text"] = " ".join(section.pop("text_parts"))

    return [section for section in sections if section["text"]]

def parse_filing(content:bytes) -> dict:
    """Extract readable text and HTML tables from a SEC filing."""

    soup = BeautifulSoup(content,'html.parser')
    body = soup.find('body')

    if soup.find('html') is None or body is None:
        raise ValueError('Invalid filing HTML')

    html_table_count = len(body.find_all('table'))
    for tag in body.find_all(['ix:header','script','style']):
        tag.decompose()

    text = body.get_text(' ',strip=True)
    if not text:
        raise ValueError('Empty filing text')
    sections = extract_sections(body)

    if html_table_count:
        dataframes = pd.read_html(
            BytesIO(content),
            keep_default_na=False
        )
    else:
        dataframes = []

    extracted_tables = []

    for index,df in enumerate(dataframes,start=1):
        extracted_tables.append({
            'parsed_table_number': index,
            'shape': list(df.shape),
            'data': json.loads(
                df.to_json(
                    orient='split',
                    force_ascii=False
                )
            )
        })

    return {
        'text': text,
        'html_table_count': html_table_count,
        'parsed_table_count': len(extracted_tables),
        'tables': extracted_tables,
        'section_count': len(sections),
        'sections': sections
    }


def process_filing(
    manifest_path: Path,
    data_root: Path,
    accession: str | None = None,
) -> Path:
    """Verify a stored filing, parse it, and save processed JSON."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    if manifest.get("status") != "validated":
        raise ValueError("Filing ingestion was not successful")

    cik = normalize_cik(manifest['cik'])
    filings = manifest["sources"].get("filings")

    if filings is not None:

        if accession is None:
            raise ValueError("Accession number required")

        matches = [
            filing
            for filing in filings
            if filing["accession_number"] == accession
        ]

        if len(matches) != 1:
            raise ValueError("Filing not found or duplicated in manifest")

        source = matches[0]

    else:
        # Support old single-10-Q manifests.
        source = manifest["sources"]["latest_10q"]

        if accession is not None and accession != source["accession_number"]:
            raise ValueError("Accession number mismatch")

        accession = source["accession_number"]

    expected_url = build_filing_url(
        cik,
        accession,
        source["primary_document"],
    )

    if source["url"] != expected_url:
        raise ValueError("Filing source URL mismatch")

    expected_dir = (
        data_root / "raw" / "sec" / cik / "filings" / accession
    ).resolve()

    raw_path = (data_root / source["path"]).resolve()

    if not raw_path.is_relative_to(expected_dir):
        raise ValueError("Invalid raw filing path")

    if raw_path.suffix.lower() != ".html":
        raise ValueError("Expected an HTML filing snapshot")

    content = raw_path.read_bytes()

    actual_hash = hashlib.sha256(content).hexdigest()

    if (
        actual_hash != source["sha256"]
        or raw_path.stem != actual_hash
    ):
        raise OSError("Raw filing SHA-256 integrity check failed")
    extracted = parse_filing(content)

    # Preserve documents whose sections are not recognized.
    if not extracted["sections"]:

        extracted["sections"] = [{
            "section_id": "full_document",
            "part": None,
            "item": None,
            "heading": f"{source['form']} Full Document",
            "text": extracted["text"],
        }]

        extracted["section_count"] = 1

    for section in extracted["sections"]:
        section["accession_number"] = accession
        section["source_url"] = source["url"]

    processed = {
        "parser_version": PARSER_VERSION,
        "ticker": manifest["ticker"],
        "cik": cik,
        "filing_type": source["form"],
        "filing_date": source["filing_date"],
        "report_date": source.get("report_date") or None,
        "accession_number": accession,
        "source": {
            "url": source["url"],
            "raw_path": source["path"],
            "sha256": actual_hash,
        },
        **extracted,
    }

    

    output_path = (
        data_root
        / "processed"
        / "sec"
        / cik
        / accession
        / f"{actual_hash}.{PARSER_VERSION}.json"
    )

    output = json.dumps(
        processed,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

   
    write_atomic(output_path, output)

    return output_path
    
def main():
    parser = argparse.ArgumentParser(
        description="Process an existing SEC filing"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a successful filing ingestion manifest",
    )

    args = parser.parse_args()

    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )

    filings = manifest.get("sources", {}).get("filings")

    if filings is not None:

        accessions = [
            filing["accession_number"]
            for filing in filings
        ]

    else:
        accessions = [None]


    for accession in accessions:

        output_path = process_filing(
            args.manifest,
            PROJECT_ROOT / "data",
            accession=accession,
        )

        print(f"Processed filing saved: {output_path}")


if __name__ == "__main__":
    main()

