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
PARSER_VERSION ='v1'

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
        'tables': extracted_tables
    }


def process_filing(manifest_path:Path,data_root:Path) -> Path:
    """Verify a stored filing, parse it, and save processed JSON."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    if manifest.get("status") != "validated":
        raise ValueError("Filing ingestion was not successful")

    cik = normalize_cik(manifest['cik'])
    source = manifest["sources"]["latest_10q"]
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

    processed = {
        "parser_version": PARSER_VERSION,
        "ticker": manifest["ticker"],
        "cik": cik,
        "filing_type": source["form"],
        "filing_date": source["filing_date"],
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

    output_path = process_filing(
        args.manifest,
        PROJECT_ROOT / "data",
    )

    print(f"Processed filing saved: {output_path}")


if __name__ == "__main__":
    main()

