


-- Existing financial document chunks.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    filing_type VARCHAR NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR NOT NULL,
    section_id VARCHAR NOT NULL,
    heading VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    source_url VARCHAR NOT NULL,
    source_sha256 VARCHAR NOT NULL,
    embedding REAL[] NOT NULL,
    CONSTRAINT embedding_dimension CHECK (
        array_ndims(embedding) = 1
        AND cardinality(embedding) = 384
)
);

-- SEC filing metadata: one row per filing.
CREATE TABLE IF NOT EXISTS filings (
    accession_number VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    filing_type VARCHAR NOT NULL,
    filing_date DATE NOT NULL,
    report_date DATE,
    source_url VARCHAR NOT NULL,
    source_sha256 VARCHAR NOT NULL
);

-- Speed up filing selection by company, type and period.
CREATE INDEX IF NOT EXISTS idx_filings_lookup
ON filings (ticker, filing_type, report_date);
