import os
import re
import time
import httpx
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from finsight.validation import normalize_cik


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT/'.env')

logger = logging.getLogger(__name__)

DIRECTORY_URL = 'https://www.sec.gov/files/company_tickers.json'
BASE_URL = 'https://data.sec.gov'
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

RETRYABLE_STATUS = {408,429,500,502,503,504}

class RetryDeferredError(RuntimeError):
    """A request must be retried later"""
    def __init__(self,delay:float):
        self.delay = delay
        super().__init__(f'Retry deferred for {delay:.1f} seconds')

def build_filing_url(cik: str | int, accession: str, document: str) -> str:
    """Build a safe SEC filing document URL."""

    cik = normalize_cik(cik)

    if not isinstance(accession, str) or not re.fullmatch(
        r"\d{10}-\d{2}-\d{6}", accession
    ):
        raise ValueError("Invalid SEC accession number")

    if not isinstance(document, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:htm|html)",
        document,
        flags=re.IGNORECASE,
    ):
        raise ValueError("Invalid SEC HTML document filename")

    return (
        f"{ARCHIVE_URL}/{int(cik)}/"
        f"{accession.replace('-', '')}/{document}"
    )    

class SECClient:
    """Reusable HTTP client for SEC EDGAR APIs."""

    def __init__(self,requests_per_second:float=2.0,max_attempts:int=3,max_retry_wait:float=120.0):
        user_agent = os.getenv('SEC_USER_AGENT','').strip()

        if not user_agent or '@' not in user_agent:
            raise ValueError('Configure SEC_USER_AGENT in .env')

        if not 0< requests_per_second <=10:
            raise ValueError('Invalid request rate')
        if max_attempts <1:
            raise ValueError('max_attempts must be atleast 1')
        if max_retry_wait <0:
            raise ValueError('Invalid retry wait')

        self.interval = 1/requests_per_second
        self.max_attempts = max_attempts
        self.max_retry_wait = max_retry_wait
        self.last_request = None

        self.client = httpx.Client(
            headers={
               'User-Agent': user_agent,
               'Accept': 'application/json'
            },
            timeout=httpx.Timeout(30.0),
            follow_redirects=False
        )

    def close(self):
        self.client.close()

    def _wait_for_rate_limit(self):
        if self.last_request is not None:
            elapsed = time.monotonic() - self.last_request
            remaining = self.interval - elapsed

            if remaining > 0:
                time.sleep(remaining)

        self.last_request = time.monotonic()    

    def _retry_delay(self,response,attempt):
        if response is not None:
            header = response.headers.get('Retry-After')

            if header:
                if header.isascii() and header.isdecimal():
                    return float(header)
                try:
                    retry_date = parsedate_to_datetime(header)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=timezone.utc)

                    return max(0.0,(retry_date - datetime.now(timezone.utc)).total_seconds())

                except (TypeError,ValueError,OverflowError):
                    logger.warning(f'Invalid Retry-After: {header}')

        return min(30.0, 2 ** attempt) + random.uniform(0, 0.25)

    def _get(self, url: str, accept: str = "application/json") -> bytes:
        for attempt in range(self.max_attempts):
            self._wait_for_rate_limit()

            try:
                response = self.client.get(url, headers={"Accept": accept})
                response.raise_for_status()
                return response.content

            except (httpx.HTTPStatusError,httpx.TransportError) as error:
                response = None

                if isinstance(error,httpx.HTTPStatusError):
                    response = error.response

                    if response.status_code not in RETRYABLE_STATUS:
                        raise

                delay = self._retry_delay(response,attempt)

                if delay > self.max_retry_wait:
                    raise RetryDeferredError(delay) from error
                if attempt == self.max_attempts - 1:
                    raise

                logger.warning(f'SEC request failed. Retrying in {delay} seconds')
                time.sleep(delay)

    def get_company_directory(self) -> bytes:
        return self._get(DIRECTORY_URL)

    def get_submissions(self, cik:str|int) -> bytes:
        cik = normalize_cik(cik)

        url = f'{BASE_URL}/submissions/CIK{cik}.json'
        return self._get(url)

    def get_companyfacts(self,cik:str|int) -> bytes:
        cik = normalize_cik(cik)

        url = f'{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json'
        return self._get(url)
    
    def get_filing(self,cik: str | int,accession: str,document: str,) -> bytes:
        """Download an original SEC HTML filing."""

        url = build_filing_url(cik, accession, document)

        return self._get(url, accept="text/html,application/xhtml+xml")

                    






        
        



