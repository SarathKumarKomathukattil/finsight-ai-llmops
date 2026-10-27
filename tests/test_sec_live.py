import json
from finsight.sec_client import SECClient
from finsight.company_resolver import resolve_ticker
from finsight.validation import validate_submissions,validate_companyfacts

def test_sec_integration():
    client = SECClient()

    try:
        directory = json.loads(client.get_company_directory())

        cik = resolve_ticker(directory,'NVDA')
        assert cik == '0001045810'

        submissions = json.loads(client.get_submissions(cik))
        validate_submissions(submissions,cik)

        facts = json.loads(client.get_companyfacts(cik))
        validate_companyfacts(facts,cik)

        assert bool(facts['facts'])

    finally:
        client.close()




