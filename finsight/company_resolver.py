from finsight.validation import normalize_cik

def resolve_ticker(directory:dict,ticker:str) -> str:
    """Return the SEC CIK associated with a ticker"""

    if not isinstance(directory,dict) or not directory:
        raise ValueError('Invalid SEC company directory')

    if not isinstance(ticker,str) or ticker.strip()=='':
        raise ValueError('Ticker must be a non-empty string')

    ticker = ticker.strip().upper()
    matches = set()

    for company in directory.values():
        if not isinstance(company,dict):
            continue

        company_ticker = company.get('ticker')
        if not isinstance(company_ticker,str):
            continue

        if company_ticker.strip().upper() == ticker:
            matches.add(normalize_cik(company.get('cik_str')))


    if not matches:
        raise ValueError(f'Ticker not found: {ticker}')

    if len(matches) > 1:
        raise ValueError(f'Ambiguous CIK mapping: {ticker}')

    return next(iter(matches))

   

