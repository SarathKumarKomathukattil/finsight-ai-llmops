import re
from datetime import date

def normalize_cik(cik:str | int) -> str:
    """Return a validated 10-digit CIK"""

    if isinstance(cik,bool) or not isinstance(cik,(str,int)):
        raise ValueError('Invalid CIK type')
    
    cik = str(cik)

    if not cik.isascii() or not cik.isdecimal() or not 1<=len(cik)<=10:
        raise ValueError('CIK must contain 1 to 10 digits')

    return cik.zfill(10)

def validate_cik(data:dict,expected_cik:str) -> None:
    if not isinstance(data,dict):
        raise ValueError('SEC response must be a JSON object')

    actual_cik = normalize_cik(data.get('cik'))

    if actual_cik != normalize_cik(expected_cik):
        raise ValueError('CIK does not match requested company')

    
def validate_submissions(data:dict,cik:str) -> None:
    """Validate SEC submissions data structure"""
    validate_cik(data,cik)

    filings = data.get('filings')
    if not isinstance(filings,dict):
        raise ValueError('Missing filings object')

    recent = filings.get('recent')
    if not isinstance(recent,dict):
        raise ValueError('Missing recent filings')

    required = ["accessionNumber", "filingDate", "form"]
    for field in required:
        if not isinstance(recent.get(field),list):
            raise ValueError(f'Missing or invalid field: {field}')

    expected_length = len(recent['accessionNumber'])
    for field, values in recent.items():
        if not isinstance(values,list):
            raise ValueError(f'Invalid filing column: {field}')

        if len(values) != expected_length:
            raise ValueError(f'Filing column length mismatch: {field}')

    for i in range(expected_length):
        accession = recent['accessionNumber'][i]
        filing_date = recent['filingDate'][i]
        form = recent['form'][i]

        if not isinstance(accession,str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',accession):
            raise ValueError(f'Invalid accession number at index {i}')
        
        try:
            date.fromisoformat(filing_date)
        except (TypeError,ValueError):
            raise ValueError(f'Invalid filing date at index {i}')

        if not isinstance(form,str) or form.strip()=='':
            raise ValueError(f'Invalid filing form at index {i}')


def validate_companyfacts(data:dict,cik:str) -> None:
    """Validate SEC Company Facts response structure"""

    validate_cik(data,cik)

    facts = data.get('facts')

    if not isinstance(facts,dict) or not facts:
        raise ValueError('Missing financial facts')

    for taxonomy,concepts in facts.items():
        if not isinstance(concepts,dict):
            raise ValueError(f'Invalid taxonomy: {taxonomy}')

        for concept_name, concept in concepts.items():
            if not isinstance(concept, dict):
                raise ValueError(f'Invalid financial concept: {concept_name}')
            
            if not isinstance(concept.get('units'), dict):
                raise ValueError(f'Missing units for financial concept: {concept_name}')

        



        



    
    

