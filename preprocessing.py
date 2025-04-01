"""
Data preprocessing functions for entity resolution.
"""

import re
import pandas as pd
from typing import Dict, Any


def normalize_company_name(name: str) -> str:
    """
    Normalize company name by removing suffixes and standardizing format.
    
    Args:
        name: Raw company name
        
    Returns:
        Normalized company name
    """
    if not isinstance(name, str) or not name.strip():
        return ""
            
    # Convert to lowercase
    name = name.lower()
    
    # Remove common legal suffixes
    suffixes = [
        r'\binc\.?\b', r'\bincorporated\b', r'\bcorp\.?\b', r'\bcorporation\b', 
        r'\bllc\b', r'\bltd\.?\b', r'\blimited\b', r'\bgmbh\b', r'\blp\b', r'\bplc\b', 
        r'\bco\.?\b', r'\bcompany\b', r'\bgroup\b', r'\bholdings\b', r'\bhldgs\b',
        r'\bservices\b', r'\bsolutions\b', r'\btechnologies\b', r'\btechnology\b', 
        r'\btech\b', r'\bsystems\b', r'\bllp\b', r'\bpllc\b', r'\bs\.?a\.?\b', 
        r'\bc\.?o\.?\b', r'\binc\b', r'\bcorp\b', r'\bltd\b'
    ]
    
    for suffix in suffixes:
        name = re.sub(suffix, '', name)
    
    # Remove punctuation and extra whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def normalize_domain(domain: str) -> str:
    """
    Normalize web domain by removing prefixes and standardizing format.
    
    Args:
        domain: Raw domain or URL
        
    Returns:
        Normalized domain
    """
    if not isinstance(domain, str) or not domain.strip():
        return ""
            
    domain = domain.lower()
    domain = re.sub(r'^https?://', '', domain)
    domain = re.sub(r'^www\.', '', domain)
    domain = domain.split('/')[0].split('?')[0]
    
    return domain


def normalize_address(address: str) -> str:
    """
    Normalize address by standardizing abbreviations and format.
    
    Args:
        address: Raw address string
        
    Returns:
        Normalized address
    """
    if not isinstance(address, str) or not address.strip():
        return ""
            
    address = address.lower()
    
    # Standardize common abbreviations
    abbr_dict = {
        'street': 'st', 'avenue': 'ave', 'boulevard': 'blvd', 'road': 'rd', 
        'drive': 'dr', 'lane': 'ln', 'suite': 'ste', 'apartment': 'apt',
        'building': 'bldg', 'floor': 'fl', 'highway': 'hwy', 'parkway': 'pkwy'
    }
    
    for full, abbr in abbr_dict.items():
        address = re.sub(r'\b' + full + r'\b', abbr, address)
        
    # Remove punctuation except for numbers with periods and hyphens in addresses
    address = re.sub(r'[^\w\s\-\.]', ' ', address)
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number by extracting digits only.
    
    Args:
        phone: Raw phone number
        
    Returns:
        Normalized phone number (digits only)
    """
    if not isinstance(phone, str) or not phone.strip():
        return ""
            
    digits = re.sub(r'\D', '', phone)
    
    # For US numbers, keep the last 10 digits
    if len(digits) > 10:
        # Check if it's an international phone number with country code
        digits = digits[-10:]
            
    return digits


def extract_record_data(record: pd.Series) -> Dict[str, Any]:
    """
    Extract relevant fields from a record for comparison.
    
    Args:
        record: Pandas Series containing a data record
        
    Returns:
        Dictionary of normalized fields for comparison
    """
    return {
        'id': record.name,
        'normalized_name': record.get('normalized_name', ''),
        'normalized_domain': record.get('normalized_domain', ''),
        'normalized_address': record.get('normalized_address', ''),
        'normalized_phone': record.get('normalized_phone', ''),
        'main_country_code': record.get('main_country_code', ''),
        'main_region': record.get('main_region', ''),
        'business_tags': record.get('business_tags', ''),
        'naics_primary_code': record.get('naics_2022_primary_code', ''),
        'email_domain': record.get('email_domain', ''),
        'has_name': record.get('has_name', False),
        'has_domain': record.get('has_domain', False),
        'has_address': record.get('has_address', False),
        'has_phone': record.get('has_phone', False)
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the entire dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Processed dataframe with normalized fields and blocking keys
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Normalize company name
    processed_df['normalized_name'] = processed_df['company_name'].fillna('').apply(normalize_company_name)
    
    # Normalize domain
    processed_df['normalized_domain'] = processed_df['website_domain'].fillna('').apply(normalize_domain)
    
    # Normalize address
    processed_df['normalized_address'] = processed_df['main_address_raw_text'].fillna('').apply(normalize_address)
    
    # Normalize phone
    processed_df['normalized_phone'] = processed_df['primary_phone'].fillna('').apply(normalize_phone)
    
    # Extract additional features
    processed_df['has_name'] = processed_df['normalized_name'].str.len() > 0
    processed_df['has_domain'] = processed_df['normalized_domain'].str.len() > 0
    processed_df['has_address'] = processed_df['normalized_address'].str.len() > 0
    processed_df['has_phone'] = processed_df['normalized_phone'].str.len() > 0
    
    # Create blocking keys
    processed_df['name_prefix'] = processed_df['normalized_name'].str[:3].fillna('')
    processed_df['country_code'] = processed_df['main_country_code'].fillna('')
    processed_df['region'] = processed_df['main_region'].fillna('')
    processed_df['domain_tld'] = processed_df['normalized_domain'].apply(
        lambda x: x.split('.')[-1] if x and '.' in x else ""
    )
    
    # Enhanced feature: Extract industry code
    processed_df['industry_code'] = processed_df['naics_2022_primary_code'].fillna('').astype(str).apply(
        lambda x: x[:2] if x and len(x) >= 2 else ""
    )
    
    # Enhanced feature: Create email domain feature
    processed_df['email_domain'] = processed_df['primary_email'].fillna('').apply(
        lambda x: x.split('@')[-1] if x and '@' in x else ""
    )
    
    return processed_df
