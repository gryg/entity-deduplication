"""
Rule-based entity resolution approach.
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Any
from rapidfuzz import fuzz

from base import EntityResolutionBase

logger = logging.getLogger("entity_resolution")

class RuleBasedResolver(EntityResolutionBase):
    """Enhanced rule-based entity resolution approach using deterministic rules."""
    
    def __init__(self, match_threshold=0.7):
        """
        Initialize the rule-based resolver with configurable parameters.
        
        Args:
            match_threshold: Similarity threshold for name matching (default: 0.7)
        """
        super().__init__(
            name="Rule-based Entity Resolution",
            description="Uses deterministic rules and exact/fuzzy matching to identify duplicates"
        )
        self.match_threshold = match_threshold
    
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run rule-based entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Generate rule-based matches
        matches = self._generate_rule_matches(processed_df)
        
        # Create clusters
        cluster_mapping = self._create_clusters(processed_df, matches)
        
        # Add clusters to original DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = result_df.index.map(lambda x: cluster_mapping.get(x, x))
        
        # Create canonical records
        canonical_df = self._create_canonical_records(df, cluster_mapping)
        
        return result_df, canonical_df
    
    def _generate_rule_matches(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Generate matches based on deterministic rules.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched record pairs
        """
        logger.info(f"[{self.name}] Generating rule-based matches...")
        
        matches = []
        
        # Rule 1: Exact domain match with moderate name similarity
        domain_groups = df[df['normalized_domain'] != ''].groupby('normalized_domain').indices
        for domain, indices in domain_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        rec1 = self._extract_record_data(df.iloc[idx1])
                        rec2 = self._extract_record_data(df.iloc[idx2])
                        name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                        
                        # If domain matches and name similarity is moderate, consider it a match
                        if name_sim > self.match_threshold:
                            matches.append((idx1, idx2, 0.9))
        
        # Rule 2: Exact phone match (complete numbers only)
        phone_groups = df[df['normalized_phone'] != ''].groupby('normalized_phone').indices
        for phone, indices in phone_groups.items():
            if len(indices) > 1 and len(phone) >= 10:  # Complete phone numbers only
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        # Check if they have the same country or region as an additional validation
                        rec1 = self._extract_record_data(df.iloc[idx1])
                        rec2 = self._extract_record_data(df.iloc[idx2])
                        
                        # Higher confidence if in same country/region
                        if rec1['main_country_code'] == rec2['main_country_code'] and rec1['main_country_code']:
                            matches.append((idx1, idx2, 0.95))
                        else:
                            matches.append((idx1, idx2, 0.85))
        
        # Rule 3: Very high name similarity + same country/region
        name_prefix_groups = df.groupby('name_prefix').indices
        for prefix, indices in name_prefix_groups.items():
            if len(indices) > 1 and len(indices) <= 100:  # Skip very large groups
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        rec1 = self._extract_record_data(df.iloc[idx1])
                        rec2 = self._extract_record_data(df.iloc[idx2])
                        
                        # Only compare if in same country
                        if rec1['main_country_code'] == rec2['main_country_code'] and rec1['main_country_code']:
                            name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                            
                            # Very high name similarity
                            if name_sim > 0.9:
                                matches.append((idx1, idx2, name_sim))
        
        # Rule 4: Email domain + name match
        email_domain_groups = df[df['email_domain'] != ''].groupby('email_domain').indices
        for domain, indices in email_domain_groups.items():
            if len(indices) > 1 and domain not in ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com']:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        rec1 = self._extract_record_data(df.iloc[idx1])
                        rec2 = self._extract_record_data(df.iloc[idx2])
                        name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                        
                        # Business email domain match is a strong signal 
                        if name_sim > 0.6:  # Lower threshold for email domain matches
                            matches.append((idx1, idx2, 0.85))
        
        # Rule 5: Industry code + name match
        if 'naics_2022_primary_code' in df.columns:
            naics_groups = df[df['naics_2022_primary_code'].notna()].groupby('naics_2022_primary_code').indices
            for code, indices in naics_groups.items():
                if len(indices) > 1 and len(indices) <= 200:  # Skip very large industry groups
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            idx1, idx2 = indices[i], indices[j]
                            rec1 = self._extract_record_data(df.iloc[idx1])
                            rec2 = self._extract_record_data(df.iloc[idx2])
                            
                            # Only compare if same country
                            if rec1['main_country_code'] == rec2['main_country_code'] and rec1['main_country_code']:
                                name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                                
                                # Industry + moderate name + country match
                                if name_sim > 0.85:
                                    matches.append((idx1, idx2, 0.8))
        
        logger.info(f"[{self.name}] Found {len(matches)} matches using rules")
        return matches
