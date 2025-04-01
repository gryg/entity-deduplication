"""
Scalable Deterministic Feature-based entity resolution approach.

This resolver focuses on identifying matching entities by systematically comparing 
critical deterministic features like website, phone, and email, while properly 
handling missing data and using a hierarchical confidence system.

This implementation uses advanced blocking strategies and parallelization
to efficiently handle large datasets with tens of thousands of records.
"""

import logging
import pandas as pd
import numpy as np
import multiprocessing
from typing import Dict, List, Tuple, Set, Any, Optional, Callable
from collections import defaultdict
from rapidfuzz import fuzz
from datetime import datetime
import time
import itertools
import hashlib
import re
from functools import partial

from base import EntityResolutionBase

logger = logging.getLogger("entity_resolution")

class DeterministicFeatureResolver(EntityResolutionBase):
    """
    Deterministic feature-based entity resolution that systematically compares entities.
    
    This approach focuses on crucial deterministic features like website, phone, and email domain,
    applying a confidence-based matching system that properly handles missing data.
    It performs efficient matching using advanced blocking strategies.
    """
    
    def __init__(self, 
                 match_threshold=0.65, 
                 feature_weights=None,
                 batch_size=10000,
                 max_comparisons=None,
                 n_jobs=-1,
                 use_advanced_blocking=True,
                 use_progressive_resolution=True,
                 blocking_threshold=10000,
                 block_size_limit=1000,
                 max_deduplication_rate=0.40):
        """
        Initialize the deterministic feature resolver with configurable parameters.
        
        Args:
            match_threshold: Threshold for considering a match (default: 0.65)
            feature_weights: Dictionary of feature names and their weights
            batch_size: Process comparisons in batches of this size
            max_comparisons: Maximum number of comparisons to perform (for large datasets)
            n_jobs: Number of parallel jobs (-1 for all available cores)
            use_advanced_blocking: Whether to use advanced blocking strategies
            use_progressive_resolution: Whether to use progressive resolution
            blocking_threshold: Max number of records for traditional blocking
            block_size_limit: Maximum block size to prevent quadratic explosion
            max_deduplication_rate: Maximum allowed deduplication rate (0.0-1.0)
        """
        super().__init__(
            name="Scalable Deterministic Feature Entity Resolution",
            description="Uses critical deterministic features to identify duplicates with advanced blocking and parallelization"
        )
        self.match_threshold = match_threshold
        self.batch_size = batch_size
        self.max_comparisons = max_comparisons
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count() - 1)
        self.use_advanced_blocking = use_advanced_blocking
        self.use_progressive_resolution = use_progressive_resolution
        self.blocking_threshold = blocking_threshold
        self.block_size_limit = block_size_limit
        self.max_deduplication_rate = max_deduplication_rate
        
        # Define the weights for different features
        self.feature_weights = feature_weights or {
            # Deterministic features (highest weights)
            'website_exact': 0.99,       # Same website - very strong signal
            'email_domain_exact': 0.90,  # Same email domain (not gmail, etc.) 
            'phone_exact': 0.95,         # Same phone number
            'social_media_exact': 0.90,  # Same social media profiles
            
            # Near-deterministic features - not enough on their own
            'name_exact': 0.85,          # Exact company name match
            'name_very_similar': 0.75,   # Very similar name (>90% similarity)
            'domain_exact': 0.85,        # Same domain
            
            # Strong indicators - supporting evidence
            'address_exact': 0.70,       # Same address
            'address_similar': 0.40,     # Similar address
            'phone_partial': 0.60,       # Partial phone match
            
            # Supporting evidence
            'same_country': 0.20,        # Same country
            'same_region': 0.25,         # Same region/state
            'same_city': 0.30,           # Same city
            'business_category': 0.25    # Same business category
        }
        
        # For tracking
        self.total_comparisons = 0
        self.total_matches = 0
        self.processing_stats = {}
    
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the deterministic feature-based entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        start_time = datetime.now()
        
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Initialize counters
        self.total_comparisons = 0
        self.total_matches = 0
        
        # Use a hybrid approach instead of all-or-nothing
        logger.info(f"[{self.name}] Using high-precision deterministic matching")
        matches = self._high_precision_matching(processed_df)
        
        logger.info(f"[{self.name}] Found {len(matches)} high-precision matches")
        
        # Create clusters from the matches
        cluster_mapping = self._create_clusters(processed_df, matches)
        
        # Add cluster IDs to the original DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = result_df.index.map(lambda x: cluster_mapping.get(x, x))
        
        # Create canonical records
        canonical_df = self._create_canonical_records(df, cluster_mapping)
        
        # Record processing stats
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.processing_stats = {
            'total_records': len(df),
            'unique_entities': len(canonical_df),
            'total_comparisons': self.total_comparisons,
            'total_matches': self.total_matches,
            'processing_time': processing_time,
            'comparisons_per_second': self.total_comparisons / processing_time if processing_time > 0 else 0,
            'deduplication_rate': (len(df) - len(canonical_df)) / len(df) * 100
        }
        
        logger.info(f"[{self.name}] Resolution completed: {self.total_matches} matches found from {self.total_comparisons:,} comparisons")
        logger.info(f"[{self.name}] Comparison rate: {self.processing_stats['comparisons_per_second']:.2f} comparisons/second")
        logger.info(f"[{self.name}] Deduplication rate: {self.processing_stats['deduplication_rate']:.2f}%")
        
        return result_df, canonical_df
    
    def _find_matches_with_advanced_blocking(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Find matching entities using advanced blocking strategies for scalability.
        
        This method uses multiple blocking strategies in parallel:
        1. Exact deterministic feature blocking (domain, phone, etc.)
        2. Name token blocking (individual words in company names)
        3. N-gram token blocking for fuzzy matching
        4. Meta-blocking to reduce block sizes
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched record pairs
        """
        logger.info(f"[{self.name}] Finding matches using advanced blocking strategies...")
        
        # Initialize tracking variables
        all_matches = []
        total_blocks = 0
        total_block_comparisons = 0
        compared_pairs = set()
        
        # Step 1: Create multiple types of blocks
        blocking_start = time.time()
        
        # Create blocks based on deterministic features first (domains, phones, etc.)
        deterministic_blocks = self._create_deterministic_blocks(df)
        total_blocks += len(deterministic_blocks)
        logger.info(f"[{self.name}] Created {len(deterministic_blocks)} deterministic blocks")
        
        # Create name-based blocks (token blocking)
        name_token_blocks = self._create_name_token_blocks(df)
        total_blocks += len(name_token_blocks)
        logger.info(f"[{self.name}] Created {len(name_token_blocks)} name token blocks")
        
        # Create n-gram blocks for fuzzy matching (if dataset is not too large)
        ngram_blocks = {}
        if len(df) < 10000:  # Only for smaller datasets as this can create many blocks
            ngram_blocks = self._create_ngram_blocks(df)
            total_blocks += len(ngram_blocks)
            logger.info(f"[{self.name}] Created {len(ngram_blocks)} n-gram blocks")
        
        # Create special blocks for likely matches
        special_blocks = self._create_special_blocks(df)
        total_blocks += len(special_blocks)
        logger.info(f"[{self.name}] Created {len(special_blocks)} special feature blocks")
        
        # Combine all blocking strategies
        all_blocks = {}
        all_blocks.update(deterministic_blocks)
        all_blocks.update(name_token_blocks)
        all_blocks.update(ngram_blocks)
        all_blocks.update(special_blocks)
        
        # Filter out oversized blocks to prevent quadratic explosion
        filtered_blocks = self._filter_blocks(all_blocks, df)
        
        # Log blocking statistics
        blocking_time = time.time() - blocking_start
        avg_block_size = sum(len(indices) for indices in filtered_blocks.values()) / max(1, len(filtered_blocks))
        logger.info(f"[{self.name}] Created {len(filtered_blocks)} blocks (avg size: {avg_block_size:.1f}) in {blocking_time:.2f}s")
        
        # Step 2: Process blocks in parallel
        if self.use_progressive_resolution:
            # Process blocks in stages based on confidence
            block_confidence_levels = {
                'high': {k: v for k, v in filtered_blocks.items() if k.startswith(('domain_', 'phone_', 'email_', 'special_'))},
                'medium': {k: v for k, v in filtered_blocks.items() if k.startswith('social_') or k.startswith('name_exact_')},
                'low': {k: v for k, v in filtered_blocks.items() if k.startswith(('token_', 'ngram_'))}
            }
            
            # Process each confidence level
            for level, blocks in block_confidence_levels.items():
                if not blocks:
                    continue
                
                logger.info(f"[{self.name}] Processing {len(blocks)} {level}-confidence blocks...")
                level_matches, level_comparisons, level_compared = self._process_blocks_parallel(df, blocks, compared_pairs)
                
                # Update tracking variables
                all_matches.extend(level_matches)
                total_block_comparisons += level_comparisons
                compared_pairs.update(level_compared)
                
                # Log progress
                logger.info(f"[{self.name}] Found {len(level_matches)} matches from {level} confidence blocks")
                logger.info(f"[{self.name}] Total matches so far: {len(all_matches)}")
        else:
            # Process all blocks at once
            block_matches, block_comparisons, block_compared = self._process_blocks_parallel(df, filtered_blocks, compared_pairs)
            
            # Update tracking variables
            all_matches.extend(block_matches)
            total_block_comparisons += block_comparisons
            compared_pairs.update(block_compared)
        
        # Step 3: Add additional high-confidence feature combinations
        if len(df) < 15000:  # Only for moderate-sized datasets
            logger.info(f"[{self.name}] Finding additional matches using feature combinations...")
            combo_matches = self._find_matches_with_feature_combinations(df, compared_pairs)
            
            # Update tracking variables
            all_matches.extend(combo_matches)
            logger.info(f"[{self.name}] Found {len(combo_matches)} additional matches from feature combinations")
        
        # Update summary stats
        self.total_comparisons = total_block_comparisons
        self.total_matches = len(all_matches)
        
        # Log summary
        logger.info(f"[{self.name}] Total: {self.total_matches} matches from {self.total_comparisons:,} comparisons")
        logger.info(f"[{self.name}] Efficiency: Performed {self.total_comparisons:,} comparisons out of "
                  f"{len(df) * (len(df) - 1) // 2:,} possible ({self.total_comparisons/(len(df)*(len(df)-1)/2)*100:.4f}%)")
        
        return all_matches
    
    def _find_matches(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Find matching entities using standard blocking for smaller datasets.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched record pairs
        """
        logger.info(f"[{self.name}] Finding matches using deterministic features...")
        
        matches = []
        compared_pairs = set()
        
        # Step 1: Process deterministic blocks for efficient initial matching
        deterministic_blocks = self._create_deterministic_blocks(df)
        block_matches, block_comparisons, block_compared = self._process_blocks_parallel(df, deterministic_blocks, compared_pairs)
        
        # Add block matches and track compared pairs
        matches.extend(block_matches)
        compared_pairs.update(block_compared)
        self.total_comparisons += block_comparisons
        
        logger.info(f"[{self.name}] Found {len(block_matches)} matches from deterministic blocks")
        
        # Step 2: Perform systematic comparison of remaining pairs if the dataset is small enough
        if len(df) <= 2000:  # Only for small datasets
            total_possible = len(df) * (len(df) - 1) // 2
            remaining = total_possible - len(compared_pairs)
            
            # Set the maximum comparisons if not already set
            if self.max_comparisons is None:
                self.max_comparisons = remaining
            else:
                self.max_comparisons = min(self.max_comparisons, remaining)
            
            if self.max_comparisons > 0:
                logger.info(f"[{self.name}] Performing systematic comparison of up to {self.max_comparisons:,} remaining pairs")
                
                # Create batches for parallel processing
                remaining_pairs = []
                for i in range(len(df)):
                    for j in range(i+1, len(df)):
                        if (i, j) not in compared_pairs and len(remaining_pairs) < self.max_comparisons:
                            remaining_pairs.append((i, j))
                
                # Process in parallel batches
                if remaining_pairs:
                    batch_matches = self._process_pairs_parallel(df, remaining_pairs)
                    matches.extend(batch_matches)
                    self.total_comparisons += len(remaining_pairs)
                    
                    logger.info(f"[{self.name}] Found {len(batch_matches)} additional matches from systematic comparison")
        
        self.total_matches = len(matches)
        logger.info(f"[{self.name}] Total: {self.total_matches} matches from {self.total_comparisons:,} comparisons")
        
        return matches
    
    def _create_deterministic_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create blocks based on deterministic features for efficient matching.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        blocks = defaultdict(list)
        
        # Website/domain exact match
        if 'normalized_domain' in df.columns:
            domain_groups = df[df['normalized_domain'] != ''].groupby('normalized_domain').indices
            for domain, indices in domain_groups.items():
                if domain and len(indices) > 1:
                    blocks[f"domain_{domain}"] = list(indices)
        
        # Phone number exact match
        if 'normalized_phone' in df.columns:
            df_with_phone = df[df['normalized_phone'] != ''].copy()
            # For phones, ensure we have at least 8 digits
            valid_phones = df_with_phone[df_with_phone['normalized_phone'].str.len() >= 8].copy()
            if not valid_phones.empty:
                phone_groups = valid_phones.groupby('normalized_phone').indices
                for phone, indices in phone_groups.items():
                    if len(indices) > 1:
                        blocks[f"phone_{phone}"] = list(indices)
                
                # Also create blocks based on the last 8 digits (handles different country codes)
                valid_phones.loc[:, 'phone_suffix'] = valid_phones['normalized_phone'].apply(lambda x: x[-8:] if len(x) >= 8 else x)
                suffix_groups = valid_phones.groupby('phone_suffix').indices
                for suffix, indices in suffix_groups.items():
                    if len(indices) > 1 and len(suffix) == 8:  # Only use full 8-digit suffixes
                        blocks[f"phone_suffix_{suffix}"] = list(indices)
        
        # Email domain exact match (business domains, not generic)
        if 'primary_email' in df.columns:
            # Extract email domains
            df['email_domain'] = df['primary_email'].fillna('').apply(
                lambda x: x.split('@')[-1] if x and '@' in x else ""
            )
            # Filter out common personal email domains
            common_domains = {'gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'aol.com', 'icloud.com'}
            business_emails = df[(df['email_domain'] != '') & (~df['email_domain'].isin(common_domains))]
            
            if not business_emails.empty:
                email_groups = business_emails.groupby('email_domain').indices
                for domain, indices in email_groups.items():
                    if len(indices) > 1:
                        blocks[f"email_{domain}"] = list(indices)
        
        # Social media profile exact match
        for social_field in ['facebook_url', 'twitter_url', 'linkedin_url']:
            if social_field in df.columns:
                social_df = df[df[social_field].notna() & (df[social_field] != '')].copy()
                if not social_df.empty:
                    # Extract just the handle/ID part to handle URL variations
                    social_df.loc[:, 'handle'] = social_df[social_field].apply(self._extract_social_handle)
                    handle_groups = social_df[social_df['handle'] != ''].groupby('handle').indices
                    for handle, indices in handle_groups.items():
                        if len(indices) > 1:
                            blocks[f"social_{social_field}_{handle}"] = list(indices)
        
        # Exact name match (only for distinctive names)
        if 'normalized_name' in df.columns:
            df_with_name = df[(df['normalized_name'] != '') & (df['normalized_name'].str.len() > 5)]
            if not df_with_name.empty:
                name_groups = df_with_name.groupby('normalized_name').indices
                for name, indices in name_groups.items():
                    if len(indices) > 1 and len(name.split()) > 1:  # Multi-word names only
                        blocks[f"name_exact_{name}"] = list(indices)
        
        # Create combined blocks for higher precision
        # Name + Country blocks
        if 'normalized_name' in df.columns and 'main_country_code' in df.columns:
            df_with_both = df[(df['normalized_name'] != '') & (df['main_country_code'] != '')].copy()
            if not df_with_both.empty:
                # Use name prefix (first 3 chars) + country
                df_with_both.loc[:, 'name_prefix'] = df_with_both['normalized_name'].apply(lambda x: x[:3] if len(x) >= 3 else x)
                prefix_country_groups = df_with_both.groupby(['name_prefix', 'main_country_code']).indices
                for (prefix, country), indices in prefix_country_groups.items():
                    if len(indices) > 1 and len(prefix) == 3:
                        blocks[f"prefix_country_{prefix}_{country}"] = list(indices)
        
        # Count total potential comparisons
        total_comparisons = 0
        for indices in blocks.values():
            total_comparisons += len(indices) * (len(indices) - 1) // 2
        
        logger.info(f"[{self.name}] Created {len(blocks)} deterministic blocks with {total_comparisons:,} potential comparisons")
        return blocks
    
    
    def _high_precision_matching(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Perform high-precision deterministic matching - focusing only on highly reliable signals.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of matched pairs
        """
        logger.info(f"[{self.name}] Running high-precision deterministic matching")
        matches = []
        
        # Track exact match blocks for various fields
        exact_match_blocks = {
            'domain': defaultdict(list),
            'phone': defaultdict(list),
            'email': defaultdict(list),
            'name_and_location': defaultdict(list),
            'social_media': defaultdict(list)
        }
        
        # 1. Identify exact matches on key fields
        # These fields by themselves are strong indicators of duplicate entities
        
        # Domain exact matches (non-empty, distinctive domains)
        if 'normalized_domain' in df.columns:
            for idx, domain in df['normalized_domain'].items():
                if domain and len(domain) > 5:  # Skip short/empty domains
                    exact_match_blocks['domain'][domain].append(idx)
        
        # Phone exact matches (complete numbers only)
        if 'normalized_phone' in df.columns:
            for idx, phone in df['normalized_phone'].items():
                if phone and len(phone) >= 8:  # Only use longer phone numbers
                    exact_match_blocks['phone'][phone].append(idx)
        
        # Business email domain exact matches (not personal emails)
        if 'primary_email' in df.columns:
            for idx, email in df['primary_email'].items():
                if email and '@' in email:
                    email_domain = email.split('@')[-1].lower()
                    # Skip common personal email domains
                    if email_domain not in {'gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'aol.com', 'icloud.com'}:
                        exact_match_blocks['email'][email_domain].append(idx)
        
        # Social media profile exact matches
        for social_field in ['facebook_url', 'twitter_url', 'linkedin_url']:
            if social_field in df.columns:
                for idx, url in df[social_field].items():
                    if url:
                        # Extract social handle for more precise matching
                        handle = self._extract_social_handle(url)
                        if handle and len(handle) > 3:  # Skip very short handles
                            exact_match_blocks['social_media'][f"{social_field}_{handle}"].append(idx)
        
        # Name + location combinations (highly specific)
        if 'normalized_name' in df.columns and 'main_country_code' in df.columns:
            for idx, row in df.iterrows():
                name = row.get('normalized_name', '')
                country = row.get('main_country_code', '')
                region = row.get('main_region', '')
                
                if name and country and len(name) >= 8:
                    # Only use longer, more distinctive names
                    if len(name.split()) >= 2:  # Multi-word names
                        key = f"{name}|{country}"
                        if region:
                            key += f"|{region}"
                        exact_match_blocks['name_and_location'][key].append(idx)
        
        # 2. Create pairs from blocks
        total_comparisons = 0
        potential_matches = []
        
        for field, blocks in exact_match_blocks.items():
            field_matches = 0
            
            for key, indices in blocks.items():
                if len(indices) > 1:
                    # Process pairs within this block
                    # Limited to reasonable size to prevent quadratic explosion
                    if len(indices) <= 200:
                        for i in range(len(indices)):
                            for j in range(i+1, len(indices)):
                                idx1, idx2 = indices[i], indices[j]
                                
                                # Add to matches with confidence based on field
                                confidence = {
                                    'domain': 0.95,
                                    'phone': 0.95,
                                    'email': 0.9,
                                    'name_and_location': 0.85,
                                    'social_media': 0.9
                                }.get(field, 0.85)
                                
                                potential_matches.append((idx1, idx2, field, key, confidence))
                                total_comparisons += 1
                    else:
                        # For large blocks, sample pairs to avoid quadratic explosion
                        logger.info(f"[{self.name}] Large {field} block '{key}' with {len(indices)} records - sampling pairs")
                        
                        # Set a limit for number of comparisons from this block
                        max_pairs = min(1000, len(indices) * 2)
                        pairs_added = 0
                        tries = 0
                        already_added = set()
                        
                        while pairs_added < max_pairs and tries < max_pairs * 3:
                            tries += 1
                            # Randomly sample two indices
                            i, j = np.random.choice(len(indices), 2, replace=False)
                            idx1, idx2 = indices[i], indices[j]
                            
                            # Ensure we haven't already added this pair
                            pair = (min(idx1, idx2), max(idx1, idx2))
                            if pair not in already_added:
                                # Add to matches with slightly lower confidence (due to sampling)
                                confidence = {
                                    'domain': 0.9,
                                    'phone': 0.9,
                                    'email': 0.85,
                                    'name_and_location': 0.8,
                                    'social_media': 0.85
                                }.get(field, 0.8)
                                
                                potential_matches.append((idx1, idx2, field, key, confidence))
                                already_added.add(pair)
                                pairs_added += 1
                                total_comparisons += 1
            
            logger.info(f"[{self.name}] Found {len(indices)} potential matches from {field} exact matches")
        
        # 3. Validate matches with additional checks
        logger.info(f"[{self.name}] Validating {len(potential_matches)} potential matches")
        valid_matches = []
        
        for idx1, idx2, field, key, confidence in potential_matches:
            # Extract record data
            rec1 = self._extract_record_data(df.iloc[idx1])
            rec2 = self._extract_record_data(df.iloc[idx2])
            
            # Skip if either record is missing key data
            if not rec1 or not rec2:
                continue
            
            is_valid = False
            match_reason = f"{field} exact match: {key}"
            
            # Validate based on additional criteria specific to the match type
            if field == 'domain':
                # Domain matches need name similarity or other supporting evidence
                name_sim = 0.0
                if rec1.get('normalized_name') and rec2.get('normalized_name'):
                    name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                
                # Valid if names are somewhat similar or same country
                is_valid = (name_sim > 0.7 or 
                           (rec1.get('main_country_code') and rec2.get('main_country_code') and
                            rec1['main_country_code'] == rec2['main_country_code']))
                
                if name_sim > 0.7:
                    match_reason += f" with name similarity {name_sim:.2f}"
                elif is_valid:
                    match_reason += f" with same country"
            
            elif field == 'phone':
                # Phone matches always valid, but check if same country for higher confidence
                is_valid = True
                
                if (rec1.get('main_country_code') and rec2.get('main_country_code') and
                    rec1['main_country_code'] == rec2['main_country_code']):
                    confidence = min(0.98, confidence + 0.03)
                    match_reason += " with same country"
            
            elif field == 'email':
                # Email domain matches need name similarity
                name_sim = 0.0
                if rec1.get('normalized_name') and rec2.get('normalized_name'):
                    name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                
                # Valid if names are somewhat similar
                is_valid = name_sim > 0.5
                
                if is_valid:
                    match_reason += f" with name similarity {name_sim:.2f}"
            
            elif field == 'name_and_location':
                # Name + location matches are valid if names are highly similar
                is_valid = True  # Already matched on name+location
                
                # Check for other supporting evidence
                if (rec1.get('normalized_domain') and rec2.get('normalized_domain') and
                    rec1['normalized_domain'] == rec2['normalized_domain']):
                    confidence = min(0.98, confidence + 0.05)
                    match_reason += " with same domain"
            
            elif field == 'social_media':
                # Social media matches need supporting evidence
                name_sim = 0.0
                if rec1.get('normalized_name') and rec2.get('normalized_name'):
                    name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                
                # Valid if names are somewhat similar or same country
                is_valid = (name_sim > 0.5 or 
                           (rec1.get('main_country_code') and rec2.get('main_country_code') and
                            rec1['main_country_code'] == rec2['main_country_code']))
                
                if name_sim > 0.5:
                    match_reason += f" with name similarity {name_sim:.2f}"
                elif is_valid:
                    match_reason += f" with same country"
            
            # Add if valid
            if is_valid:
                valid_matches.append((idx1, idx2, confidence))
        
        self.total_comparisons = total_comparisons
        self.total_matches = len(valid_matches)
        
        logger.info(f"[{self.name}] Found {len(valid_matches)} valid matches from {total_comparisons} comparisons")
        
        return valid_matches    
    
    
    def _create_special_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create special blocks for high-probability matches using combinations of features.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        blocks = defaultdict(list)
        
        # Combination 1: Same name + country + business category
        if all(col in df.columns for col in ['normalized_name', 'main_country_code', 'naics_primary_code']):
            # Group by name and country first
            name_country_codes = {}
            
            for idx, row in df.iterrows():
                name = row.get('normalized_name', '')
                country = row.get('main_country_code', '')
                code = row.get('naics_primary_code', '')
                
                if name and country and code and len(name) > 5:
                    key = f"{name.lower()}|{country}|{code}"
                    if key not in name_country_codes:
                        name_country_codes[key] = []
                    name_country_codes[key].append(idx)
            
            # Create blocks for groups with more than one record
            for key, indices in name_country_codes.items():
                if len(indices) > 1:
                    blocks[f"special_name_country_code_{hash(key) % 10000}"] = indices
        
        # Combination 2: Similar names by sorting
        if 'normalized_name' in df.columns:
            names_by_sorted_chars = {}
            for idx, row in df.iterrows():
                name = row.get('normalized_name', '')
                if name and len(name) > 7:
                    # Create a sorted character key to find anagrams and similar names
                    sorted_key = ''.join(sorted(name.replace(' ', '')))
                    if len(sorted_key) >= 7:  # Only use longer names to avoid false positives
                        if sorted_key not in names_by_sorted_chars:
                            names_by_sorted_chars[sorted_key] = []
                        names_by_sorted_chars[sorted_key].append(idx)
            
            # Create blocks for groups with more than one record
            for key, indices in names_by_sorted_chars.items():
                if len(indices) > 1:
                    blocks[f"special_sorted_name_{hash(key) % 10000}"] = indices
        
        # Combination 3: Name prefix + phone area code
        if 'normalized_name' in df.columns and 'normalized_phone' in df.columns:
            name_phone_blocks = {}
            for idx, row in df.iterrows():
                name = row.get('normalized_name', '')
                phone = row.get('normalized_phone', '')
                
                if name and phone and len(name) >= 5 and len(phone) >= 6:
                    # Use beginning of name and area code of phone
                    name_prefix = name[:5].lower()
                    phone_prefix = phone[:3] if len(phone) >= 6 else phone
                    
                    key = f"{name_prefix}|{phone_prefix}"
                    if key not in name_phone_blocks:
                        name_phone_blocks[key] = []
                    name_phone_blocks[key].append(idx)
            
            # Create blocks for groups with more than one record
            for key, indices in name_phone_blocks.items():
                if len(indices) > 1 and len(indices) <= self.block_size_limit:
                    blocks[f"special_name_phone_{hash(key) % 10000}"] = indices
        
        logger.info(f"[{self.name}] Created {len(blocks)} special feature blocks")
        return blocks


    
    
    def _extract_social_handle(self, url: str) -> str:
        """Extract the handle/ID part from a social media URL."""
        if not url or not isinstance(url, str):
            return ""
        
        # Extract the last part of the path
        try:
            # Remove query parameters
            clean_url = url.split('?')[0].rstrip('/')
            # Get the last part of the path
            parts = clean_url.split('/')
            if len(parts) > 1:
                return parts[-1].lower()
        except:
            pass
        
        return ""
    
    def _create_name_token_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create blocks based on name tokens (words in company names).
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        blocks = defaultdict(list)
        
        if 'normalized_name' not in df.columns:
            return blocks
        
        # Extract records with valid names
        df_with_name = df[(df['normalized_name'] != '') & (df['normalized_name'].str.len() > 3)]
        if df_with_name.empty:
            return blocks
        
        # Create blocks based on individual tokens in names
        names_by_index = {}
        for idx, name in df_with_name['normalized_name'].items():
            names_by_index[idx] = name
        
        # Common words to exclude
        common_words = {'and', 'the', 'of', 'in', 'for', 'on', 'by', 'with', 'to', 'a', 'an', 'inc', 'llc', 'ltd', 'corp'}
        
        # Create token-based blocks
        token_to_indices = defaultdict(list)
        for idx, name in names_by_index.items():
            tokens = set(word.lower() for word in name.split() if len(word) > 3 and word.lower() not in common_words)
            for token in tokens:
                token_to_indices[token].append(idx)
        
        # Filter to tokens that appear in multiple records but not too many
        for token, indices in token_to_indices.items():
            if 1 < len(indices) <= self.block_size_limit:
                blocks[f"token_{token}"] = indices
        
        logger.info(f"[{self.name}] Created {len(blocks)} name token blocks")
        return blocks
    
    def _create_ngram_blocks(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create blocks based on n-grams of company names for fuzzy matching.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        blocks = defaultdict(list)
        
        if 'normalized_name' not in df.columns or len(df) > 10000:
            return blocks  # Skip for larger datasets
        
        # Extract records with valid names
        df_with_name = df[(df['normalized_name'] != '') & (df['normalized_name'].str.len() > 5)]
        if df_with_name.empty:
            return blocks
        
        # Create blocks based on character n-grams for fuzzy matching
        names_by_index = {}
        for idx, name in df_with_name['normalized_name'].items():
            names_by_index[idx] = name
        
        # Generate n-grams for each name
        n = 3  # trigrams
        ngram_to_indices = defaultdict(list)
        
        for idx, name in names_by_index.items():
            # Generate n-grams
            name_ngrams = set()
            for i in range(len(name) - n + 1):
                ngram = name[i:i+n]
                if ngram.strip():
                    name_ngrams.add(ngram)
            
            # Use selected n-grams to create blocks
            # Select a subset of n-grams to avoid too many blocks
            selected_ngrams = list(name_ngrams)[:5]  # Limit to 5 n-grams per name
            for ngram in selected_ngrams:
                ngram_to_indices[ngram].append(idx)
        
        # Filter to n-grams that appear in multiple records but not too many
        for ngram, indices in ngram_to_indices.items():
            if 1 < len(indices) <= 100:  # Smaller limit for n-gram blocks
                blocks[f"ngram_{ngram}"] = indices
        
        logger.info(f"[{self.name}] Created {len(blocks)} n-gram blocks")
        return blocks
    
    def _filter_blocks(self, blocks: Dict[str, List[int]], df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Filter and refine blocks to prevent quadratic explosion.
        
        Args:
            blocks: Dictionary mapping block keys to record indices
            df: Preprocessed dataframe
            
        Returns:
            Filtered dictionary of blocks
        """
        # Find oversized blocks
        large_blocks = {k: v for k, v in blocks.items() if len(v) > self.block_size_limit}
        
        if not large_blocks:
            return blocks
        
        filtered_blocks = {k: v for k, v in blocks.items() if len(v) <= self.block_size_limit}
        
        # Process each large block
        for block_key, indices in large_blocks.items():
            # If it's a deterministic block (high confidence), use it but sample
            if block_key.startswith(('domain_', 'phone_', 'email_', 'social_')):
                # For deterministic blocks, we can safely sample record pairs
                if len(indices) <= 2000:
                    # If moderately large, keep it
                    filtered_blocks[block_key] = indices
                else:
                    # Sample from larger blocks
                    sample_size = min(2000, int(len(indices) * 0.2))
                    np.random.seed(42)  # For reproducibility
                    sampled_indices = np.random.choice(indices, sample_size, replace=False)
                    filtered_blocks[f"{block_key}_sampled"] = list(sampled_indices)
            
            # For name-based blocks, split by additional criteria if possible
            elif block_key.startswith(('name_', 'token_', 'ngram_')):
                # Skip extremely large blocks
                if len(indices) > 5000:
                    continue
                
                # For moderate blocks, try to split by country if available
                if len(indices) > self.block_size_limit and len(indices) <= 5000:
                    try:
                        # Get country codes for these records
                        country_codes = {}
                        for idx in indices:
                            if 'main_country_code' in df.columns:
                                country = df.iloc[idx].get('main_country_code', '')
                                if country:
                                    if country not in country_codes:
                                        country_codes[country] = []
                                    country_codes[country].append(idx)
                        
                        # Create sub-blocks by country
                        for country, country_indices in country_codes.items():
                            if len(country_indices) > 1:
                                filtered_blocks[f"{block_key}_country_{country}"] = country_indices
                    except:
                        pass
        
        # Log filtering results
        total_before = sum(len(v) for v in blocks.values())
        total_after = sum(len(v) for v in filtered_blocks.values())
        logger.info(f"[{self.name}] Block filtering: {len(blocks)} blocks with {total_before} records -> "
                  f"{len(filtered_blocks)} blocks with {total_after} records")
        
        return filtered_blocks
    
    def _process_block(self, block_info_with_data):
        """
        Process a single block to find matches.
        This needs to be a class method for multiprocessing to work.
        
        Args:
            block_info_with_data: Tuple of (block_key, indices, df_dict, threshold, already_compared)
            
        Returns:
            Tuple of (matches, compared pairs, comparison count)
        """
        block_key, indices, df_dict, threshold, already_compared = block_info_with_data
        block_type = block_key.split('_')[0]
        
        # Set initial confidence based on block type
        initial_confidence = {
            'domain': 0.8,      # Domain/website match
            'phone': 0.8,       # Phone number match
            'email': 0.8,       # Business email domain match
            'name_exact': 0.7,  # Exact name match
            'social': 0.7,      # Social media match
            'special': 0.6,     # Special feature combination
            'prefix': 0.4,      # Name prefix + country
            'token': 0.3,       # Token match
            'ngram': 0.2        # N-gram match
        }.get(block_type, 0.3)
        
        if 'special' in block_key:
            # Give higher initial confidence to special blocks
            initial_confidence = 0.6
            
        # For debug: add a trace for every block
        debug_trace = f"Processing block {block_key} with {len(indices)} records and initial confidence {initial_confidence}"
        
        local_matches = []
        local_compared = set()
        
        # Extract record data outside the inner loop for better performance
        records_data = [df_dict.get(idx, {}) for idx in indices]
        
        # Generate pairs within this block
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                pair = (min(idx1, idx2), max(idx1, idx2))
                
                # Skip if already processed
                if pair in already_compared:
                    continue
                local_compared.add(pair)
                
                # Calculate match score using pre-extracted record data
                rec1 = records_data[i]
                rec2 = records_data[j]
                if not rec1 or not rec2:
                    continue
                    
                match_score, match_reason = self._calculate_match_score(rec1, rec2, initial_confidence)
                
                # Add to matches if above threshold
                if match_score >= threshold:
                    local_matches.append((idx1, idx2, match_score))
        
        # For debug: add trace of found matches
        if local_matches:
            debug_trace += f" - Found {len(local_matches)} matches from {len(local_compared)} comparisons"
        
        return local_matches, local_compared, len(local_compared)
    
    def _process_blocks_parallel(self, df: pd.DataFrame, blocks: Dict[str, List[int]], 
                               already_compared: Set[Tuple[int, int]] = None) -> Tuple[List[Tuple[int, int, float]], int, Set[Tuple[int, int]]]:
        """
        Process blocks in parallel to find matches.
        
        Args:
            df: Preprocessed dataframe
            blocks: Dictionary mapping block keys to record indices
            already_compared: Set of already compared pairs to skip
            
        Returns:
            Tuple of (matches, total_comparisons, compared_pairs)
        """
        if not blocks:
            return [], 0, set()
        
        if already_compared is None:
            already_compared = set()
        
        # Pre-extract all record data to avoid doing it in each process
        record_data_dict = {}
        all_indices = set()
        for indices in blocks.values():
            all_indices.update(indices)
        
        # Extract record data once for all records that will be compared
        for idx in all_indices:
            record_data_dict[idx] = self._extract_record_data(df.iloc[idx])
            
        # Prepare block items with all necessary data for processing
        enriched_block_items = [
            (block_key, indices, record_data_dict, self.match_threshold, already_compared) 
            for block_key, indices in blocks.items()
        ]
        
        # Process blocks in parallel
        if self.n_jobs > 1 and len(blocks) > 1:
            try:
                with multiprocessing.Pool(processes=self.n_jobs) as pool:
                    results = pool.map(self._process_block, enriched_block_items)
            except Exception as e:
                logger.warning(f"[{self.name}] Parallel processing failed, falling back to sequential: {e}")
                results = [self._process_block(item) for item in enriched_block_items]
        else:
            # Process sequentially for small number of blocks
            results = [self._process_block(item) for item in enriched_block_items]
        
        # Combine results
        all_matches = []
        all_compared = set()
        total_comparisons = 0
        
        for matches, compared, comparisons in results:
            all_matches.extend(matches)
            all_compared.update(compared)
            total_comparisons += comparisons
        
        return all_matches, total_comparisons, all_compared
    
    def _process_pair_batch(self, batch_data):
        """
        Process a batch of pairs to find matches.
        This needs to be a class method for multiprocessing to work.
        
        Args:
            batch_data: Tuple of (batch_pairs, record_data, match_threshold)
            
        Returns:
            List of (idx1, idx2, score) tuples representing matches
        """
        batch_pairs, record_data, match_threshold = batch_data
        batch_matches = []
        
        for idx1, idx2 in batch_pairs:
            rec1 = record_data.get(idx1, {})
            rec2 = record_data.get(idx2, {})
            
            if not rec1 or not rec2:
                continue
                
            # Calculate match score
            match_score, match_reason = self._calculate_match_score(rec1, rec2)
            
            # Add to matches if above threshold
            if match_score >= match_threshold:
                batch_matches.append((idx1, idx2, match_score))
        
        return batch_matches
        
    def _process_pairs_parallel(self, df: pd.DataFrame, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """
        Process pairs in parallel to find matches.
        
        Args:
            df: Preprocessed dataframe
            pairs: List of (idx1, idx2) pairs to compare
            
        Returns:
            List of (idx1, idx2, score) tuples representing matches
        """
        if not pairs:
            return []
        
        # Extract all record data once
        record_data = {idx: self._extract_record_data(df.iloc[idx]) for idx in set(idx for pair in pairs for idx in pair)}
        
        # Split into batches for parallel processing
        batch_size = max(100, len(pairs) // (self.n_jobs * 10))  # Ensure enough batches for parallelization
        batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
        
        # Prepare batches with necessary data
        batch_data = [(batch, record_data, self.match_threshold) for batch in batches]
        
        # Process batches in parallel
        if self.n_jobs > 1 and len(batches) > 1:
            try:
                with multiprocessing.Pool(processes=self.n_jobs) as pool:
                    batch_results = pool.map(self._process_pair_batch, batch_data)
            except Exception as e:
                logger.warning(f"[{self.name}] Parallel processing failed, falling back to sequential: {e}")
                batch_results = [self._process_pair_batch(data) for data in batch_data]
        else:
            # Process sequentially for small number of batches
            batch_results = [self._process_pair_batch(data) for data in batch_data]
        
        # Combine results
        all_matches = []
        for matches in batch_results:
            all_matches.extend(matches)
        
        return all_matches
    
    def _find_matches_with_feature_combinations(self, df: pd.DataFrame, compared_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """
        Find additional matches using combinations of features.
        
        This method targets specific feature combinations that might indicate matches
        but wouldn't be captured by block-based processing.
        
        Args:
            df: Preprocessed dataframe
            compared_pairs: Set of already compared pairs to skip
            
        Returns:
            List of (idx1, idx2, score) tuples representing matches
        """
        matches = []
        additional_pairs = []
        
        # Similar name + same country, only for smaller datasets
        if len(df) < 10000 and 'normalized_name' in df.columns and 'main_country_code' in df.columns:
            # Group by country first to reduce search space
            country_groups = df[df['main_country_code'] != ''].groupby('main_country_code').indices
            
            for country, country_indices in country_groups.items():
                if len(country_indices) > 500:
                    # For large country groups, sample
                    np.random.seed(42)
                    country_indices = np.random.choice(country_indices, 500, replace=False)
                
                # Find pairs with similar names within same country
                name_pairs = []
                name_by_idx = {idx: df.iloc[idx]['normalized_name'] for idx in country_indices if df.iloc[idx]['normalized_name']}
                
                # Sort by name to make it more efficient
                sorted_idxs = sorted(name_by_idx.keys(), key=lambda idx: name_by_idx[idx])
                
                # Compare only with nearby names in sorted order
                window_size = min(20, len(sorted_idxs))
                for i, idx1 in enumerate(sorted_idxs):
                    name1 = name_by_idx[idx1]
                    for j in range(i+1, min(i+window_size, len(sorted_idxs))):
                        idx2 = sorted_idxs[j]
                        name2 = name_by_idx[idx2]
                        
                        # Skip if already compared
                        pair = (min(idx1, idx2), max(idx1, idx2))
                        if pair in compared_pairs:
                            continue
                        
                        # Quick initial check to save computation
                        if abs(len(name1) - len(name2)) <= 5:
                            name_pairs.append(pair)
                
                additional_pairs.extend(name_pairs)
        
        # Process additional pairs in parallel
        if additional_pairs:
            logger.info(f"[{self.name}] Processing {len(additional_pairs)} additional pairs from feature combinations")
            combo_matches = self._process_pairs_parallel(df, additional_pairs)
            matches.extend(combo_matches)
        
        return matches
    
    def _calculate_match_score(self, rec1: Dict[str, Any], rec2: Dict[str, Any], initial_confidence=0.0) -> Tuple[float, str]:
        """
        Calculate a match score between two records based on feature similarity.
        
        Args:
            rec1: First record data
            rec2: Second record data
            initial_confidence: Starting confidence value from deterministic blocking
            
        Returns:
            Tuple of (match_score, match_reason)
        """
        # Track individual feature scores and reasons
        feature_scores = {}
        match_reasons = []
        requires_secondary_evidence = True  # Most matches require secondary confirmation
        
        # Start with initial confidence if provided
        cumulative_score = initial_confidence
        
        # === DETERMINISTIC FEATURES ===
        
        # Website domain match
        if rec1.get('normalized_domain') and rec2.get('normalized_domain'):
            if rec1['normalized_domain'] == rec2['normalized_domain'] and len(rec1['normalized_domain']) > 5:
                feature_scores['website_exact'] = self.feature_weights['website_exact']
                match_reasons.append(f"same website: {rec1['normalized_domain']}")
                requires_secondary_evidence = False  # Strong signal may not need secondary confirmation
        
        # Phone number match
        if rec1.get('normalized_phone') and rec2.get('normalized_phone') and len(rec1['normalized_phone']) >= 8:
            if rec1['normalized_phone'] == rec2['normalized_phone']:
                feature_scores['phone_exact'] = self.feature_weights['phone_exact']
                match_reasons.append(f"same phone: {rec1['normalized_phone']}")
                requires_secondary_evidence = False  # Strong signal may not need secondary confirmation
            # Partial phone match (one contained in the other)
            elif (len(rec1['normalized_phone']) >= 8 and len(rec2['normalized_phone']) >= 8 and
                  (rec1['normalized_phone'] in rec2['normalized_phone'] or rec2['normalized_phone'] in rec1['normalized_phone'])):
                feature_scores['phone_partial'] = self.feature_weights['phone_partial']
                match_reasons.append("partial phone match")
        
        # Email domain match
        if 'primary_email' in rec1 and 'primary_email' in rec2:
            email1 = rec1.get('primary_email', '')
            email2 = rec2.get('primary_email', '')
            if email1 and email2 and '@' in email1 and '@' in email2:
                domain1 = email1.split('@')[-1].lower()
                domain2 = email2.split('@')[-1].lower()
                # Skip common personal email domains
                common_domains = {'gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'aol.com', 'icloud.com'}
                if domain1 and domain2 and domain1 == domain2 and domain1 not in common_domains:
                    feature_scores['email_domain_exact'] = self.feature_weights['email_domain_exact']
                    match_reasons.append(f"same email domain: {domain1}")
                    requires_secondary_evidence = False  # Strong signal for business domains
        
        # Social media profile match
        for social_field in ['facebook_url', 'twitter_url', 'linkedin_url']:
            if social_field in rec1 and social_field in rec2:
                url1 = rec1.get(social_field, '')
                url2 = rec2.get(social_field, '')
                if url1 and url2 and url1 == url2:
                    feature_scores['social_media_exact'] = self.feature_weights['social_media_exact']
                    match_reasons.append(f"same {social_field.split('_')[0]} profile")
                    requires_secondary_evidence = False  # Strong signal
        
        # === NAME MATCHING ===
        
        # Company name match
        name_score = 0.0
        if rec1.get('normalized_name') and rec2.get('normalized_name'):
            name1 = rec1['normalized_name']
            name2 = rec2['normalized_name']
            
            # Skip very short names (too generic)
            if len(name1) < 5 or len(name2) < 5:
                pass
            # Exact name match
            elif name1 == name2:
                name_score = self.feature_weights['name_exact']
                match_reasons.append(f"exact name match: {name1}")
                
                # For distinctive names (longer, multiple words), this can be a strong signal
                if len(name1) > 10 and len(name1.split()) > 2:
                    requires_secondary_evidence = False
            else:
                # Name similarity using token sort ratio (handles word order differences)
                name_sim = fuzz.token_sort_ratio(name1, name2) / 100.0
                
                # Very similar name (>90%)
                if name_sim >= 0.9:
                    name_score = self.feature_weights['name_very_similar'] * name_sim
                    match_reasons.append(f"very similar names ({name_sim:.2f})")
                    
                    # For distinctive names with high similarity, this can be a strong signal
                    if name_sim > 0.95 and len(name1) > 10 and len(name1.split()) > 2:
                        requires_secondary_evidence = False
        
        if name_score > 0:
            feature_scores['name_match'] = name_score
        
        # === ADDRESS MATCHING ===
        
        # Address match
        if rec1.get('normalized_address') and rec2.get('normalized_address'):
            addr1 = rec1['normalized_address']
            addr2 = rec2['normalized_address']
            
            # Skip very short addresses (too generic)
            if len(addr1) < 10 or len(addr2) < 10:
                pass
            # Exact address match
            elif addr1 == addr2:
                feature_scores['address_exact'] = self.feature_weights['address_exact']
                match_reasons.append("exact address match")
            else:
                # Address similarity using token set ratio (handles partial matches better)
                addr_sim = fuzz.token_set_ratio(addr1, addr2) / 100.0
                if addr_sim >= 0.85:  # Higher threshold for addresses
                    feature_scores['address_similar'] = self.feature_weights['address_similar'] * addr_sim
                    match_reasons.append(f"similar addresses ({addr_sim:.2f})")
        
        # === SUPPORTING FEATURES (lower weights) ===
        
        # Location match
        if rec1.get('main_country_code') and rec2.get('main_country_code'):
            if rec1['main_country_code'] == rec2['main_country_code']:
                feature_scores['same_country'] = self.feature_weights['same_country']
                
                # Same region/state
                if rec1.get('main_region') and rec2.get('main_region') and rec1['main_region'] == rec2['main_region']:
                    feature_scores['same_region'] = self.feature_weights['same_region']
                    
                    # Same city (if available)
                    if 'city' in rec1 and 'city' in rec2:
                        city1 = rec1.get('city', '')
                        city2 = rec2.get('city', '')
                        if city1 and city2 and city1.lower() == city2.lower():
                            feature_scores['same_city'] = self.feature_weights['same_city']
        
        # Business category match
        if 'naics_primary_code' in rec1 and 'naics_primary_code' in rec2:
            code1 = rec1.get('naics_primary_code', '')
            code2 = rec2.get('naics_primary_code', '')
            if code1 and code2 and code1 == code2:
                feature_scores['business_category'] = self.feature_weights['business_category']
        
        # Calculate final score based on cumulative evidence
        if feature_scores:
            # If we already have initial confidence, combine with the best additional feature
            if initial_confidence > 0:
                if feature_scores:
                    # Get the best feature score
                    best_feature = max(feature_scores.values())
                    
                    # If we need secondary evidence but don't have a good second signal, reduce score
                    if requires_secondary_evidence and len(feature_scores) < 2:
                        cumulative_score = min(initial_confidence, best_feature)
                    else:
                        # Combine initial confidence with best feature (with diminishing returns)
                        cumulative_score = initial_confidence + best_feature * 0.5
                        cumulative_score = min(0.95, cumulative_score)  # Cap at 0.95
            else:
                # Otherwise, use a stricter approach
                
                # Get the top features
                scores = sorted(feature_scores.values(), reverse=True)
                
                if len(scores) == 0:
                    cumulative_score = 0.0
                elif len(scores) == 1:
                    # Single feature - only consider it a match if it's a very strong signal
                    single_score = scores[0]
                    if requires_secondary_evidence:
                        # For features that need secondary confirmation, slightly reduce score
                        cumulative_score = single_score * 0.85  # Less reduction to allow more matches
                    else:
                        # Strong deterministic features can stand alone
                        cumulative_score = single_score
                else:
                    # Multiple features - primary signal with supporting evidence
                    primary = scores[0]
                    secondary = scores[1]
                    
                    # More secondary pieces of evidence increase confidence
                    supporting_weight = min(0.7, 0.3 * (len(scores) - 1))  # Higher weight for supporting evidence
                    supporting_score = sum(scores[1:]) * supporting_weight
                    
                    # Combine primary and supporting evidence
                    if requires_secondary_evidence:
                        # Require stronger secondary evidence but with higher weight
                        if secondary < 0.2:  # Weak secondary evidence
                            supporting_score *= 0.7  # Less reduction
                        
                        # Primary evidence with supporting evidence
                        cumulative_score = primary * 0.8 + supporting_score
                    else:
                        # Strong primary evidence with supporting evidence
                        cumulative_score = primary * 0.95 + supporting_score
                        
                # Cap at 0.95 to avoid too much confidence
                cumulative_score = min(0.95, cumulative_score)
        
        # Generate reason string
        match_reason = ", ".join(match_reasons) if match_reasons else "insufficient matching features"
        
        # Add more special case combinations that are reliable signals
        if ('website_exact' in feature_scores and feature_scores['website_exact'] > 0.85) or \
           ('phone_exact' in feature_scores and feature_scores['phone_exact'] > 0.85) or \
           ('email_domain_exact' in feature_scores and feature_scores['email_domain_exact'] > 0.85) or \
           ('social_media_exact' in feature_scores and feature_scores['social_media_exact'] > 0.85):
            
            # If one strong deterministic feature and any other supporting evidence
            if len(feature_scores) >= 2:
                # Boost the score significantly
                boost = 0.2 * min(len(feature_scores) - 1, 3)  # Up to 3 supporting features
                cumulative_score = min(0.99, cumulative_score + boost)
                match_reason = f"strong deterministic match with {len(feature_scores)-1} supporting features"
        
        # Exact name match with address or location evidence
        if 'name_match' in feature_scores and feature_scores['name_match'] > 0.8:
            address_evidence = ('address_exact' in feature_scores) or \
                              ('address_similar' in feature_scores and feature_scores['address_similar'] > 0.3)
            location_evidence = ('same_city' in feature_scores) or \
                               (('same_region' in feature_scores) and ('same_country' in feature_scores))
            
            if address_evidence or location_evidence:
                # Boost score for name + address/location
                cumulative_score = min(0.95, cumulative_score + 0.15)
                if address_evidence:
                    match_reason = "strong name match with address evidence"
                else:
                    match_reason = "strong name match with location evidence"
        
        # Apply a random small noise to prevent mass ties in clustering
        if cumulative_score > 0:
            # Add tiny random noise (±0.001) to prevent exact ties
            noise = (np.random.random() - 0.5) * 0.002
            cumulative_score = max(0, min(0.95, cumulative_score + noise))
        
        return cumulative_score, match_reason