"""
Machine Learning-based entity resolution approach.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
from rapidfuzz import fuzz

from base import EntityResolutionBase

logger = logging.getLogger("entity_resolution")

class MLEntityResolver(EntityResolutionBase):
    """Machine Learning-based entity resolution using Random Forest classification."""
    
    def __init__(self, match_threshold=0.5, training_pairs=10000, feature_selection=True):
        """
        Initialize with configurable parameters.
        
        Args:
            match_threshold: Threshold for considering a match prediction (default: 0.5)
            training_pairs: Number of pairs to generate for training (default: 10000)
            feature_selection: Whether to perform feature selection (default: True)
        """
        super().__init__(
            name="ML-based Entity Resolution",
            description="Uses a Random Forest classifier to predict matches based on feature similarity"
        )
        self.match_threshold = match_threshold
        self.training_pairs = training_pairs
        self.feature_selection = feature_selection
        self.similarity_model = None
        self.feature_importances = None
        self.selected_features = None
    
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run ML-based entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Train similarity model
        self._train_similarity_model(processed_df, num_pairs=self.training_pairs)
        
        # Create blocking keys
        blocks = self._create_blocking_keys(processed_df)
        
        # Find candidate pairs
        candidate_pairs = self._find_candidate_pairs(processed_df, blocks)
        
        # Predict matches
        matches = self._predict_matches(processed_df, candidate_pairs, threshold=self.match_threshold)
        
        # Create clusters
        cluster_mapping = self._create_clusters(processed_df, matches)
        
        # Add clusters to original DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = result_df.index.map(lambda x: cluster_mapping.get(x, x))
        
        # Create canonical records
        canonical_df = self._create_canonical_records(df, cluster_mapping)
        
        return result_df, canonical_df
    
    def _calculate_similarities(self, rec1: Dict[str, Any], rec2: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive similarity features between two records.
        
        Args:
            rec1: First record data
            rec2: Second record data
            
        Returns:
            Dictionary of similarity features
        """
        # Name similarity - multiple metrics for robustness
        name_sim = {
            'name_exact': 1.0 if rec1['normalized_name'] == rec2['normalized_name'] else 0.0,
            'name_token_sort': fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0,
            'name_token_set': fuzz.token_set_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0,
            'name_partial': fuzz.partial_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0,
            # Remove the problematic jaro_similarity function
            # 'name_jaro': fuzz.jaro_similarity(rec1['normalized_name'], rec2['normalized_name']) 
            #               if rec1['normalized_name'] and rec2['normalized_name'] else 0.0
        }
        
        # Domain similarity
        domain_sim = {
            'domain_exact': 1.0 if (rec1['normalized_domain'] and rec2['normalized_domain'] and 
                                    rec1['normalized_domain'] == rec2['normalized_domain']) else 0.0,
            'domain_partial': fuzz.partial_ratio(rec1['normalized_domain'], rec2['normalized_domain']) / 100.0 
                            if rec1['normalized_domain'] and rec2['normalized_domain'] else 0.0
        }
        
        # Address similarity
        address_sim = {
            'address_token_sort': fuzz.token_sort_ratio(rec1['normalized_address'], rec2['normalized_address']) / 100.0
                                if rec1['normalized_address'] and rec2['normalized_address'] else 0.0,
            'address_token_set': fuzz.token_set_ratio(rec1['normalized_address'], rec2['normalized_address']) / 100.0
                                if rec1['normalized_address'] and rec2['normalized_address'] else 0.0
        }
        
        # Phone similarity
        phone_sim = {
            'phone_exact': 1.0 if (rec1['normalized_phone'] and rec2['normalized_phone'] and 
                                rec1['normalized_phone'] == rec2['normalized_phone']) else 0.0,
            'phone_partial': fuzz.partial_ratio(rec1['normalized_phone'], rec2['normalized_phone']) / 100.0
                            if rec1['normalized_phone'] and rec2['normalized_phone'] else 0.0
        }
        
        # Location similarity
        location_sim = {
            'same_country': 1.0 if (rec1['main_country_code'] and rec2['main_country_code'] and 
                                    rec1['main_country_code'] == rec2['main_country_code']) else 0.0,
            'same_region': 1.0 if (rec1['main_region'] and rec2['main_region'] and 
                                rec1['main_region'] == rec2['main_region']) else 0.0
        }
        
        # Business type similarity
        business_sim = {
            'same_naics': 1.0 if (rec1['naics_primary_code'] and rec2['naics_primary_code'] and
                                rec1['naics_primary_code'] == rec2['naics_primary_code']) else 0.0,
            'business_tags_sim': fuzz.token_set_ratio(str(rec1['business_tags']), str(rec2['business_tags'])) / 100.0
                                if rec1['business_tags'] and rec2['business_tags'] else 0.0
        }
        
        # Data completeness features
        completeness = {
            'both_have_name': 1.0 if (rec1['has_name'] and rec2['has_name']) else 0.0,
            'both_have_domain': 1.0 if (rec1['has_domain'] and rec2['has_domain']) else 0.0,
            'both_have_address': 1.0 if (rec1['has_address'] and rec2['has_address']) else 0.0,
            'both_have_phone': 1.0 if (rec1['has_phone'] and rec2['has_phone']) else 0.0
        }
        
        # Email domain similarity
        email_sim = {
            'same_email_domain': 1.0 if (rec1['email_domain'] and rec2['email_domain'] and
                                    rec1['email_domain'] == rec2['email_domain']) else 0.0
        }
        
        # Combine all features
        similarities = {
            **name_sim, 
            **domain_sim, 
            **address_sim, 
            **phone_sim, 
            **location_sim, 
            **business_sim, 
            **completeness,
            **email_sim
        }
        
        return similarities
    
    def _train_similarity_model(self, df: pd.DataFrame, num_pairs=10000) -> None:
        """
        Train a machine learning model to predict record matching.
        
        Args:
            df: Preprocessed dataframe
            num_pairs: Number of pairs to use for training
        """
        logger.info(f"[{self.name}] Training similarity model...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.feature_selection import SelectFromModel
            
            # Generate training data
            pairs_df = self._generate_training_pairs(df, num_pairs)
            
            # Prepare features and labels
            feature_cols = [col for col in pairs_df.columns if col not in ['id1', 'id2', 'label']]
            X = pairs_df[feature_cols].values
            y = pairs_df['label'].values
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Create and train a random forest classifier
            clf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
            
            # Optionally perform feature selection
            if self.feature_selection:
                selector = SelectFromModel(clf, threshold='median')
                selector.fit(X_train, y_train)
                
                # Get selected features
                self.selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
                logger.info(f"[{self.name}] Selected {len(self.selected_features)} features: {', '.join(self.selected_features)}")
                
                # Retrain with selected features
                X_train_selected = selector.transform(X_train)
                X_val_selected = selector.transform(X_val)
                
                clf = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    class_weight='balanced'
                )
                clf.fit(X_train_selected, y_train)
            
            # Store feature importances
            self.feature_importances = dict(zip(feature_cols, clf.feature_importances_))
            logger.info(f"[{self.name}] Top 5 important features:")
            for feature, importance in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  - {feature}: {importance:.4f}")
            
            # Save the model
            self.similarity_model = clf
            
            # Evaluate on validation set
            if self.feature_selection:
                val_score = clf.score(X_val_selected, y_val)
            else:
                val_score = clf.score(X_val, y_val)
            logger.info(f"[{self.name}] Validation accuracy: {val_score:.4f}")
        
        except ImportError:
            logger.warning(f"[{self.name}] scikit-learn not available, using fallback rule-based matching")
            self.similarity_model = None
    
    def _generate_training_pairs(self, df: pd.DataFrame, num_pairs=10000) -> pd.DataFrame:
        """
        Generate training pairs for model training.
        
        Args:
            df: Preprocessed dataframe
            num_pairs: Number of pairs to generate
            
        Returns:
            Dataframe of record pairs with features and match labels
        """
        logger.info(f"[{self.name}] Generating {num_pairs} training pairs...")
        
        # Strategy for positive examples: exact domain matches, high name similarity, etc.
        positive_pairs = []
        
        # Get domain groups
        domain_groups = df[df['normalized_domain'] != ''].groupby('normalized_domain').indices
        for domain, indices in domain_groups.items():
            if len(indices) > 1:
                # Take up to 5 pairs from each domain group
                for _ in range(min(5, len(indices))):
                    idx1, idx2 = np.random.choice(indices, 2, replace=False)
                    positive_pairs.append((idx1, idx2, 1))  # 1 = match
        
        # Get high name similarity pairs
        name_prefix_groups = df.groupby('name_prefix').indices
        for prefix, indices in name_prefix_groups.items():
            if len(indices) > 1 and len(indices) <= 100:  # Skip very large groups
                pairs_from_group = min(10, len(indices))
                for _ in range(pairs_from_group):
                    idx1, idx2 = np.random.choice(list(indices), 2, replace=False)
                    rec1 = self._extract_record_data(df.iloc[idx1])
                    rec2 = self._extract_record_data(df.iloc[idx2])
                    name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                    
                    # If names are very similar and same country, consider it a match
                    if name_sim > 0.9 and rec1['main_country_code'] == rec2['main_country_code']:
                        positive_pairs.append((idx1, idx2, 1))
        
        # Phone number match pairs
        phone_groups = df[df['normalized_phone'] != ''].groupby('normalized_phone').indices
        for phone, indices in phone_groups.items():
            if len(indices) > 1 and len(phone) >= 10:
                pairs_from_group = min(5, len(indices))
                for _ in range(pairs_from_group):
                    idx1, idx2 = np.random.choice(list(indices), 2, replace=False)
                    positive_pairs.append((idx1, idx2, 1))
        
        # Strategy for negative examples: different domains, different countries, different name prefixes
        negative_pairs = []
        
        # Different domains
        domains = df[df['normalized_domain'] != '']['normalized_domain'].unique()
        if len(domains) >= 2:
            for _ in range(min(1000, num_pairs // 3)):
                domain1, domain2 = np.random.choice(domains, 2, replace=False)
                idx1 = np.random.choice(df[df['normalized_domain'] == domain1].index)
                idx2 = np.random.choice(df[df['normalized_domain'] == domain2].index)
                negative_pairs.append((idx1, idx2, 0))  # 0 = non-match
        
        # Different countries and name prefixes
        for _ in range(min(2000, num_pairs // 3)):
            idx1 = np.random.randint(0, len(df))
            idx2 = np.random.randint(0, len(df))
            rec1 = self._extract_record_data(df.iloc[idx1])
            rec2 = self._extract_record_data(df.iloc[idx2])
            
            # If different countries and name prefixes, likely non-match
            if (rec1['main_country_code'] != rec2['main_country_code'] and 
                rec1['normalized_name'][:3] != rec2['normalized_name'][:3]):
                negative_pairs.append((idx1, idx2, 0))
        
        # Strategy for challenging examples: moderate name similarity, same country
        challenging_pairs = []
        for _ in range(min(2000, num_pairs // 3)):
            idx1 = np.random.randint(0, len(df))
            idx2 = np.random.randint(0, len(df))
            rec1 = self._extract_record_data(df.iloc[idx1])
            rec2 = self._extract_record_data(df.iloc[idx2])
            
            name_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
            
            # Moderate name similarity (0.6-0.8)
            if 0.6 <= name_sim <= 0.8 and rec1['main_country_code'] == rec2['main_country_code']:
                # Manually determine if these are matches (50/50)
                is_match = np.random.choice([0, 1])
                challenging_pairs.append((idx1, idx2, is_match))
        
        # Combine all pairs and shuffle
        all_pairs = positive_pairs + negative_pairs + challenging_pairs
        np.random.shuffle(all_pairs)
        
        # Take the requested number of pairs
        selected_pairs = all_pairs[:num_pairs]
        
        # Calculate features for each pair
        pair_features = []
        for idx1, idx2, label in selected_pairs:
            rec1 = self._extract_record_data(df.iloc[idx1])
            rec2 = self._extract_record_data(df.iloc[idx2])
            similarities = self._calculate_similarities(rec1, rec2)
            similarities['id1'] = idx1
            similarities['id2'] = idx2
            similarities['label'] = label
            pair_features.append(similarities)
        
        return pd.DataFrame(pair_features)
    
    def _create_blocking_keys(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create efficient blocking keys to reduce comparison space.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        logger.info(f"[{self.name}] Creating blocking keys...")
        
        blocks = defaultdict(list)
        
        # 1. Name prefix + country code
        for idx, row in df.iterrows():
            prefix = row['normalized_name'][:3] if row['normalized_name'] else ''
            country = row['main_country_code'] if row['main_country_code'] else ''
            
            if prefix and country:
                blocks[f"name_country_{prefix}_{country}"].append(idx)
        
        # 2. Domain exact match (strong signal)
        df_with_domain = df[df['normalized_domain'] != '']
        domain_groups = df_with_domain.groupby('normalized_domain').indices
        for domain, indices in domain_groups.items():
            if domain:  # Skip empty domains
                blocks[f"domain_{domain}"] = list(indices)
        
        # 3. Phone number exact match (strong signal)
        df_with_phone = df[df['normalized_phone'] != '']
        phone_groups = df_with_phone.groupby('normalized_phone').indices
        for phone, indices in phone_groups.items():
            if phone and len(phone) >= 10:  # Only use full phone numbers
                blocks[f"phone_{phone}"] = list(indices)
        
        # 4. Industry/business tags for companies in same region
        if 'naics_2022_primary_code' in df.columns:
            # Extract major industry from NAICS code
            df['major_industry'] = df['naics_2022_primary_code'].astype(str).apply(
                lambda x: x[:2] if len(x) >= 2 else ''
            )
            
            # Group by industry and region
            industry_region_groups = df[
                (df['major_industry'] != '') & (df['main_region'] != '')
            ].groupby(['major_industry', 'main_region']).indices
            
            for (industry, region), indices in industry_region_groups.items():
                if len(indices) <= 1000:  # Skip overly large blocks
                    blocks[f"industry_region_{industry}_{region}"] = list(indices)
        
        # Filter out large blocks to avoid quadratic explosion
        filtered_blocks = {k: v for k, v in blocks.items() if len(v) <= 1000 and len(v) > 1}
        
        logger.info(f"[{self.name}] Created {len(filtered_blocks)} blocking keys")
        
        # Log summary stats
        block_sizes = [len(indices) for indices in filtered_blocks.values()]
        if block_sizes:
            logger.info(f"  - Average block size: {np.mean(block_sizes):.1f}")
            logger.info(f"  - Max block size: {np.max(block_sizes)}")
            logger.info(f"  - Total pairs to compare: {sum(len(v) * (len(v) - 1) // 2 for v in filtered_blocks.values())}")
        
        return filtered_blocks
    
    def _find_candidate_pairs(self, df: pd.DataFrame, blocks: Dict[str, List[int]], max_pairs=1000000) -> Set[Tuple[int, int]]:
        """
        Find candidate pairs using blocking.
        
        Args:
            df: Preprocessed dataframe
            blocks: Dictionary mapping block keys to record indices
            max_pairs: Maximum number of pairs to consider
            
        Returns:
            Set of tuples (idx1, idx2) representing candidate pairs
        """
        logger.info(f"[{self.name}] Finding candidate pairs...")
        
        candidate_pairs = set()
        total_comparisons = 0
        
        # Count total potential comparisons
        potential_comparisons = sum(len(indices) * (len(indices) - 1) // 2 for indices in blocks.values())
        logger.info(f"[{self.name}] Potential comparisons from blocks: {potential_comparisons}")
        
        # Process each block
        for block_key, indices in blocks.items():
            # Skip if too many potential comparisons
            if len(indices) > 100:  # Skip very large blocks
                continue
                
            # Generate pairs within this block
            block_pairs = set()
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    # Ensure consistent ordering
                    pair = tuple(sorted([indices[i], indices[j]]))
                    block_pairs.add(pair)
            
            # Add pairs to candidates
            candidate_pairs.update(block_pairs)
            total_comparisons += len(block_pairs)
            
            # Check if we've exceeded the maximum
            if total_comparisons >= max_pairs:
                logger.info(f"[{self.name}] Reached maximum number of pairs ({max_pairs})")
                break
        
        logger.info(f"[{self.name}] Found {len(candidate_pairs)} candidate pairs")
        return candidate_pairs
    
    def _predict_matches(self, df: pd.DataFrame, candidate_pairs: Set[Tuple[int, int]], threshold=0.5) -> List[Tuple[int, int, float]]:
        """
        Predict matches among candidate pairs.
        
        Args:
            df: Preprocessed dataframe
            candidate_pairs: Set of candidate pairs to evaluate
            threshold: Threshold for match prediction
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched pairs
        """
        logger.info(f"[{self.name}] Predicting matches with threshold {threshold}...")
        
        matches = []
        
        # Batch processing to avoid memory issues
        batch_size = 10000
        pairs_list = list(candidate_pairs)
        
        for i in range(0, len(pairs_list), batch_size):
            batch_pairs = pairs_list[i:i+batch_size]
            batch_features = []
            
            # Calculate features for each pair
            for idx1, idx2 in batch_pairs:
                rec1 = self._extract_record_data(df.iloc[idx1])
                rec2 = self._extract_record_data(df.iloc[idx2])
                similarities = self._calculate_similarities(rec1, rec2)
                batch_features.append(similarities)
            
            # Convert to DataFrame
            batch_df = pd.DataFrame(batch_features)
            
            if self.similarity_model:
                try:
                    # Get feature columns in the correct order
                    feature_cols = list(self.feature_importances.keys())
                    
                    # Handle missing columns
                    for col in feature_cols:
                        if col not in batch_df.columns:
                            batch_df[col] = 0.0
                    
                    # Select features if feature selection was applied
                    if self.feature_selection and self.selected_features:
                        X = batch_df[self.selected_features].values
                    else:
                        X = batch_df[feature_cols].values
                    
                    # Predict match probability
                    proba = self.similarity_model.predict_proba(X)[:, 1]  # Probability of class 1 (match)
                    
                    # Add matches above threshold
                    for (idx1, idx2), prob in zip(batch_pairs, proba):
                        if prob >= threshold:
                            matches.append((idx1, idx2, float(prob)))
                except Exception as e:
                    logger.error(f"[{self.name}] Error in prediction: {e}, falling back to rule-based")
                    # Fallback to rule-based scoring
                    self._fallback_rule_based_prediction(batch_df, batch_pairs, matches, threshold)
            else:
                # Fallback if no model: use weighted combination of features
                self._fallback_rule_based_prediction(batch_df, batch_pairs, matches, threshold)
        
        logger.info(f"[{self.name}] Found {len(matches)} matches")
        return matches
    
    def _fallback_rule_based_prediction(self, batch_df, batch_pairs, matches, threshold):
        """Fallback method for prediction when ML model is not available."""
        for j, (idx1, idx2) in enumerate(batch_pairs):
            # High weights to exact matches, moderate to fuzzy matches
            weights = {
                'name_exact': 0.4, 'name_token_sort': 0.2, 'name_token_set': 0.1,
                'domain_exact': 0.2, 'address_token_sort': 0.05, 'phone_exact': 0.05
            }
            # Calculate weighted score
            score = 0.0
            for k, w in weights.items():
                if k in batch_df.columns:
                    score += batch_df.iloc[j].get(k, 0) * w
            
            if score >= threshold:
                matches.append((idx1, idx2, score))