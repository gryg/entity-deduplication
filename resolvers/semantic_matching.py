"""
Semantic matching-based entity resolution approach.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from rapidfuzz import fuzz

from base import EntityResolutionBase

logger = logging.getLogger("entity_resolution")

class SemanticMatchingResolver(EntityResolutionBase):
    """Enhanced semantic matching-based entity resolution using name embeddings."""
    
    def __init__(self, similarity_threshold=0.7, use_tfidf=True):
        """
        Initialize with configurable parameters.
        
        Args:
            similarity_threshold: Threshold for considering a match (default: 0.7)
            use_tfidf: Whether to use TF-IDF for word importance (default: True)
        """
        super().__init__(
            name="Semantic Matching Entity Resolution",
            description="Uses semantic name embeddings and similarity to identify duplicates"
        )
        self.similarity_threshold = similarity_threshold
        self.use_tfidf = use_tfidf
        self.name_embeddings = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.word_to_idx = {}
        
    def _run_resolution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run semantic matching-based entity resolution process.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (result_df, canonical_df)
        """
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Generate name embeddings
        self._generate_name_embeddings(processed_df)
        
        # Find semantically similar pairs
        matches = self._find_semantic_matches(processed_df)
        
        # Create clusters
        cluster_mapping = self._create_clusters(processed_df, matches)
        
        # Add clusters to original DataFrame
        result_df = df.copy()
        result_df['cluster_id'] = result_df.index.map(lambda x: cluster_mapping.get(x, x))
        
        # Create canonical records
        canonical_df = self._create_canonical_records(df, cluster_mapping)
        
        return result_df, canonical_df
    
    def _generate_name_embeddings(self, df: pd.DataFrame) -> None:
        """
        Generate embeddings for company names using simple but effective techniques.
        
        Args:
            df: Preprocessed dataframe
        """
        logger.info(f"[{self.name}] Generating name embeddings...")
        
        # Get non-empty company names
        valid_records = []
        valid_names = []
        
        for idx, row in df.iterrows():
            name = row['normalized_name']
            if name and len(name.strip()) > 0:
                valid_records.append(idx)
                valid_names.append(name)
        
        # Create TF-IDF vectorizer if enabled
        if self.use_tfidf and len(valid_names) > 1:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    min_df=2, max_df=0.8, stop_words='english', 
                    ngram_range=(1, 2)  # Include both unigrams and bigrams
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_names)
                
                # Store embeddings
                for i, idx in enumerate(valid_records):
                    self.name_embeddings[idx] = self.tfidf_matrix[i]
                    
                logger.info(f"[{self.name}] Generated TF-IDF embeddings with {self.tfidf_matrix.shape[1]} features")
            except ImportError:
                logger.warning(f"[{self.name}] sklearn not available, falling back to simpler embeddings")
                self.use_tfidf = False
        
        # Fallback to simpler approach if TF-IDF is not enabled or failed
        if not self.use_tfidf or not hasattr(self, 'tfidf_matrix'):
            # Fallback to simpler approach: word set representation
            all_words = set()
            for name in valid_names:
                words = set(name.lower().split())
                all_words.update(words)
            
            self.word_to_idx = {word: i for i, word in enumerate(all_words)}
            
            # Create embeddings using binary word presence
            for i, idx in enumerate(valid_records):
                name = valid_names[i]
                words = set(name.lower().split())
                
                # Create a simple one-hot style embedding
                embedding = np.zeros(len(self.word_to_idx))
                for word in words:
                    if word in self.word_to_idx:
                        embedding[self.word_to_idx[word]] = 1
                
                self.name_embeddings[idx] = embedding
            
            logger.info(f"[{self.name}] Generated binary word embeddings with {len(self.word_to_idx)} dimensions")
        
        logger.info(f"[{self.name}] Generated embeddings for {len(self.name_embeddings)} company names")
    
    def _calculate_embedding_similarity(self, idx1: int, idx2: int) -> float:
        """
        Calculate cosine similarity between name embeddings.
        
        Args:
            idx1: First record index
            idx2: Second record index
            
        Returns:
            Similarity score (0-1)
        """
        if idx1 not in self.name_embeddings or idx2 not in self.name_embeddings:
            return 0.0
        
        emb1 = self.name_embeddings[idx1]
        emb2 = self.name_embeddings[idx2]
        
        # Calculate cosine similarity
        if self.use_tfidf and hasattr(self, 'tfidf_matrix'):
            # TF-IDF sparse matrix
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(emb1, emb2)[0][0]
            except:
                # Fallback if sklearn is not available
                dot_product = (emb1 * emb2).sum()
                norm1 = np.sqrt((emb1 ** 2).sum())
                norm2 = np.sqrt((emb2 ** 2).sum())
                sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        else:
            # Dense vector similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            sim = dot_product / (norm1 * norm2)
        
        return float(sim)
    
    def _find_semantic_matches(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Find semantically similar pairs of records.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched record pairs
        """
        logger.info(f"[{self.name}] Finding semantic matches...")
        
        matches = []
        
        # Use blocking to reduce comparison space
        blocks = defaultdict(list)
        
        # Domain blocking
        domain_groups = df[df['normalized_domain'] != ''].groupby('normalized_domain').indices
        for domain, indices in domain_groups.items():
            if domain and len(indices) > 1:
                blocks[f"domain_{domain}"] = list(indices)
        
        # Country blocking
        country_groups = df[df['main_country_code'] != ''].groupby('main_country_code').indices
        for country, indices in country_groups.items():
            if country and len(indices) <= 1000:  # Skip very large country groups
                blocks[f"country_{country}"] = list(indices)
        
        # Industry blocking
        if 'naics_2022_primary_code' in df.columns:
            industry_groups = df[df['naics_2022_primary_code'].notna()].groupby('naics_2022_primary_code').indices
            for industry, indices in industry_groups.items():
                if len(indices) > 1 and len(indices) <= 500:
                    blocks[f"industry_{industry}"] = list(indices)
        
        # Process each block
        for block_key, indices in blocks.items():
            # Skip very large blocks
            if len(indices) > 200:
                continue
                
            # Compare each pair in the block
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    
                    # Calculate semantic similarity
                    semantic_sim = self._calculate_embedding_similarity(idx1, idx2)
                    
                    # Also calculate string similarity for comparison
                    rec1 = self._extract_record_data(df.iloc[idx1])
                    rec2 = self._extract_record_data(df.iloc[idx2])
                    string_sim = fuzz.token_sort_ratio(rec1['normalized_name'], rec2['normalized_name']) / 100.0
                    
                    # Combine similarities
                    combined_sim = 0.7 * semantic_sim + 0.3 * string_sim
                    
                    # Add extra weight to matches within same domain
                    if 'domain_' in block_key and combined_sim > 0.6:
                        combined_sim = min(1.0, combined_sim + 0.1)
                        
                    # Add as match if combined similarity is high enough
                    if combined_sim > self.similarity_threshold:
                        matches.append((idx1, idx2, combined_sim))
        
        logger.info(f"[{self.name}] Found {len(matches)} semantic matches")
        return matches
